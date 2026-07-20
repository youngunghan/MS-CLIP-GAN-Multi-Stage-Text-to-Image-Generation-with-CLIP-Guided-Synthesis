#!/usr/bin/env python3
"""Prompt-swap sensitivity: the primary, loss-uncontaminated conditioning metric.

Every training entry point in this repository trains G with
``--use_contrastive_loss`` (CLIP image-text similarity), so ``clip_score`` is
partly a training objective and cannot alone prove that text conditioning
works. Prompt-swap sensitivity is not optimised by any loss: hold the noise
``z`` and the conditioning-augmentation ``epsilon`` FIXED and vary only the
caption, then measure how much the 256px output actually moves. A generator
that ignores its caption produces near-identical images regardless of which
caption is used, no matter how good its clip_score is.

Reference scale (already measured, printed alongside every run so the
numbers below are readable in context; not recomputed here):
    CLIP ceiling (real image vs. its own caption)      = 0.2722
    CLIP null    (real image vs. a shuffled caption)   = 0.2025
    usable CLIP range                                   = 0.0697
    current collapsed model: matched-shuffled CLIP gap  = 0.0017  (2.5% of range)
    current collapsed model: pixel sensitivity          ~= 2/255
    two DIFFERENT checkpoints on the SAME caption differ by ~75.7/255

Usage:
    python experiments/prompt_sensitivity.py CKPT_DIR EPOCH DATASET.zip
    python experiments/prompt_sensitivity.py CKPT_DIR EPOCH DATASET.zip \\
        --num-captions 16 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# Make the documented ``python experiments/prompt_sensitivity.py ...`` invocation
# work without relying on the caller to preconfigure PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from config.config import CLIPConfig
from criteria.metric import calculate_clip_score
from networks.generator import Generator
from scripts.checkpoint_config import resolve_checkpoint_model_config
from utils.utils import load_checkpoint, normalize, seed_fix


DEFAULT_NUM_CAPTIONS = 16
# A single seed reports a std of 0.0 from one stochastic draw, which misleads
# (see the --seeds help text below); 3 seeds is cheap (well under a minute for
# the verification run) and is the minimum that yields a non-degenerate spread.
DEFAULT_SEEDS = 3
DEFAULT_SEED = 42
# Pipeline is fixed to CLIP ViT-B/32 (512-dim); see options/base_options.py --clip_model.
CLIP_MODEL_NAME = "ViT-B/32"

# Already-measured reference scale (see module docstring). Printed for context,
# never recomputed here. Measured on data/testset_sub.zip specifically (510
# usable captions); see REFERENCE_DATASET_* below and reference_scale_applies().
REFERENCE_CLIP_CEILING = 0.2722
REFERENCE_CLIP_NULL = 0.2025
REFERENCE_CLIP_RANGE = REFERENCE_CLIP_CEILING - REFERENCE_CLIP_NULL
REFERENCE_COLLAPSED_PIXEL_SENSITIVITY = 2.0  # /255
REFERENCE_COLLAPSED_GAP_PCT = 2.5  # % of REFERENCE_CLIP_RANGE
REFERENCE_CROSS_CHECKPOINT_PIXEL_DIFF = 75.7  # /255, same caption, different checkpoints

# Fingerprint of the dataset REFERENCE_CLIP_RANGE was measured on. ``dataset_zip``
# is a free-form CLI argument, so without this check pointing the tool at any other
# zip would silently print an authoritative-looking "% of range" figure computed
# against a range that was never measured on that data.
REFERENCE_DATASET_BASENAME = "testset_sub.zip"
REFERENCE_DATASET_SAMPLE_COUNT = 510


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_dir", help="directory containing epoch_<E>_Gen.pt")
    parser.add_argument("epoch", type=int, help="checkpoint epoch to probe")
    parser.add_argument("dataset_zip", help="preprocessed dataset ZIP holding dataset.json.clip_txt_features")
    parser.add_argument(
        "--num-captions", type=int, default=DEFAULT_NUM_CAPTIONS,
        help=f"number of distinct test captions to swap between (default: {DEFAULT_NUM_CAPTIONS})",
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help=(
            "number of independent (z, CA-epsilon) draws to average over "
            f"(default: {DEFAULT_SEEDS}). The effect is seed-noisy; a single seed misleads."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base RNG seed")
    parser.add_argument("--device", default=None, help="override device (default: cuda if available, else cpu)")
    parser.add_argument("--output", type=Path, default=None, help="optional path to also write results as JSON")
    args = parser.parse_args(argv)
    if args.num_captions < 2:
        parser.error("--num-captions must be >= 2 (need at least a pair to compare)")
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    return args


def load_test_captions(
    dataset_zip: Path, num_captions: int
) -> Tuple[torch.Tensor, List[str], int]:
    """Return the first ``num_captions`` (by sorted image name) stored caption embeddings.

    Only ``dataset.json``'s ``clip_txt_features`` is read (the test set stores
    embeddings, not raw text) -- unlike ``eval_curve.load_test_data`` this does not
    decode any PNGs, since prompt-swap sensitivity never looks at real images.

    Also returns the TOTAL number of usable captions found in the dataset (before
    slicing down to ``num_captions``), which callers use to fingerprint the dataset
    against the one REFERENCE_CLIP_RANGE was measured on (see
    ``reference_scale_applies``).
    """
    if not dataset_zip.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_zip}")

    with zipfile.ZipFile(dataset_zip) as archive:
        try:
            metadata = json.loads(archive.read("dataset.json"))
        except KeyError as exc:
            raise ValueError(f"{dataset_zip} has no dataset.json") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{dataset_zip}/dataset.json is invalid JSON") from exc

        entries = metadata.get("clip_txt_features")
        if not isinstance(entries, list):
            raise ValueError("dataset.json.clip_txt_features must be a list")

        first_caption_by_name: Dict[str, torch.Tensor] = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("each clip_txt_features entry must be [filename, embeddings]")
            filename, embeddings = entry
            if not isinstance(filename, str) or not embeddings:
                continue
            tensor = torch.as_tensor(embeddings, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
                raise ValueError(f"invalid text embedding shape for {filename}: {tuple(tensor.shape)}")
            # Deliberately select the first stored caption for each image, matching
            # eval_curve.py / scripts/eval.py's convention.
            first_caption_by_name[filename] = normalize(tensor, dim=1)[0]

    names = sorted(first_caption_by_name)
    if len(names) < num_captions:
        raise ValueError(
            f"{dataset_zip} has only {len(names)} usable captions; requested {num_captions}"
        )
    selected = names[:num_captions]
    captions = torch.stack([first_caption_by_name[name] for name in selected])
    return captions, selected, len(names)


def reference_scale_applies(dataset_zip: Path, total_available_captions: int) -> bool:
    """Whether REFERENCE_CLIP_RANGE (measured once on data/testset_sub.zip, N=510) is
    valid for this dataset.

    Fingerprints on the dataset ZIP basename plus its total usable-caption count
    (not just the requested ``--num-captions`` subset). This is not a
    cryptographic guarantee -- a same-named, same-sized-but-different dataset
    would still pass -- but it is enough to catch the common mistake of pointing
    the tool at a different split (e.g. ``testset.zip``, which has a different
    caption count) and getting a "% of range" figure that looks authoritative but
    was never actually measured on that data.
    """
    return (
        dataset_zip.name == REFERENCE_DATASET_BASENAME
        and total_available_captions == REFERENCE_DATASET_SAMPLE_COUNT
    )


def load_generator(checkpoint_dir: Path, epoch: int, device: str) -> Tuple[Generator, Dict[str, Any]]:
    """Load the generator through the same checkpoint-config resolution path scripts/infer.py uses."""
    _metadata, config = resolve_checkpoint_model_config(str(checkpoint_dir), epoch)
    generator = Generator(
        config["g_in_chans"], config["g_out_chans"], config["noise_dim"],
        config["condition_dim"], config["clip_embedding_dim"], config["num_stage"],
        device, config["conditioning_activation"],
    ).to(device)
    load_args = types.SimpleNamespace(is_train=False, new_optim=False)
    load_checkpoint(
        load_args, generator, [None] * int(config["num_stage"]),
        optim_g=None, optim_d_lst=[None] * int(config["num_stage"]),
        checkpoint_path=str(checkpoint_dir), epoch=epoch,
    )
    generator.eval()
    return generator, config


@torch.no_grad()
def generate_holding_noise_fixed(
    generator: Generator, captions: torch.Tensor, noise_dim: int, seed: int, device: str,
) -> torch.Tensor:
    """Generate one 256px image per caption with z AND the CA epsilon held fixed.

    ``ConditioningAugmention.forward`` draws its reparameterization epsilon via
    ``torch.randn_like`` internally (it is not an argument the caller can pass in).
    The only way to force it identical across captions is to reset the RNG to the
    exact same state immediately before every single-caption forward call, so that
    draw is the very first (and only) RNG consumption after the reset. Generating
    one caption at a time (batch size 1) is what makes that reset safe: batching
    all N captions in one forward call would give torch.randn_like a [N, cond_dim]
    tensor and draw N independent rows, which is exactly the per-caption noise we
    are trying to eliminate.
    """
    seed_fix(seed)
    z = torch.randn(1, noise_dim, device=device)

    images = []
    for caption in captions:
        caption_batch = caption.unsqueeze(0).to(device).float()
        seed_fix(seed)  # reset so cond_aug's epsilon draw is bit-identical every time
        fake_images, _mu, _log_sigma = generator(caption_batch, z)
        images.append(fake_images[-1].squeeze(0))
    return torch.stack(images, dim=0)


def mean_pairwise_abs_pixel_diff(images_0_255: torch.Tensor) -> float:
    """Mean absolute pixel difference (0-255 scale) over all unordered image pairs."""
    n = images_0_255.shape[0]
    if n < 2:
        return float("nan")
    total = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += (images_0_255[i] - images_0_255[j]).abs().mean().item()
            pair_count += 1
    return total / pair_count


def clip_matched_shuffled_gap(
    images_neg1_1: torch.Tensor, captions: torch.Tensor, clip_model,
) -> Tuple[float, float]:
    """Return (matched clip_score, shuffled clip_score) for the same image batch."""
    matched = calculate_clip_score(images_neg1_1, captions, clip_model)
    shuffled_captions = torch.roll(captions, shifts=1, dims=0)
    shuffled = calculate_clip_score(images_neg1_1, shuffled_captions, clip_model)
    return matched, shuffled


def run_one_seed(
    generator: Generator,
    clip_model,
    captions: torch.Tensor,
    noise_dim: int,
    seed: int,
    device: str,
    reference_applicable: bool,
) -> Dict[str, float]:
    images_neg1_1 = generate_holding_noise_fixed(generator, captions, noise_dim, seed, device)
    images_0_255 = (images_neg1_1.clamp(-1, 1) + 1) * 127.5

    pixel_sensitivity = mean_pairwise_abs_pixel_diff(images_0_255)
    matched, shuffled = clip_matched_shuffled_gap(images_neg1_1, captions.to(device).float(), clip_model)
    gap_raw = matched - shuffled
    # Only meaningful when this run's dataset is fingerprinted as the one
    # REFERENCE_CLIP_RANGE was measured on -- otherwise the raw gap is still
    # reported, but the "% of range" figure is suppressed (nan) rather than
    # printing an authoritative-looking number computed against a range that was
    # never measured on this data. See reference_scale_applies().
    gap_pct = gap_raw / REFERENCE_CLIP_RANGE * 100.0 if reference_applicable else float("nan")
    return {
        "pixel_sensitivity": pixel_sensitivity,
        "clip_matched": matched,
        "clip_shuffled": shuffled,
        "clip_gap_raw": gap_raw,
        "clip_gap_pct_of_range": gap_pct,
    }


def summarize(per_seed: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in per_seed[0]:
        values = [entry[key] for entry in per_seed]
        if any(math.isnan(value) for value in values):
            # e.g. clip_gap_pct_of_range when the reference range does not apply
            # to this dataset (see run_one_seed): statistics.pstdev raises on nan
            # input rather than propagating it, so short-circuit explicitly.
            summary[key] = {"mean": float("nan"), "std": float("nan")}
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def print_report(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    per_seed: List[Dict[str, float]],
    summary: Dict[str, Dict[str, float]],
    reference_applicable: bool,
) -> None:
    print(
        f"checkpoint: {args.checkpoint_dir} epoch {args.epoch}   "
        f"conditioning={config['conditioning_activation']}   alignment={config['alignment_mode']}",
        flush=True,
    )
    print(f"captions: {args.num_captions}   seeds: {args.seeds}", flush=True)
    if args.seeds == 1:
        print(
            "  WARNING: --seeds=1 reports a std of 0.0 from a single stochastic "
            "draw; the effect is seed-noisy and one seed misleads. Prefer --seeds "
            f">= {DEFAULT_SEEDS}.",
            flush=True,
        )
    if not reference_applicable:
        print(
            f"  WARNING: dataset does not match the one REFERENCE_CLIP_RANGE was "
            f"measured on ({REFERENCE_DATASET_BASENAME}, "
            f"{REFERENCE_DATASET_SAMPLE_COUNT} captions); '% of range' figures "
            f"below are suppressed (nan). The raw CLIP gap is still meaningful.",
            flush=True,
        )
    print("", flush=True)
    for i, entry in enumerate(per_seed):
        gap_pct_str = (
            f"{entry['clip_gap_pct_of_range']:+.1f}% of range"
            if reference_applicable
            else "% of range: n/a for this dataset"
        )
        print(
            f"  seed {args.seed + i:4d}:  pixel_sensitivity = {entry['pixel_sensitivity']:6.2f}/255   "
            f"CLIP matched = {entry['clip_matched']:.4f}   shuffled = {entry['clip_shuffled']:.4f}   "
            f"gap = {entry['clip_gap_raw']:+.4f} ({gap_pct_str})",
            flush=True,
        )
    print("", flush=True)
    ps = summary["pixel_sensitivity"]
    gap_raw = summary["clip_gap_raw"]
    print(
        f"mean pixel_sensitivity = {ps['mean']:.2f} ± {ps['std']:.2f} /255   "
        f"({args.seeds} seed{'s' if args.seeds != 1 else ''})",
        flush=True,
    )
    if reference_applicable:
        gap_pct = summary["clip_gap_pct_of_range"]
        print(
            f"mean CLIP matched-shuffled gap = {gap_raw['mean']:+.4f} ± {gap_raw['std']:.4f}   "
            f"= {gap_pct['mean']:+.1f}% ± {gap_pct['std']:.1f}% of the usable CLIP range",
            flush=True,
        )
    else:
        print(
            f"mean CLIP matched-shuffled gap = {gap_raw['mean']:+.4f} ± {gap_raw['std']:.4f}   "
            f"(% of range not applicable -- dataset does not match the reference measurement)",
            flush=True,
        )
    print("", flush=True)
    print("reference scale (measured separately, not recomputed here):", flush=True)
    print(
        f"  CLIP ceiling={REFERENCE_CLIP_CEILING:.4f}  null={REFERENCE_CLIP_NULL:.4f}  "
        f"range={REFERENCE_CLIP_RANGE:.4f}  (measured on {REFERENCE_DATASET_BASENAME}, "
        f"{REFERENCE_DATASET_SAMPLE_COUNT} captions)",
        flush=True,
    )
    print(
        f"  current collapsed model: pixel_sensitivity ~= {REFERENCE_COLLAPSED_PIXEL_SENSITIVITY:.0f}/255,  "
        f"CLIP gap ~= {REFERENCE_COLLAPSED_GAP_PCT:.1f}% of range",
        flush=True,
    )
    print(
        f"  two different checkpoints on the SAME caption differ by "
        f"{REFERENCE_CROSS_CHECKPOINT_PIXEL_DIFF:.1f}/255",
        flush=True,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    captions, caption_names, total_available = load_test_captions(
        Path(args.dataset_zip), args.num_captions
    )
    reference_applicable = reference_scale_applies(Path(args.dataset_zip), total_available)
    generator, config = load_generator(checkpoint_dir, args.epoch, device)

    clip_model, _ = CLIPConfig.load_clip(CLIP_MODEL_NAME, device)
    clip_model.eval()

    per_seed = [
        run_one_seed(
            generator, clip_model, captions, int(config["noise_dim"]), args.seed + i, device,
            reference_applicable,
        )
        for i in range(args.seeds)
    ]
    summary = summarize(per_seed)
    print_report(args, config, per_seed, summary, reference_applicable)

    result = {
        "checkpoint_dir": str(checkpoint_dir),
        "epoch": args.epoch,
        "num_captions": args.num_captions,
        "caption_names": caption_names,
        "seeds": args.seeds,
        "base_seed": args.seed,
        "model_config": config,
        "per_seed": per_seed,
        "summary": summary,
        "reference_scale_applicable": reference_applicable,
        "reference": {
            "clip_ceiling": REFERENCE_CLIP_CEILING,
            "clip_null": REFERENCE_CLIP_NULL,
            "clip_range": REFERENCE_CLIP_RANGE,
            "collapsed_pixel_sensitivity": REFERENCE_COLLAPSED_PIXEL_SENSITIVITY,
            "collapsed_gap_pct_of_range": REFERENCE_COLLAPSED_GAP_PCT,
            "cross_checkpoint_pixel_diff": REFERENCE_CROSS_CHECKPOINT_PIXEL_DIFF,
            "dataset_basename": REFERENCE_DATASET_BASENAME,
            "dataset_sample_count": REFERENCE_DATASET_SAMPLE_COUNT,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}", flush=True)
    return result


def main(argv: Sequence[str] = None) -> int:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
