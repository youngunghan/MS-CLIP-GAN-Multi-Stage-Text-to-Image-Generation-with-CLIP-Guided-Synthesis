#!/usr/bin/env python3
"""Evaluate generator checkpoints with standard FID and Inception Score.

One fake is generated for each test image.  Conditioning uses the first stored
caption embedding for that image (not every caption and not a fixed prompt).

The historical positional CLI remains valid::

    python experiments/eval_curve.py TEST.zip CKPT_DIR auto OUT.json

An evaluation seed can be supplied with ``--seed`` (or as an optional fifth
positional argument for compatibility with early local runners).  The RNG is
reset *after loading every checkpoint*, so a checkpoint receives identical
latent and conditioning-augmentation draws regardless of which other epochs
are included in the same invocation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import zipfile

# Make the documented ``python experiments/eval_curve.py ...`` invocation work
# without relying on the caller to preconfigure PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from networks.generator import Generator
from utils.utils import load_checkpoint, normalize, peek_checkpoint_metadata, seed_fix


DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
LEGACY_MODEL_CONFIG: Dict[str, Any] = {
    "g_in_chans": 1024,
    "g_out_chans": 3,
    "noise_dim": 100,
    "condition_dim": 128,
    "clip_embedding_dim": 512,
    "num_stage": 3,
    "conditioning_activation": "relu",
    "alignment_mode": "legacy_conditioned",
}
MODEL_CONFIG_KEYS = frozenset(LEGACY_MODEL_CONFIG)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute standard 2048-d FID and Inception Score for generator "
            "checkpoints, using the first stored caption per test image."
        )
    )
    parser.add_argument("test_zip", help="preprocessed test dataset ZIP")
    parser.add_argument("checkpoint_dir", help="directory containing epoch_<E>_Gen.pt")
    parser.add_argument("epochs", help='comma-separated epochs (for example "0,5,10") or "auto"')
    parser.add_argument("output", help="result JSON path (existing result schema is preserved)")
    parser.add_argument(
        "positional_seed",
        nargs="?",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        dest="option_seed",
        type=int,
        help=f"evaluation RNG seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args(argv)
    if args.positional_seed is not None and args.option_seed is not None:
        parser.error("specify the seed either as --seed or as the fifth positional argument, not both")
    args.seed = (
        args.option_seed
        if args.option_seed is not None
        else args.positional_seed
        if args.positional_seed is not None
        else DEFAULT_SEED
    )
    return args


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def parse_requested_epochs(epoch_arg: str, checkpoint_dir: Path) -> List[int]:
    if epoch_arg == "auto":
        epochs: List[int] = []
        for path in checkpoint_dir.glob("epoch_*_Gen.pt"):
            parts = path.name.split("_")
            if len(parts) == 3 and parts[0] == "epoch" and parts[2] == "Gen.pt":
                try:
                    epochs.append(int(parts[1]))
                except ValueError:
                    continue
        epochs = sorted(set(epochs))
    else:
        try:
            epochs = [int(value.strip()) for value in epoch_arg.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError(f"invalid epoch list {epoch_arg!r}; expected comma-separated integers or 'auto'") from exc
        epochs = list(dict.fromkeys(epochs))

    if not epochs:
        raise FileNotFoundError(
            f"no generator checkpoints selected in {checkpoint_dir} (epochs={epoch_arg!r})"
        )
    return epochs


def existing_checkpoints(epoch_arg: str, checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    requested = parse_requested_epochs(epoch_arg, checkpoint_dir)
    found: List[Tuple[int, Path]] = []
    missing: List[int] = []
    for epoch in requested:
        path = checkpoint_dir / f"epoch_{epoch}_Gen.pt"
        if path.is_file():
            found.append((epoch, path))
        else:
            missing.append(epoch)
    if epoch_arg != "auto" and missing:
        missing_text = ", ".join(str(epoch) for epoch in missing)
        raise FileNotFoundError(
            "explicitly requested generator checkpoint(s) are missing from "
            f"{checkpoint_dir}: {missing_text}"
        )
    if not found:
        raise FileNotFoundError(
            f"none of the selected generator checkpoints exists in {checkpoint_dir}"
        )
    return found


def load_test_data(test_zip: Path) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    if not test_zip.is_file():
        raise FileNotFoundError(f"test dataset not found: {test_zip}")

    to_tensor = T.Compose(
        [
            T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ]
    )
    with zipfile.ZipFile(test_zip) as archive:
        try:
            metadata = json.loads(archive.read("dataset.json"))
        except KeyError as exc:
            raise ValueError(f"{test_zip} has no dataset.json") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{test_zip}/dataset.json is invalid JSON") from exc

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
            # Deliberately select the first stored caption for each image.
            first_caption_by_name[filename] = normalize(tensor, dim=1)[0]

        png_names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".png") and name in first_caption_by_name
        )
        if not png_names:
            raise ValueError(f"{test_zip} contains no PNG with a usable caption embedding")

        real_images: List[torch.Tensor] = []
        for name in png_names:
            try:
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    real_images.append(to_tensor(image.convert("RGB")))
            except (OSError, ValueError) as exc:
                raise ValueError(f"could not decode test image {name}: {exc}") from exc

    captions = torch.stack([first_caption_by_name[name] for name in png_names])
    reals = torch.stack(real_images)
    return reals, captions, png_names


def _metadata_to_config(raw_metadata: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    config = dict(LEGACY_MODEL_CONFIG)
    raw_config = raw_metadata.get("model_config")
    if isinstance(raw_config, Mapping):
        for key in MODEL_CONFIG_KEYS:
            if key in raw_config and raw_config[key] is not None:
                config[key] = raw_config[key]
        source = "legacy_inferred" if raw_metadata.get("legacy") else "checkpoint_metadata"
    else:
        # Historical checkpoints only stored num_stage.  It is safe to honor that
        # field while using the architecture that produced those checkpoints.
        if raw_metadata.get("num_stage") is not None:
            config["num_stage"] = raw_metadata["num_stage"]
        source = "legacy_defaults"

    integer_keys = (
        "g_in_chans",
        "g_out_chans",
        "noise_dim",
        "condition_dim",
        "clip_embedding_dim",
        "num_stage",
    )
    for key in integer_keys:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid checkpoint model_config.{key}: {value!r}")
    if config["g_out_chans"] != 3:
        raise ValueError(
            f"FID/IS evaluation requires a 3-channel generator; checkpoint declares "
            f"g_out_chans={config['g_out_chans']}"
        )
    return config, source


def read_checkpoint_config(checkpoint_dir: Path, epoch: int) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    metadata = peek_checkpoint_metadata(str(checkpoint_dir), epoch, device="cpu")
    if not isinstance(metadata, Mapping):
        raise TypeError("peek_checkpoint_metadata() must return a mapping")
    config, source = _metadata_to_config(metadata)
    return config, source, dict(metadata)


def make_generator(config: Mapping[str, Any], device: str) -> Generator:
    kwargs: Dict[str, Any] = {
        "in_chans": config["g_in_chans"],
        "out_chans": config["g_out_chans"],
        "noise_dim": config["noise_dim"],
        "cond_dim": config["condition_dim"],
        "clip_emb_dim": config["clip_embedding_dim"],
        "num_stage": config["num_stage"],
        "device": device,
    }
    signature = inspect.signature(Generator)
    # alignment_mode describes the discriminator and is recorded for provenance,
    # but has no effect on generator-only evaluation.
    if "conditioning_activation" in signature.parameters:
        kwargs["conditioning_activation"] = config["conditioning_activation"]
    elif config["conditioning_activation"] != LEGACY_MODEL_CONFIG["conditioning_activation"]:
        raise ValueError(
            f"checkpoint requires conditioning_activation={config['conditioning_activation']!r}, "
            "but this Generator implementation does not support that architecture option"
        )
    return Generator(**kwargs).to(device)


def _uint8_images(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(0, 1) * 255).to(torch.uint8)


def evaluate_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    real: torch.Tensor,
    captions: torch.Tensor,
    config: Mapping[str, Any],
    seed: int,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, float]:
    if captions.shape[1] != config["clip_embedding_dim"]:
        raise ValueError(
            f"test caption embeddings have dimension {captions.shape[1]}, but checkpoint "
            f"expects {config['clip_embedding_dim']}"
        )

    generator = fid = inception = None
    try:
        generator = make_generator(config, device)
        load_args = types.SimpleNamespace(is_train=False, new_optim=False)
        load_checkpoint(
            load_args,
            generator,
            [None] * int(config["num_stage"]),
            optim_g=None,
            optim_d_lst=[None] * int(config["num_stage"]),
            checkpoint_path=str(checkpoint_dir),
            epoch=epoch,
        )
        generator.eval()

        # Generator construction and checkpoint loading can consume RNG.  Reset here
        # for every epoch so the generated sample stream is checkpoint-order invariant.
        seed_fix(seed)

        fid = FrechetInceptionDistance(normalize=False).to(device)
        inception = InceptionScore(normalize=False).to(device)
        with torch.no_grad():
            for start in range(0, len(real), batch_size):
                fid.update(_uint8_images(real[start : start + batch_size].to(device)), real=True)
            for start in range(0, len(captions), batch_size):
                caption_batch = captions[start : start + batch_size].to(device).float()
                noise = torch.randn(
                    caption_batch.size(0), int(config["noise_dim"]), device=device
                )
                images, _, _ = generator(caption_batch, noise)
                fake_uint8 = _uint8_images((images[-1].clamp(-1, 1) + 1) / 2)
                fid.update(fake_uint8, real=False)
                inception.update(fake_uint8)

        fid_value = float(fid.compute())
        is_mean, is_std = (float(value) for value in inception.compute())
        values = (fid_value, is_mean, is_std)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"epoch {epoch} produced non-finite metrics: {values}")
        return {"fid": fid_value, "is_mean": is_mean, "is_std": is_std}
    finally:
        del generator, fid, inception
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    for distribution, key in (("torchmetrics", "torchmetrics"), ("Pillow", "pillow"), ("numpy", "numpy")):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _git_state(repo_root: Path) -> Tuple[Optional[str], Optional[bool]]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _hardware_state(device: str) -> Dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    driver_version = None
    try:
        driver_version = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0].strip()
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError):
        pass
    return {
        "device": device,
        "gpu_names": gpu_names,
        "cuda_driver": driver_version,
        "cudnn": torch.backends.cudnn.version(),
    }


def build_provenance(
    *,
    test_zip: Path,
    sample_count: int,
    caption_dimension: int,
    checkpoints: Sequence[Tuple[int, Path]],
    checkpoint_details: Mapping[str, Mapping[str, Any]],
    seed: int,
    device: str,
    epoch_arg: str,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = _git_state(repo_root)
    return {
        "schema_version": 1,
        "seed": seed,
        "sample_count": sample_count,
        "caption_selection": "first_stored_caption_per_image",
        "data": {
            "path": str(test_zip.resolve()),
            "sha256": sha256_file(test_zip),
        },
        "checkpoints": {
            str(epoch): {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                **checkpoint_details[str(epoch)],
            }
            for epoch, path in checkpoints
        },
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hardware": _hardware_state(device),
        "package_versions": _package_versions(),
        "config": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "caption_embedding_dimension": caption_dimension,
            "device": device,
            "epoch_argument": epoch_arg,
            "evaluated_epochs": [epoch for epoch, _ in checkpoints],
            "fid_feature_dimension": 2048,
            "inception_input": "uint8_rgb",
        },
    }


def run(args: argparse.Namespace) -> Tuple[Path, Path]:
    test_zip = Path(args.test_zip)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)
    provenance_path = Path(f"{output_path}.provenance.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = existing_checkpoints(args.epochs, checkpoint_dir)
    real, captions, image_names = load_test_data(test_zip)
    print(
        f"test set: {len(image_names)} images; conditioning=first stored caption per image",
        flush=True,
    )

    results: Dict[str, Dict[str, float]] = {}
    details: Dict[str, Dict[str, Any]] = {}
    for epoch, _checkpoint_path in checkpoints:
        config, config_source, metadata = read_checkpoint_config(checkpoint_dir, epoch)
        metric = evaluate_checkpoint(
            checkpoint_dir,
            epoch,
            real,
            captions,
            config,
            args.seed,
            device,
        )
        results[str(epoch)] = metric
        details[str(epoch)] = {
            "format_version": metadata.get("format_version"),
            "legacy": metadata.get("legacy"),
            "generator_weight_kind": metadata.get("generator_weight_kind"),
            "rng_state_available": metadata.get("rng_state_available"),
            "model_config": config,
            "model_config_source": config_source,
            "training_config": metadata.get("training_config"),
            "training_provenance": metadata.get("training_provenance"),
            "schedule_config": metadata.get("schedule_config"),
            "checkpoint_epoch": metadata.get("epoch"),
        }
        print(
            f"epoch {epoch:3d}:  FID(2048) = {metric['fid']:8.2f}   "
            f"IS = {metric['is_mean']:.3f} ± {metric['is_std']:.3f}",
            flush=True,
        )

    if not results:
        raise RuntimeError("evaluation produced no results")

    provenance = build_provenance(
        test_zip=test_zip,
        sample_count=len(image_names),
        caption_dimension=int(captions.shape[1]),
        checkpoints=checkpoints,
        checkpoint_details=details,
        seed=args.seed,
        device=device,
        epoch_arg=args.epochs,
    )
    # Write the result first, then bind its exact bytes into the sidecar. A crash
    # between the two leaves an obviously incomplete result (no valid sidecar),
    # never a provenance file that silently authenticates different metrics.
    atomic_json_dump(results, output_path)
    provenance["result_artifact"] = {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
    }
    atomic_json_dump(provenance, provenance_path)
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {provenance_path}", flush=True)
    return output_path, provenance_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
