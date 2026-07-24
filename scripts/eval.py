#!/usr/bin/env python3

import os
import hashlib
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision
from utils.utils import *
from config.config import CLIPConfig
from networks.generator import Generator
from options.test_options import TestOptions
from dataset.dataloader import MM_CelebA, get_dataloader
from criteria.metric import (
    calculate_clip_score, to_uint8, build_fid, build_inception_score
)
from scripts.checkpoint_config import apply_checkpoint_model_config


@torch.no_grad()
def evaluate(args, G, clip_model, device, dataloader, max_batches=-1):
    G.eval()
    clip_model.eval()

    # Single metric objects: update across ALL batches, compute ONCE at the end.
    # (Per-batch FID on a handful of images and averaging it is not a valid FID.)
    fid = build_fid(device)
    is_metric = build_inception_score(device)

    clip_score_sum = 0.0
    n_samples = 0

    for i, (real_imgs, _, txt_embedding) in enumerate(dataloader):
        if 0 <= max_batches <= i:
            break

        batch_size = txt_embedding.size(0)
        txt_embedding = normalize(txt_embedding.to(device))

        # Generate images from text embeddings
        z = torch.randn(batch_size, args.noise_dim, device=device)
        fake_images, _, _ = G(txt_embedding, z)

        fake = fake_images[-1]
        real = real_imgs[-1].to(device)

        # Save a few sample images at each stage resolution
        for stage, f in enumerate(fake_images):
            size = 64 * (2 ** stage)
            out = CLIPConfig.denormalize_image(f.detach().cpu())
            torchvision.utils.save_image(
                out,
                os.path.join(
                    str(args.result_path),
                    f"{args.eval_artifact_id}_batch_{i}_size_{size}.png",
                ),
            )

        # Accumulate metrics
        fid.update(to_uint8(real), real=True)
        fid.update(to_uint8(fake), real=False)
        is_metric.update(to_uint8(fake))

        clip_score_sum += calculate_clip_score(fake, txt_embedding, clip_model) * batch_size
        n_samples += batch_size

        if i % args.print_freq == 0:
            print(f"Processed {n_samples} samples")

    if n_samples == 0:
        print("No samples processed.")
        return None

    fid_score = fid.compute().item()
    is_mean, is_std = is_metric.compute()
    metrics = {
        'clip_score': clip_score_sum / n_samples,
        'fid_score': fid_score,
        'inception_score_mean': float(is_mean),
        'inception_score_std': float(is_std),
        'samples_processed': n_samples,
    }
    if not all(
        math.isfinite(float(metrics[key]))
        for key in (
            'clip_score', 'fid_score', 'inception_score_mean',
            'inception_score_std',
        )
    ):
        raise RuntimeError(f'evaluation produced non-finite metrics: {metrics}')

    print("\nEvaluation Results:")
    print(f"Processed {n_samples} samples")
    print(f"CLIP Score: {metrics['clip_score']:.4f}")
    print(f"FID Score: {metrics['fid_score']:.4f}")
    print(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")

    return metrics


def main():
    # print_options=False: don't create an experiment dir / opt.txt under the checkpoint path.
    args = TestOptions().parse(print_options=False)
    seed_fix(args.seed)
    gpu_ids = args.gpu_ids
    device = torch.device(f"cuda:{gpu_ids[0]}") if (torch.cuda.is_available() and len(gpu_ids) > 0) else torch.device("cpu")

    # Resolve architecture and compatibility mode before constructing the model.
    checkpoint_metadata, model_config = apply_checkpoint_model_config(args)

    clip_model, _ = CLIPConfig.load_clip(args.clip_model, device)
    clip_model.eval()

    G = Generator(
        args.g_in_chans, args.g_out_chans, args.noise_dim,
        args.condition_dim, args.clip_embedding_dim,
        args.num_stage, device, args.conditioning_activation,
        deterministic_cond=args.deterministic_cond,
    ).to(device)

    load_checkpoint(
        args, G, [None for _ in range(args.num_stage)],
        optim_g=None, optim_d_lst=[None for _ in range(args.num_stage)],
        checkpoint_path=args.checkpoint_path,
        epoch=args.load_epoch
    )
    G.eval()

    # Model/CLIP construction and checkpoint loading consume RNG. Reset at the
    # same boundary as eval_curve.py so --seed describes the actual sample stream.
    seed_fix(args.seed)
    checkpoint_file = Path(args.checkpoint_path) / f'epoch_{args.load_epoch}_Gen.pt'
    digest = hashlib.sha256()
    with checkpoint_file.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    args.eval_checkpoint_path = str(checkpoint_file.resolve())
    args.eval_checkpoint_sha256 = digest.hexdigest()
    args.eval_checkpoint_format = checkpoint_metadata.get('format_version')
    args.eval_checkpoint_legacy = checkpoint_metadata.get('legacy')
    args.eval_generator_weight_kind = checkpoint_metadata.get('generator_weight_kind')
    args.eval_conditioning_activation = model_config['conditioning_activation']
    args.eval_alignment_mode = model_config['alignment_mode']
    args.eval_artifact_id = (
        f"epoch_{args.load_epoch}_{args.eval_checkpoint_sha256[:12]}"
    )

    eval_dataset = MM_CelebA(args.eval_data_path, args.num_stage)
    eval_loader = get_dataloader(args=args, dataset=eval_dataset, is_train=False)

    mkdirs(str(args.result_path))
    metrics = evaluate(args, G, clip_model, device, eval_loader, max_batches=args.max_batches)

    if metrics is None:
        raise RuntimeError("evaluation processed no samples")
    save_metrics_to_csv(args, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
