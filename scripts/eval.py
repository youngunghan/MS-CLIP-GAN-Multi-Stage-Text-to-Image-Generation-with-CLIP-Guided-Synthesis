import os
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
                out, os.path.join(str(args.result_path), f"batch_{i}_size_{size}.png")
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

    clip_model, _ = CLIPConfig.load_clip(args.clip_model, device)
    clip_model.eval()

    G = Generator(
        args.g_in_chans, args.g_out_chans, args.noise_dim,
        args.condition_dim, args.clip_embedding_dim,
        args.num_stage, device
    ).to(device)

    load_checkpoint(
        args, G, [None for _ in range(args.num_stage)],
        optim_g=None, optim_d_lst=[None for _ in range(args.num_stage)],
        checkpoint_path=args.checkpoint_path,
        epoch=args.load_epoch
    )
    G.eval()

    eval_dataset = MM_CelebA(args.eval_data_path, args.num_stage)
    eval_loader = get_dataloader(args=args, dataset=eval_dataset, is_train=False)

    mkdirs(str(args.result_path))
    metrics = evaluate(args, G, clip_model, device, eval_loader, max_batches=args.max_batches)

    if metrics is not None:
        save_metrics_to_csv(args, metrics)
        print(metrics)


if __name__ == "__main__":
    main()
