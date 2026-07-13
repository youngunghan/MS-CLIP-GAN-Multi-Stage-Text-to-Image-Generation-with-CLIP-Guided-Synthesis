#!/usr/bin/env python3

import warnings
warnings.filterwarnings(action="ignore")

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clip
import torch
import torchvision
from utils.utils import *
from config.config import CLIPConfig
from networks.generator import Generator
from options.test_options import TestOptions
from scripts.checkpoint_config import apply_checkpoint_model_config

@torch.no_grad()
def main():
    # print_options=False: don't create an experiment dir / opt.txt under the checkpoint path.
    args = TestOptions().parse(print_options=False)
    seed_fix(args.seed)
    gpu_ids = args.gpu_ids
    device = torch.device(f"cuda:{gpu_ids[0]}") if (torch.cuda.is_available() and len(gpu_ids) > 0) else torch.device("cpu")

    # Resolve architecture and compatibility mode before constructing the model.
    apply_checkpoint_model_config(args)

    clip_model, _ = CLIPConfig.load_clip(args.clip_model, device)
    clip_model.eval()

    G = Generator(args.g_in_chans, args.g_out_chans, args.noise_dim, args.condition_dim,
                  args.clip_embedding_dim, args.num_stage, device,
                  args.conditioning_activation).to(device)

    # Inference only needs the generator; discriminator checkpoints are not required.
    load_checkpoint(args, G, [None for _ in range(args.num_stage)],
                    optim_g=None, optim_d_lst=[None for _ in range(args.num_stage)],
                    checkpoint_path=args.checkpoint_path, epoch=args.load_epoch)
    G.eval()

    prompt = clip.tokenize([args.prompt]).to(device)
    txt_feature = clip_model.encode_text(prompt)
    z = torch.randn(txt_feature.shape[0], args.noise_dim).to(device)
    txt_feature = normalize(txt_feature.to(device)).type(torch.float32)

    fake_images, _, _ = G(txt_feature, z)

    result_dir = str(args.result_path)
    mkdirs(result_dir)
    for stage, fake in enumerate(fake_images):
        size = 64 * (2 ** stage)  # stage resolution: 64, 128, 256, ...
        img = CLIPConfig.denormalize_image(fake.detach().cpu())
        torchvision.utils.save_image(img, os.path.join(result_dir, f"result_{size}.png"))
    print(f"Saved {len(fake_images)} image(s) to {result_dir}")


if __name__ == "__main__":
    main()
