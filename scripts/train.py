import time
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

from config.config import CLIPConfig
from dataset.dataloader import MM_CelebA, get_dataloader
from networks.discriminator import Discriminator
from networks.generator import Generator
from utils.utils import *
from utils.utils import _unwrap  # underscore-prefixed names are not pulled in by `import *`
from criteria.loss import *
from trainer import train_step
from options.train_options import TrainOptions

# torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = TrainOptions().parse()

    # GPU 설정: --gpu_ids를 실제 CUDA 디바이스 인덱스로 사용한다.
    # (train.sh는 CUDA_VISIBLE_DEVICES를 설정하지 않으므로 --gpu_ids가 곧 물리 인덱스이며,
    #  python scripts/train.py --gpu_ids 1 처럼 직접 실행해도 의도대로 동작한다.)
    if torch.cuda.is_available() and len(args.gpu_ids) > 0:
        device_ids = args.gpu_ids
        torch.cuda.set_device(device_ids[0])
        device = torch.device(f"cuda:{device_ids[0]}")
        print(f"Using GPU(s): {device_ids}")
    else:
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. Using CPU.")
        device = torch.device("cpu")
        device_ids = []
        args.gpu_ids = []

    # base_options.parse()가 이미 checkpoint_path를 <checkpoints>/<name>/ckpt로 설정했으므로
    # 여기서 <name>을 다시 덧붙이지 않는다(이중 중첩 방지).
    os.makedirs(str(args.checkpoint_path), exist_ok=True)
    os.makedirs(str(args.result_path), exist_ok=True)

    # Tensorboard writer 초기화
    log_dir = os.path.join('runs', args.name)
    writer = SummaryWriter(log_dir)

    lr = args.learning_rate
    num_epochs = args.num_epochs

    print("Loading dataset")
    train_dataset = MM_CelebA(args.data_path, args.num_stage)
    train_loader = get_dataloader(args=args, dataset=train_dataset, is_train=True)
    print("finish")

    # 모델 초기화
    G = Generator(args.g_in_chans, args.g_out_chans, args.noise_dim, args.condition_dim,
                 args.clip_embedding_dim, args.num_stage, device).to(device)
    G.apply(weight_init)

    # Multi-GPU 설정
    if len(device_ids) > 1:
        print(f"Using DataParallel with {len(device_ids)} GPUs")
        G = nn.DataParallel(G, device_ids=device_ids)

    D_lst = []
    for curr_stage in range(args.num_stage):
        D = Discriminator(args.g_out_chans, args.d_in_chans, args.d_out_chans,
                         args.condition_dim, args.clip_embedding_dim, curr_stage, device).to(device)
        if len(device_ids) > 1:
            D = nn.DataParallel(D, device_ids=device_ids)
        D.apply(weight_init)
        D_lst.append(D)

    # Optimizer 설정 (TTUR: D can use a separate, typically smaller LR via --d_lr)
    d_lr = args.d_lr if args.d_lr is not None and args.d_lr > 0 else lr
    optim_g = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d_lst = [Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999)) for D in D_lst]
    if args.use_ema or d_lr != lr or args.real_label_smooth != 1.0:
        print(f"Stability levers: use_ema={args.use_ema} ema_decay={args.ema_decay} "
              f"d_lr={d_lr} (G lr={lr}) real_label_smooth={args.real_label_smooth}")

    # Learning rate schedulers (resume보다 먼저 생성해야 상태를 복구할 수 있다)
    scheduler_g = CosineAnnealingLR(optim_g, T_max=num_epochs)
    scheduler_d_lst = [CosineAnnealingLR(optim_d, T_max=num_epochs)
                      for optim_d in optim_d_lst]

    # 체크포인트 로드 (optimizer + scheduler 상태까지 복구)
    if args.resume_checkpoint_path is not None and args.resume_epoch != -1:
        epoch, num_stage = load_checkpoint(args, G, D_lst, optim_g, optim_d_lst,
                                         args.resume_checkpoint_path, args.resume_epoch,
                                         scheduler_g=scheduler_g, scheduler_d_lst=scheduler_d_lst)
        print('Resumed from saved checkpoint')

    # EMA generator (optional): a temporal average of G, initialised from the current
    # (possibly resumed) weights. Used for sampling and checkpointing when --use_ema.
    G_ema = None
    if args.use_ema:
        G_ema = Generator(args.g_in_chans, args.g_out_chans, args.noise_dim, args.condition_dim,
                          args.clip_embedding_dim, args.num_stage, device).to(device)
        G_ema.load_state_dict(_unwrap(G).state_dict())
        for p in G_ema.parameters():
            p.requires_grad_(False)
        G_ema.eval()
        print(f'EMA generator enabled (decay={args.ema_decay})')

    loss_fn = BCELoss()
    clip_model, _ = CLIPConfig.load_clip(args.clip_model, device)
    # CLIP is used only to guide the generator; freeze it so no spurious gradients/updates occur.
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    for epoch in range(args.resume_epoch + 1, num_epochs):
        print(f"Epoch: {epoch} start")
        start_time = time.time()

        d_loss, g_loss, txt_feature = train_step(
            train_loader, args.noise_dim, G, D_lst, optim_g, optim_d_lst,
            loss_fn, args.num_stage, args.use_uncond_loss, args.use_contrastive_loss,
            args.use_mixed_loss, clip_model, gamma=5, lam=10,
            report_interval=args.report_interval, device=device,
            epoch=epoch, writer=writer,
            g_ema=G_ema, ema_decay=args.ema_decay, real_label_smooth=args.real_label_smooth,
            d_update_every=args.d_update_every
        )

        end_time = time.time()
        print(f"Epoch: {epoch} \t d_loss: {d_loss:.4f} \t g_loss: {g_loss:.4f} \t esti. time: {(end_time - start_time):.2f}s")

        # Scheduler step (end of epoch) BEFORE saving, so the checkpoint stores the
        # post-epoch scheduler state and resume continues at the correct LR.
        scheduler_g.step()
        for scheduler_d in scheduler_d_lst:
            scheduler_d.step()

        # 샘플링 및 이미지 저장 + 체크포인트
        if epoch % args.save_freq == 0:  # save_freq 마다 이미지/체크포인트 저장
            # When EMA is on, sample AND checkpoint the EMA generator (it is always in eval mode);
            # otherwise toggle the live G to eval for sampling and back to train afterwards.
            sample_G = G_ema if args.use_ema else G
            if not args.use_ema:
                G.eval()
            with torch.no_grad():
                z = torch.randn(txt_feature.shape[0], args.noise_dim).to(device)
                txt_feature = txt_feature.to(device)

                fake_images, _, _ = sample_G(txt_feature, z)
                fake_image = fake_images[-1].detach().cpu()
                epoch_ret = torchvision.utils.make_grid(fake_image, padding=2, normalize=True)
                save_path = os.path.join(str(args.result_path), f"{args.name}_epoch_{epoch}.png")
                torchvision.utils.save_image(epoch_ret, save_path)
            if not args.use_ema:
                G.train()

            # 체크포인트 저장 (EMA 사용 시 EMA 가중치를 Gen.pt로 저장 → eval/infer가 EMA 모델 사용)
            save_checkpoint(args, sample_G, D_lst, optim_g, optim_d_lst, epoch, args.num_stage,
                            scheduler_g=scheduler_g, scheduler_d_lst=scheduler_d_lst)

    writer.close()
