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
from trainer import train_step, warmup_training_losses
from options.train_options import TrainOptions

# torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = TrainOptions().parse()

    # Resolve the scheduler phase before constructing models/optimizers. An exact
    # resume inside a prior --new_optim extension must recreate that phase's saved
    # T_max (for example 50 for epochs [150, 200)), not the global num_epochs=200.
    resume_metadata = None
    if args.resume_checkpoint_path is not None and args.resume_epoch != -1:
        resume_metadata = peek_checkpoint_metadata(
            args.resume_checkpoint_path, args.resume_epoch, device='cpu'
        )
    cosine_horizon = scheduler_horizon(
        args.num_epochs,
        resume_epoch=args.resume_epoch,
        new_optim=args.new_optim,
        schedule_config=(
            resume_metadata.get('schedule_config')
            if resume_metadata is not None else None
        ),
    )

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

    # Hash data and all training-relevant Python sources once. Reusing this static
    # payload in every checkpoint avoids multi-GB dataset rehashing at save time.
    print('Computing training provenance (dataset/source SHA-256)')
    args.training_provenance = build_training_provenance(
        args.data_path, device_ids=device_ids
    )
    print(
        'Training provenance: data='
        f'{args.training_provenance["dataset"]["sha256"][:12]} '
        'source='
        f'{args.training_provenance["source"]["sha256"][:12]}'
    )

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
                 args.clip_embedding_dim, args.num_stage, device,
                 conditioning_activation=args.conditioning_activation).to(device)
    G.apply(weight_init)

    # Multi-GPU 설정
    if len(device_ids) > 1:
        print(f"Using DataParallel with {len(device_ids)} GPUs")
        G = nn.DataParallel(G, device_ids=device_ids)

    D_lst = []
    for curr_stage in range(args.num_stage):
        D = Discriminator(args.g_out_chans, args.d_in_chans, args.d_out_chans,
                         args.condition_dim, args.clip_embedding_dim, curr_stage, device,
                         alignment_mode=args.alignment_mode).to(device)
        if len(device_ids) > 1:
            D = nn.DataParallel(D, device_ids=device_ids)
        D.apply(weight_init)
        D_lst.append(D)

    # Optimizer 설정 (TTUR: D can use a separate, typically smaller LR via --d_lr)
    d_lr = args.d_lr if args.d_lr is not None and args.d_lr > 0 else lr
    optim_g = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d_lst = [Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999)) for D in D_lst]
    if (args.use_ema or d_lr != lr or args.real_label_smooth != 1.0
            or args.d_update_every != 1 or args.use_diffaugment):
        print(f"Stability levers: use_ema={args.use_ema} ema_decay={args.ema_decay} "
              f"d_lr={d_lr} (G lr={lr}) real_label_smooth={args.real_label_smooth} "
              f"d_update_every={args.d_update_every} use_diffaugment={args.use_diffaugment} "
              f"diffaugment_policy={args.diffaugment_policy if args.use_diffaugment else '-'}")

    # Learning rate schedulers (resume보다 먼저 생성해야 상태를 복구할 수 있다)
    scheduler_g = CosineAnnealingLR(optim_g, T_max=cosine_horizon)
    scheduler_d_lst = [CosineAnnealingLR(optim_d, T_max=cosine_horizon)
                      for optim_d in optim_d_lst]
    if args.new_optim:
        print(f'New optimizer schedule spans {cosine_horizon} remaining epoch(s)')

    # EMA generator (optional): a temporal average of G. Built BEFORE resume so that
    # load_checkpoint can restore the saved EMA weights into it (EMA checkpoints keep
    # the EMA weights in Gen.pt and the live training weights in Gen_raw.pt).
    G_ema = None
    if args.use_ema:
        G_ema = Generator(args.g_in_chans, args.g_out_chans, args.noise_dim, args.condition_dim,
                          args.clip_embedding_dim, args.num_stage, device,
                          conditioning_activation=args.conditioning_activation).to(device)
        G_ema.load_state_dict(_unwrap(G).state_dict())
        for p in G_ema.parameters():
            p.requires_grad_(False)
        G_ema.eval()
        print(f'EMA generator enabled (decay={args.ema_decay})')

    # Construct/freeze CLIP before restoring the checkpoint RNG. Model construction
    # may consume random numbers; restoring after all model initialization is what
    # makes the next epoch's sampler/noise sequence exact.
    loss_fn = BCELoss()
    clip_model, _ = CLIPConfig.load_clip(args.clip_model, device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    if args.use_mixed_loss:
        # VGG construction/weight loading may consume CPU RNG. Warm the lazy cache
        # before load_checkpoint restores RNG, so resumed and uninterrupted runs
        # enter the next batch from the same recorded random state.
        warmup_training_losses(args.use_mixed_loss, device)

    # 체크포인트 로드 (optimizer + scheduler + compatibility mode + RNG 상태까지 복구).
    # path/epoch 중 하나만 지정하면 아무것도 로드하지 않은 채 resume_epoch+1부터 도는
    # 잘못된 런이 조용히 만들어지므로, 반쪽 지정은 즉시 에러로 막는다.
    if (args.resume_checkpoint_path is None) != (args.resume_epoch == -1):
        raise ValueError(
            '--resume_checkpoint_path and --resume_epoch must be given together to resume '
            f'(got path={args.resume_checkpoint_path}, epoch={args.resume_epoch})')
    if args.resume_checkpoint_path is not None and args.resume_epoch != -1:
        epoch, num_stage = load_checkpoint(args, G, D_lst, optim_g, optim_d_lst,
                                         args.resume_checkpoint_path, args.resume_epoch,
                                         scheduler_g=scheduler_g, scheduler_d_lst=scheduler_d_lst,
                                         g_ema=G_ema)
        print('Resumed from saved checkpoint')

    for epoch in range(args.resume_epoch + 1, num_epochs):
        print(f"Epoch: {epoch} start")
        start_time = time.time()

        d_loss, g_loss, txt_feature = train_step(
            train_loader, args.noise_dim, G, D_lst, optim_g, optim_d_lst,
            loss_fn, args.num_stage, args.use_uncond_loss, args.use_contrastive_loss,
            args.use_mixed_loss, clip_model, gamma=args.gamma, lam=args.lam,
            report_interval=args.report_interval, device=device,
            epoch=epoch, writer=writer,
            g_ema=G_ema, ema_decay=args.ema_decay, real_label_smooth=args.real_label_smooth,
            d_update_every=args.d_update_every,
            use_diffaugment=args.use_diffaugment, diffaugment_policy=args.diffaugment_policy,
            use_mismatched_condition=args.use_mismatched_condition,
            cond_warmup_epochs=args.cond_warmup_epochs, cond_ramp_epochs=args.cond_ramp_epochs
        )

        end_time = time.time()
        print(f"Epoch: {epoch} \t d_loss: {d_loss:.4f} \t g_loss: {g_loss:.4f} \t esti. time: {(end_time - start_time):.2f}s")

        # Scheduler step (end of epoch) BEFORE saving, so the checkpoint stores the
        # post-epoch scheduler state and resume continues at the correct LR.
        scheduler_g.step()
        for scheduler_d in scheduler_d_lst:
            scheduler_d.step()

        # 샘플링 및 이미지 저장 + 체크포인트 (save_freq 마다 + 마지막 epoch —
        # save_freq의 배수가 아니면 학습 마지막 구간이 통째로 버려지는 것을 방지)
        if epoch % args.save_freq == 0 or epoch == num_epochs - 1:
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

            # 체크포인트 저장 (EMA 사용 시 EMA 가중치를 Gen.pt로 → eval/infer가 EMA 모델 사용,
            # 라이브 G는 Gen_raw.pt로 → resume이 실제 학습 가중치에서 이어짐)
            save_checkpoint(args, sample_G, D_lst, optim_g, optim_d_lst, epoch, args.num_stage,
                            scheduler_g=scheduler_g, scheduler_d_lst=scheduler_d_lst,
                            g_raw=G if args.use_ema else None)

    writer.close()
