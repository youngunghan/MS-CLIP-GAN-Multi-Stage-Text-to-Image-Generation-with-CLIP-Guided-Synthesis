import time
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
from criteria.loss import *
from trainer import train_step
from options.train_options import TrainOptions  

# torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = TrainOptions().parse()
    
    # GPU 설정
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Total available GPUs: {n_gpus}")
        
        # GPU IDs 문자열을 정수 리스트로 변환
        if isinstance(args.gpu_ids, str):
            print(f"Original GPU IDs string: '{args.gpu_ids}'")
            try:
                # CUDA_VISIBLE_DEVICES에 의해 이미 GPU가 재매핑되었으므로
                # 단순히 사용 가능한 GPU 수만큼 순차적으로 번호를 매깁니다
                args.gpu_ids = list(range(n_gpus))
                print(f"Using GPUs: {args.gpu_ids}")
            except ValueError as e:
                print(f"Error setting GPU IDs: {e}")
                args.gpu_ids = [0]
        
        if not args.gpu_ids:
            print("Warning: No GPU IDs specified. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")  # 항상 첫 번째 가용 GPU를 기본으로 사용
    else:
        print("Warning: CUDA is not available. Using CPU.")
        device = torch.device("cpu")
        args.gpu_ids = []
    
    # 체크포인트와 결과 디렉토리 생성
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    
    # 현재 실험을 위한 체크포인트 디렉토리 생성
    experiment_checkpoint_path = os.path.join(args.checkpoint_path, args.name)
    os.makedirs(experiment_checkpoint_path, exist_ok=True)
    args.checkpoint_path = experiment_checkpoint_path  # 체크포인트 경로 업데이트
    
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
    if len(args.gpu_ids) > 1:
        print(f"Using DataParallel with {len(args.gpu_ids)} GPUs")
        G = nn.DataParallel(G)  # device_ids는 자동으로 설정됨

    D_lst = []
    for curr_stage in range(args.num_stage):
        D = Discriminator(args.g_out_chans, args.d_in_chans, args.d_out_chans, 
                         args.condition_dim, args.clip_embedding_dim, curr_stage, device).to(device)
        if len(args.gpu_ids) > 1:
            D = nn.DataParallel(D)  # device_ids는 자동으로 설정됨
        D.apply(weight_init)
        D_lst.append(D)
    
    # Optimizer 설정
    optim_g = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d_lst = [Adam(D.parameters(), lr=lr, betas=(0.5, 0.999)) for D in D_lst]
    
    # 체크포인트 로드
    if args.resume_checkpoint_path is not None and args.resume_epoch != -1:
        epoch, num_stage = load_checkpoint(args, G, D_lst, optim_g, optim_d_lst, 
                                         args.resume_checkpoint_path, args.resume_epoch)
        print('Resumed from saved checkpoint')
    
    loss_fn = BCELoss()
    clip_model, _ = CLIPConfig.load_clip("ViT-B/32", device)

    # Learning rate schedulers
    scheduler_g = CosineAnnealingLR(optim_g, T_max=num_epochs)
    scheduler_d_lst = [CosineAnnealingLR(optim_d, T_max=num_epochs) 
                      for optim_d in optim_d_lst]

    for epoch in range(args.resume_epoch + 1, num_epochs):
        print(f"Epoch: {epoch} start")
        start_time = time.time()
        
        d_loss, g_loss, txt_feature = train_step(
            train_loader, args.noise_dim, G, D_lst, optim_g, optim_d_lst, 
            loss_fn, args.num_stage, args.use_uncond_loss, args.use_contrastive_loss, 
            args.use_mixed_loss, clip_model, gamma=5, lam=10, 
            report_interval=args.report_interval, device=device,
            epoch=epoch, writer=writer
        )
        
        end_time = time.time()
        print(f"Epoch: {epoch} \t d_loss: {d_loss:.4f} \t g_loss: {g_loss:.4f} \t esti. time: {(end_time - start_time):.2f}s")

        # 샘플링 및 이미지 저장
        if epoch % args.save_freq == 0:  # save_freq 마다 이미지 저장
            with torch.no_grad():
                z = torch.randn(txt_feature.shape[0], args.noise_dim).to(device)
                txt_feature = txt_feature.to(device)

                fake_images, _, _ = G(txt_feature, z)
                fake_image = fake_images[-1].detach().cpu()
                epoch_ret = torchvision.utils.make_grid(fake_image, padding=2, normalize=True)
                save_path = os.path.join(args.result_path, f"{args.name}_epoch_{epoch}.png")
                torchvision.utils.save_image(epoch_ret, save_path)

            # 체크포인트 저장
            save_checkpoint(args, G, D_lst, optim_g, optim_d_lst, epoch, args.num_stage)

        # Scheduler step
        scheduler_g.step()
        for scheduler_d in scheduler_d_lst:
            scheduler_d.step()

    writer.close()
