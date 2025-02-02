from utils.utils import *
from criteria.loss import *
from torch.utils.tensorboard import SummaryWriter
import os

d_losses = []
g_losses = []

def train_step(train_loader, noise_dim, model_G, model_D_lst, optim_g, optim_d_lst, 
               loss_fn, num_stage, use_uncond_loss, use_contrastive_loss, use_mixed_loss,
               clip_model, gamma, lam, report_interval, device, epoch, writer):
    
    d_loss_train = g_loss_train = 0
    save_txt_feature = None
    total_iter = len(train_loader)
    
    # Loss 스케일링
    d_scale = 0.5
    g_scale = 1.0
    
    # Loss smoothing
    d_loss_ema = 0.9
    g_loss_ema = 0.9
    
    for iter, batch in enumerate(train_loader):
        real_imgs, img_feature, txt_feature = batch
        if iter == 0: save_txt_feature = txt_feature.clone()

        BATCH_SIZE = real_imgs[-1].shape[0]
        real_imgs = [img.to(device) for img in real_imgs]
        img_feature, txt_feature = img_feature.to(device), txt_feature.to(device)
        
        '''
        Improved Pseudo Text Feature Generation:
        1. Style Mixing: StyleGAN-NADA의 style mixing 아이디어 적용
        2. Feature Interpolation: DiffusionCLIP의 interpolation 전략 사용
        3. Directional Guidance: 의미있는 방향으로의 특징 변형
        '''
        # 1. Style Direction 계산
        style_directions = []
        for i in range(BATCH_SIZE):
            # 현재 이미지의 스타일 방향 추출
            img_direction = normalize(img_feature[i])
            # 텍스트의 스타일 방향 추출
            txt_direction = normalize(txt_feature[i])
            # 스타일 방향 계산
            style_direction = txt_direction - img_direction
            style_directions.append(style_direction)
        style_directions = torch.stack(style_directions)

        # 2. Adaptive Mixing Strength
        # 이미지와 텍스트 특징의 유사도에 따라 mixing strength 조정
        similarities = torch.sum(img_feature * txt_feature, dim=1)
        mixing_strengths = torch.sigmoid(similarities).unsqueeze(1)
        
        # 3. Feature Interpolation with Guidance
        for idx in range(BATCH_SIZE):
            # Base noise 생성
            noise = torch.randn(img_feature.shape[-1]).to(device)
            noise = normalize(noise)
            
            # Style-guided noise
            style_noise = noise + 0.2 * style_directions[idx]
            style_noise = normalize(style_noise)
            
            # Adaptive mixing
            alpha = 0.3 * mixing_strengths[idx]  # 기본 0.3에 유사도 기반 조정
            img_embedding = (1 - alpha) * normalize(img_feature[idx]) + \
                          alpha * style_noise
            
            # Feature enhancement with directional guidance
            img_embedding = img_embedding + \
                          0.1 * normalize(style_directions[idx])  # 약한 방향성 가이드
            
            img_feature[idx] = normalize(img_embedding)

        # Tensorboard logging
        if iter % report_interval == 0:
            writer.add_scalar('Training/style_similarity', 
                            similarities.mean().item(),
                            epoch * total_iter + iter)
            writer.add_scalar('Training/mixing_strength',
                            mixing_strengths.mean().item(),
                            epoch * total_iter + iter)

        # Use modified img_feature instead of txt_feature
        g_label = torch.ones(BATCH_SIZE).type(torch.float32).to(device)
        d_real_label = torch.ones(BATCH_SIZE).type(torch.float32).to(device)
        d_fake_label = torch.zeros(BATCH_SIZE).type(torch.float32).to(device)

        # Phase 1. Optimize Discriminator
        noise = torch.randn(BATCH_SIZE, noise_dim).to(device)
        fake_images, mu, log_sigma = model_G(img_feature, noise)  # txt_feature -> img_feature
        
        d_loss = 0
        for i in range(num_stage):
            optim_d = optim_d_lst[i]
            optim_d.zero_grad()

            d_loss_i = D_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn, 
                              use_uncond_loss, use_contrastive_loss,
                              gamma,
                              mu, txt_feature,
                              d_fake_label, d_real_label)
            
            # Scale discriminator loss
            d_loss_i *= d_scale
            
            d_loss += d_loss_i.detach().item()
            d_loss_i.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_D_lst[i].parameters(), max_norm=1.0)
            
            optim_d.step()
            d_loss_train = d_loss_ema * d_loss_train + (1 - d_loss_ema) * d_loss_i.item()
            
            writer.add_scalar(f'D_loss/stage_{i}', d_loss_i.item(), 
                            epoch * total_iter + iter)

        # Phase 2. Optimize Generator
        optim_g.zero_grad()
        noise = torch.randn(BATCH_SIZE, noise_dim).to(device)
        fake_images, mu, log_sigma = model_G(txt_feature, noise)
        
        g_loss = 0
        for i in range(num_stage):
            g_loss_i = G_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn,
                              use_uncond_loss, use_contrastive_loss, use_mixed_loss,
                              clip_model, gamma, lam, 
                              mu, txt_feature,
                              g_label,
                              device)
            g_loss += g_loss_i
            
            writer.add_scalar(f'G_loss/stage_{i}', g_loss_i.item(), 
                            epoch * total_iter + iter)

        # Augmentation loss
        aug_loss = KL_divergence(mu, log_sigma)
        writer.add_scalar('Loss/aug_loss', aug_loss.item(), 
                         epoch * total_iter + iter)
        
        g_loss = g_scale * (g_loss + aug_loss)  # Scale generator loss
        g_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1.0)
        
        optim_g.step()
        g_loss_train = g_loss_ema * g_loss_train + (1 - g_loss_ema) * g_loss.item()

        # Monitoring metrics
        writer.add_scalar('Loss/D_total', d_loss, epoch * total_iter + iter)
        writer.add_scalar('Loss/G_total', g_loss.item(), epoch * total_iter + iter)
        writer.add_scalar('Loss/D_G_ratio', d_loss / (g_loss.item() + 1e-8), 
                         epoch * total_iter + iter)
        writer.add_scalar('Parameters/learning_rate', optim_g.param_groups[0]['lr'], epoch)
        writer.add_scalar('Parameters/D_G_loss_ratio', d_loss_train / (g_loss_train + 1e-8), epoch)

        # Gradient norms
        def log_gradients(model, name, writer, epoch):
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar(f'Gradients/{name}_norm', total_norm, epoch)

        log_gradients(model_G, 'Generator', writer, epoch)
        for i, D in enumerate(model_D_lst):
            log_gradients(D, f'Discriminator_{i}', writer, epoch)

        if iter % report_interval == 0 and iter >= report_interval:
            print(f"    Iteration {iter} \t d_loss: {d_loss:.4f}, g_loss: {g_loss.item():.4f}")

    d_loss_train /= len(train_loader)
    g_loss_train /= len(train_loader)
    d_losses.append(d_loss_train)
    g_losses.append(g_loss_train)
    
    writer.add_scalar('Loss/D_epoch', d_loss_train, epoch)
    writer.add_scalar('Loss/G_epoch', g_loss_train, epoch)

    return d_loss_train, g_loss_train, save_txt_feature