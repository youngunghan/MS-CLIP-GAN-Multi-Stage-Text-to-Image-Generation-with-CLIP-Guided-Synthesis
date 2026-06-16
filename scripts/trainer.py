from utils.utils import *
from criteria.loss import *
from torch.utils.tensorboard import SummaryWriter
import os

d_losses = []
g_losses = []

def train_step(train_loader, noise_dim, model_G, model_D_lst, optim_g, optim_d_lst,
               loss_fn, num_stage, use_uncond_loss, use_contrastive_loss, use_mixed_loss,
               clip_model, gamma, lam, report_interval, device, epoch, writer):

    model_G.train()
    for D in model_D_lst:
        D.train()

    d_loss_epoch = 0.0
    g_loss_epoch = 0.0
    save_txt_feature = None
    total_iter = len(train_loader)

    # Loss scaling
    d_scale = 0.5
    g_scale = 1.0

    for iter, batch in enumerate(train_loader):
        # NOTE: img_feature (CLIP image embedding) is intentionally unused. Both the
        # discriminator and the generator are conditioned on the SAME signal (the CLIP
        # text embedding) so that training matches inference, where only text is available.
        real_imgs, _img_feature, txt_feature = batch
        if iter == 0:
            save_txt_feature = txt_feature.clone()

        BATCH_SIZE = real_imgs[-1].shape[0]
        real_imgs = [img.to(device) for img in real_imgs]
        txt_feature = txt_feature.to(device)

        g_label = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
        d_real_label = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
        d_fake_label = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)

        # ---------------- Phase 1: Optimize Discriminator ----------------
        # Generate fakes under no_grad so they are detached from the generator graph.
        noise = torch.randn(BATCH_SIZE, noise_dim, device=device)
        with torch.no_grad():
            fake_images, mu, log_sigma = model_G(txt_feature, noise)

        d_loss_iter = 0.0
        for i in range(num_stage):
            optim_d = optim_d_lst[i]
            optim_d.zero_grad()

            d_loss_i = D_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn,
                              use_uncond_loss, use_contrastive_loss,
                              gamma,
                              mu, txt_feature,
                              d_fake_label, d_real_label)
            d_loss_i = d_scale * d_loss_i
            d_loss_i.backward()

            torch.nn.utils.clip_grad_norm_(model_D_lst[i].parameters(), max_norm=1.0)
            optim_d.step()

            d_loss_iter += d_loss_i.item()
            writer.add_scalar(f'D_loss/stage_{i}', d_loss_i.item(), epoch * total_iter + iter)

        # ---------------- Phase 2: Optimize Generator ----------------
        optim_g.zero_grad()
        noise = torch.randn(BATCH_SIZE, noise_dim, device=device)
        fake_images, mu, log_sigma = model_G(txt_feature, noise)

        g_loss = 0.0
        for i in range(num_stage):
            g_loss_i = G_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn,
                              use_uncond_loss, use_contrastive_loss, use_mixed_loss,
                              clip_model, gamma, lam,
                              mu, txt_feature,
                              g_label,
                              device)
            g_loss = g_loss + g_loss_i
            writer.add_scalar(f'G_loss/stage_{i}', g_loss_i.item(), epoch * total_iter + iter)

        # Conditioning-augmentation KL regularizer
        aug_loss = KL_divergence(mu, log_sigma)
        writer.add_scalar('Loss/aug_loss', aug_loss.item(), epoch * total_iter + iter)

        g_loss = g_scale * (g_loss + aug_loss)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1.0)
        optim_g.step()

        # ---------------- Logging ----------------
        g_loss_iter = g_loss.item()
        d_loss_epoch += d_loss_iter
        g_loss_epoch += g_loss_iter

        writer.add_scalar('Loss/D_total', d_loss_iter, epoch * total_iter + iter)
        writer.add_scalar('Loss/G_total', g_loss_iter, epoch * total_iter + iter)
        writer.add_scalar('Parameters/learning_rate',
                          optim_g.param_groups[0]['lr'], epoch * total_iter + iter)

        if iter % report_interval == 0 and iter >= report_interval:
            print(f"    Iteration {iter} \t d_loss: {d_loss_iter:.4f}, g_loss: {g_loss_iter:.4f}")

    d_loss_epoch /= max(total_iter, 1)
    g_loss_epoch /= max(total_iter, 1)
    d_losses.append(d_loss_epoch)
    g_losses.append(g_loss_epoch)

    writer.add_scalar('Loss/D_epoch', d_loss_epoch, epoch)
    writer.add_scalar('Loss/G_epoch', g_loss_epoch, epoch)

    return d_loss_epoch, g_loss_epoch, save_txt_feature
