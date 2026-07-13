from utils.utils import *
from criteria.loss import *
from torch.utils.tensorboard import SummaryWriter
from contextlib import contextmanager
import os

d_losses = []
g_losses = []


def warmup_training_losses(use_mixed_loss, device):
    """Materialize lazy loss networks before a resumed RNG snapshot is restored.

    VGG construction consumes CPU RNG even though pretrained weights subsequently
    overwrite the initialization. Delaying it until the first post-resume G step
    would therefore diverge from an uninterrupted run.
    """
    if use_mixed_loss:
        get_vgg_perceptual_loss(device)


@contextmanager
def preserved_module_buffers(module):
    """Restore all registered buffers after a state-neutral training-mode forward.

    The discriminator phase needs generator outputs computed with train-mode batch
    statistics, but it must not count as a generator state update. Keeping the
    module in train mode preserves output semantics; snapshotting/restoring buffers
    rolls back BatchNorm running statistics and counters afterward. Parameters,
    module train/eval flags, and RNG streams are intentionally untouched.
    """
    snapshots = [(buffer, buffer.detach().clone()) for buffer in module.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in snapshots:
                buffer.copy_(value)


@contextmanager
def frozen_discriminators(discriminators):
    """Temporarily make discriminators read-only while retaining input gradients.

    Evaluation mode is important in addition to ``requires_grad_(False)``: it
    prevents BatchNorm running-stat and spectral-normalization power-iteration
    updates during the generator phase.  Every module's train/eval flag and every
    parameter's original ``requires_grad`` value are restored exactly.
    """
    module_modes = [
        [(module, module.training) for module in discriminator.modules()]
        for discriminator in discriminators
    ]
    parameter_modes = [
        [(parameter, parameter.requires_grad) for parameter in discriminator.parameters()]
        for discriminator in discriminators
    ]

    try:
        for discriminator in discriminators:
            discriminator.eval()
        for modes in parameter_modes:
            for parameter, _ in modes:
                parameter.requires_grad_(False)
        yield
    finally:
        for modes in parameter_modes:
            for parameter, requires_grad in modes:
                parameter.requires_grad_(requires_grad)
        # Assign flags directly so a parent .train() does not overwrite a child's
        # independently saved mode.
        for modes in module_modes:
            for module, training in modes:
                module.training = training

def train_step(train_loader, noise_dim, model_G, model_D_lst, optim_g, optim_d_lst,
               loss_fn, num_stage, use_uncond_loss, use_contrastive_loss, use_mixed_loss,
               clip_model, gamma, lam, report_interval, device, epoch, writer,
               g_ema=None, ema_decay=0.999, real_label_smooth=1.0, d_update_every=1,
               use_diffaugment=False, diffaugment_policy='color,translation,cutout',
               use_mismatched_condition=True):

    model_G.train()
    for D in model_D_lst:
        D.train()

    # DiffAugment policy passed to D_loss/G_loss (None = off); applied to discriminator inputs only.
    diffaug = diffaugment_policy if use_diffaugment else None

    d_loss_epoch = 0.0
    g_loss_epoch = 0.0
    d_update_count = 0
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
        if use_contrastive_loss and BATCH_SIZE < 2:
            raise ValueError(
                'contrastive training requires batch_size >= 2; configure the '
                'training loader to drop a singleton remainder'
            )
        real_imgs = [img.to(device) for img in real_imgs]
        txt_feature = txt_feature.to(device)

        g_label = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
        # one-sided label smoothing: real target may be < 1.0 to keep D from saturating
        d_real_label = torch.full((BATCH_SIZE,), real_label_smooth, dtype=torch.float32, device=device)
        d_fake_label = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)

        # ---------------- Phase 1: Optimize Discriminator ----------------
        # Update-ratio: update D only every `d_update_every` iters so the generator
        # (updated every iter below) takes more steps than the discriminator (n_critic<1).
        d_loss_iter = 0.0
        if iter % d_update_every == 0:
            # Generate fakes under no_grad so they are detached from the generator graph.
            noise = torch.randn(BATCH_SIZE, noise_dim, device=device)
            # Preserve train-mode batch-stat output semantics without letting a
            # D-only forward advance any generator buffers (especially BN state).
            with torch.no_grad(), preserved_module_buffers(model_G):
                fake_images, mu, log_sigma = model_G(txt_feature, noise)

            for i in range(num_stage):
                optim_d = optim_d_lst[i]
                optim_d.zero_grad()

                d_loss_i = D_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn,
                                  use_uncond_loss, use_contrastive_loss,
                                  gamma,
                                  mu, txt_feature,
                                  d_fake_label, d_real_label, diffaug=diffaug,
                                  use_mismatched_condition=use_mismatched_condition)
                d_loss_i = d_scale * d_loss_i
                d_loss_i.backward()

                torch.nn.utils.clip_grad_norm_(model_D_lst[i].parameters(), max_norm=1.0)
                optim_d.step()

                d_loss_iter += d_loss_i.item()
                writer.add_scalar(f'D_loss/stage_{i}', d_loss_i.item(), epoch * total_iter + iter)

            d_update_count += 1
            d_loss_epoch += d_loss_iter

        # ---------------- Phase 2: Optimize Generator ----------------
        # Remove gradients left by the preceding D update. Frozen parameters below
        # then remain grad-free throughout the generator phase.
        for optim_d in optim_d_lst:
            optim_d.zero_grad(set_to_none=True)
        optim_g.zero_grad()
        noise = torch.randn(BATCH_SIZE, noise_dim, device=device)
        fake_images, mu, log_sigma = model_G(txt_feature, noise)

        with frozen_discriminators(model_D_lst):
            g_loss = 0.0
            for i in range(num_stage):
                g_loss_i = G_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn,
                                  use_uncond_loss, use_contrastive_loss, use_mixed_loss,
                                  clip_model, gamma, lam,
                                  mu, txt_feature,
                                  g_label,
                                  device, diffaug=diffaug)
                g_loss = g_loss + g_loss_i
                writer.add_scalar(f'G_loss/stage_{i}', g_loss_i.item(), epoch * total_iter + iter)

            # Conditioning-augmentation KL regularizer
            aug_loss = KL_divergence(mu, log_sigma)
            writer.add_scalar('Loss/aug_loss', aug_loss.item(), epoch * total_iter + iter)

            g_loss = g_scale * (g_loss + aug_loss)
            g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1.0)
        optim_g.step()

        # EMA of the generator (updated every G step, after the optimizer step)
        if g_ema is not None:
            ema_update(g_ema, model_G, ema_decay)

        # ---------------- Logging ----------------
        g_loss_iter = g_loss.item()
        g_loss_epoch += g_loss_iter

        writer.add_scalar('Loss/D_total', d_loss_iter, epoch * total_iter + iter)
        writer.add_scalar('Loss/G_total', g_loss_iter, epoch * total_iter + iter)
        writer.add_scalar('Parameters/learning_rate',
                          optim_g.param_groups[0]['lr'], epoch * total_iter + iter)

        if iter % report_interval == 0 and iter >= report_interval:
            print(f"    Iteration {iter} \t d_loss: {d_loss_iter:.4f}, g_loss: {g_loss_iter:.4f}")

    d_loss_epoch /= max(d_update_count, 1)
    g_loss_epoch /= max(total_iter, 1)
    d_losses.append(d_loss_epoch)
    g_losses.append(g_loss_epoch)

    writer.add_scalar('Loss/D_epoch', d_loss_epoch, epoch)
    writer.add_scalar('Loss/G_epoch', g_loss_epoch, epoch)

    return d_loss_epoch, g_loss_epoch, save_txt_feature
