import torch
from torch.nn import BCELoss
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config.config import CLIPConfig
#from utils.utils import normalize
from utils.utils import *
from criteria.diffaugment import DiffAugment

def gather_all(dicts) -> float:
    """Sum all values in dictionary"""
    return sum(dicts.values())

def KL_divergence(mu, log_sigma):
    """Calculate KL divergence loss"""
    kldiv = -log_sigma - 0.5 + (torch.exp(2 * log_sigma) + mu ** 2) * 0.5
    return torch.mean(torch.sum(kldiv, dim=1))

class VGGPerceptualLoss(nn.Module):
    """VGG Perceptual loss calculator"""
    def __init__(self, device):
        super().__init__()
        # Load pretrained VGG16 and extract specific feature layers
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features.eval()
        self.vgg_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ]).to(device)

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # Register VGG mean/std normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize input images for VGG"""
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """Calculate perceptual loss between pred and target images"""
        # Normalize inputs
        pred = self.normalize(pred)
        target = self.normalize(target)

        # Calculate feature loss at different layers
        loss = 0
        for layer in self.vgg_layers:
            pred = layer(pred)
            with torch.no_grad():
                target = layer(target)
            loss += F.mse_loss(pred, target)

        return loss

def mixed_loss(pred, target, perceptual_loss_fn, alpha=0.3):
    """
    Calculate mixed loss between predicted and target images

    Args:
        pred: predicted images [B, 3, H, W]
        target: target images [B, 3, H, W]
        perceptual_loss_fn: VGG perceptual loss calculator
        alpha: weight for L1 loss (1-alpha for perceptual loss)

    Returns:
        weighted sum of L1 and perceptual losses
    """
    # L1 loss on the model's native [-1, 1] range
    l1_loss = F.l1_loss(pred, target)

    # VGG expects images in [0, 1]; denormalize before the perceptual term
    pred_01 = (pred.clamp(-1, 1) + 1) / 2
    target_01 = (target.clamp(-1, 1) + 1) / 2
    perceptual_loss = perceptual_loss_fn(pred_01, target_01)

    return alpha * l1_loss + (1-alpha) * perceptual_loss

# Cache VGG perceptual loss per device so it is not rebuilt every iteration/stage.
_VGG_PERCEPTUAL_CACHE = {}

def get_vgg_perceptual_loss(device):
    key = str(device)
    if key not in _VGG_PERCEPTUAL_CACHE:
        _VGG_PERCEPTUAL_CACHE[key] = VGGPerceptualLoss(device).to(device)
    return _VGG_PERCEPTUAL_CACHE[key]

def contrastive_loss_D(d_out_align, txt_feature, tau=0.5):
    '''
    Cross-modal contrastive (InfoNCE) loss between the discriminator's alignment output
    and the CLIP text feature. Both are L2-normalized so the similarity scale is controlled;
    matched (image, text) pairs are the diagonal targets. Mirrors contrastive_loss_G.

    Inputs:
        d_out_align: discriminator alignment output [B, D]
        txt_feature: text feature [B, D]
    Outputs:
        L_cont: contrastive loss value (scalar)
    '''
    batch_size = d_out_align.size(0)
    if batch_size < 2:
        raise ValueError(
            'contrastive_loss_D requires at least two samples; a singleton batch '
            'has no negatives'
        )
    d = normalize(d_out_align.view(batch_size, -1))
    t = normalize(txt_feature.view(batch_size, -1))

    logits = torch.mm(d, t.t()) / tau
    labels = torch.arange(batch_size, device=d_out_align.device)
    L_cont = cross_entropy(logits, labels)
    return L_cont

def D_loss(real_image, fake_image, model_D, loss_fn,
               use_uncond_loss, use_contrastive_loss,
               gamma,
               mu, txt_feature,
               d_fake_label, d_real_label, diffaug=None,
               use_mismatched_condition=True):

    loss_d_comp = {}

    # DiffAugment: the discriminator sees the SAME differentiable transform on real and fake.
    if diffaug:
        real_image = DiffAugment(real_image, policy=diffaug)
        fake_image = DiffAugment(fake_image, policy=diffaug)

    mismatched_mu = (
        mu.roll(shifts=1, dims=0)
        if use_mismatched_condition and real_image.shape[0] > 1 else None
    )
    fake_details = model_D(
        img=fake_image, condition=mu,
        compute_alignment=use_contrastive_loss,
        compute_unconditional=use_uncond_loss,
        return_details=True,
    )
    loss_d_comp["d_loss_fake_cond"] = loss_fn(
        fake_details['conditional'], d_fake_label
    )

    real_details = model_D(
        img=real_image, condition=mu,
        compute_alignment=use_contrastive_loss,
        mismatched_condition=mismatched_mu,
        compute_unconditional=use_uncond_loss,
        return_details=True,
    )
    loss_d_comp["d_loss_real_cond"] = loss_fn(
        real_details['conditional'], d_real_label
    )

    # A real image paired with another sample's text is a conditional negative.
    # Without this term, the conditional discriminator only learns real-vs-fake and
    # is never directly required to check whether real content matches its text.
    # A one-position cyclic shift guarantees no fixed points for every B > 1.
    if mismatched_mu is not None:
        loss_d_comp["d_loss_real_mismatched_cond"] = loss_fn(
            real_details['mismatched_conditional'], d_fake_label
        )
        # Preserve the original 1:1 positive/negative mass: matched real keeps
        # weight 1 while generated-fake and mismatched-real share negative weight 1.
        # Otherwise adding the matching supervision would also increase total D
        # scale by 50%, confounding the intended change with a D/G balance change.
        loss_d_comp["d_loss_fake_cond"] = (
            0.5 * loss_d_comp["d_loss_fake_cond"]
        )
        loss_d_comp["d_loss_real_mismatched_cond"] = (
            0.5 * loss_d_comp["d_loss_real_mismatched_cond"]
        )

    # NOTE: This is a vanilla (Sigmoid + BCE) GAN with spectral normalization in the
    # discriminator for Lipschitz control. The previous WGAN-GP gradient penalty was
    # removed because it is inconsistent with a bounded (sigmoid) probability output.

    if use_uncond_loss:
        loss_d_comp["d_loss_fake_uncond"] = loss_fn(
            fake_details['unconditional'], d_fake_label
        )
        loss_d_comp["d_loss_real_uncond"] = loss_fn(
            real_details['unconditional'], d_real_label
        )

    if use_contrastive_loss:
        loss_d_comp['d_loss_fake_cond_contrastive'] = gamma * contrastive_loss_D(fake_details['alignment'], txt_feature)
        loss_d_comp['d_loss_real_cond_contrastive'] = gamma * contrastive_loss_D(real_details['alignment'], txt_feature)

    d_loss = gather_all(loss_d_comp)
    return d_loss

def contrastive_loss_G(fake_image, clip_model, txt_embedding, device, tau=0.5):
    if fake_image.shape[0] < 2:
        raise ValueError(
            'contrastive_loss_G requires at least two samples; a singleton batch '
            'has no negatives'
        )
    clip_norm_img = CLIPConfig.get_transform()(
        CLIPConfig.denormalize_image(torch.clamp(fake_image, -1, 1))
    ).to(device)
    # CLIP runs internally in its own (possibly fp16) dtype; cast features to float32
    # so the contrastive matmul / cross-entropy are numerically stable. Gradients still
    # flow back to the generator through fake_image.
    image_feat = normalize(clip_model.encode_image(clip_norm_img).float())

    # Similarity between samples within a batch (InfoNCE)
    logits = (txt_embedding.float() @ image_feat.T) / tau
    labels = torch.arange(logits.shape[0], device=device)  # [0, 1, ..., B-1]
    L_cont = cross_entropy(logits, labels)

    return L_cont

def G_loss(real_image, fake_image, model_D, loss_fn,
           use_uncond_loss, use_contrastive_loss, use_mixed_loss,
           clip_model, gamma, lam,
           mu, txt_feature,
           g_label,
           device, diffaug=None):

    loss_g_comp = {}

    # DiffAugment only the discriminator's view of the fake (gradients still flow to G);
    # the CLIP (contrastive_loss_G) and VGG (mixed_loss) terms below use the RAW fake.
    fake_for_d = DiffAugment(fake_image, policy=diffaug) if diffaug else fake_image

    details = model_D(
        img=fake_for_d, condition=mu,
        compute_alignment=use_contrastive_loss,
        compute_unconditional=use_uncond_loss,
        return_details=True,
    )
    loss_g_comp["g_loss_cond"] = loss_fn(details['conditional'], g_label)

    if use_uncond_loss:
        loss_g_comp["g_loss_uncond"] = loss_fn(
            details['unconditional'], g_label
        )

    if use_contrastive_loss:
        if min(fake_image.shape[-2:]) >= CLIPConfig.MIN_QUALITY_SIZE:
            loss_g_comp['g_loss_cond_contrastive'] = lam * contrastive_loss_G(fake_image, clip_model, txt_feature, device)
        loss_g_comp['d_loss_cond_contrastive'] = gamma * contrastive_loss_D(details['alignment'], txt_feature)

    # Add EIGGAN mixed loss
    if use_mixed_loss:
        mixed_loss_weight = 0.1
        perceptual_loss_fn = get_vgg_perceptual_loss(device)
        loss_g_comp["g_loss_mixed"] = mixed_loss_weight * mixed_loss(
            fake_image,
            real_image,
            perceptual_loss_fn
        )

    # Loss 가중치 조정
    loss_weights = {
        "g_loss_cond": 1.0,
        "g_loss_uncond": 0.5,
        "g_loss_cond_contrastive": 1.0,
        "d_loss_cond_contrastive": 0.5,
        "g_loss_mixed": 1.0
    }

    for key in loss_g_comp:
        loss_g_comp[key] = loss_weights.get(key, 1.0) * loss_g_comp[key]

    g_loss = gather_all(loss_g_comp)
    return g_loss
