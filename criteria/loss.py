import torch
from torch.nn import BCELoss
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config.config import CLIPConfig
#from utils.utils import normalize
from utils.utils import *

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
        vgg = models.vgg16(pretrained=True).features.eval()
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
    # L1 loss for pixel-level reconstruction
    l1_loss = F.l1_loss(pred, target)
    
    # Perceptual loss for semantic features
    perceptual_loss = perceptual_loss_fn(pred, target)
    
    return alpha * l1_loss + (1-alpha) * perceptual_loss

def contrastive_loss_D(d_out_align, txt_feature):
    '''
    Inputs:
        d_out_align: discriminator alignment output [B, D] e.g., [32, 256]
        txt_feature: text feature [B, D] e.g., [32, 256]
    Outputs:
        L_cont: contrastive loss value (scalar)
    '''
    batch_size = d_out_align.size(0)
    
    # Ensure 2D matrices for similarity computation
    d_out_align = d_out_align.view(batch_size, -1)
    txt_feature = txt_feature.view(batch_size, -1)
    
    # Create identity matrix as target distribution
    labels = torch.eye(batch_size).to(d_out_align.device)
    
    # Compute similarity scores
    logits = torch.mm(d_out_align, txt_feature.t())
    
    # Calculate KL divergence loss
    log_probs = F.log_softmax(logits, dim=1)
    L_cont = F.kl_div(log_probs, labels, reduction='batchmean')
    return L_cont

# use InfoNCE Loss

def compute_gradient_penalty(discriminator, real_samples, fake_samples, condition, device):
    """Gradient penalty for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Calculate discriminator output for interpolated images
    d_interpolates, _ = discriminator(img=interpolates, condition=condition)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def D_loss(real_image, fake_image, model_D, loss_fn, 
               use_uncond_loss, use_contrastive_loss, 
               gamma,
               mu, txt_feature,
               d_fake_label, d_real_label):
    
    loss_d_comp = {}
    device = real_image.device

    d_out_cond, d_out_align_fake = model_D(img=fake_image, condition=mu,)
    loss_d_comp["d_loss_fake_cond"] = loss_fn(d_out_cond, d_fake_label)

    d_out_cond, d_out_align_real = model_D(img=real_image, condition=mu,)
    loss_d_comp["d_loss_real_cond"] = loss_fn(d_out_cond, d_real_label)
    
    # Add gradient penalty
    gradient_penalty = 10.0 * compute_gradient_penalty(
        model_D, real_image, fake_image, mu, device
    )
    loss_d_comp["gradient_penalty"] = gradient_penalty
    
    if use_uncond_loss:
        d_out_uncond, _ = model_D(img=fake_image, condition=None,)
        loss_d_comp["d_loss_fake_uncond"] = loss_fn(d_out_uncond, d_fake_label)

        d_out_uncond, _ = model_D(img=real_image, condition=None,)
        loss_d_comp["d_loss_real_uncond"] = loss_fn(d_out_uncond, d_real_label)
    
    if use_contrastive_loss:
        loss_d_comp['d_loss_fake_cond_contrastive'] = gamma * contrastive_loss_D(d_out_align_fake, txt_feature)
        loss_d_comp['d_loss_real_cond_contrastive'] = gamma * contrastive_loss_D(d_out_align_real, txt_feature)

    d_loss = gather_all(loss_d_comp)
    return d_loss

def contrastive_loss_G(fake_image, clip_model, txt_embedding, device, tau=0.5):    
    clip_norm_img = CLIPConfig.get_transform()(
        CLIPConfig.denormalize_image(fake_image)
    ).to(device)
    image_feat = normalize( clip_model.encode_image(clip_norm_img) )
    
    # Similarity between samples within a batch
    logits = (txt_embedding.type(torch.float16) @ image_feat.T) / tau
    
    # sim_matrix = torch.log(
    #     (txt_embedding.type(torch.float16) @ image_feat.T / tau).softmax(dim=-1)
    # )
    # L_cont = -tau * torch.sum(torch.diagonal(sim_matrix))
    
    labels = torch.arange(logits.shape[0], device=device) # [0, 1, ..., B-1]
    L_cont = cross_entropy(logits, labels)
    
    return L_cont
# use InfoNCE Loss

def G_loss(real_image, fake_image, model_D, loss_fn,
           use_uncond_loss, use_contrastive_loss, use_mixed_loss,
           clip_model, gamma, lam, 
           mu, txt_feature,
           g_label,
           device):
    
    loss_g_comp = {}

    g_out_cond, g_out_align = model_D(img=fake_image, condition=mu,)
    loss_g_comp["g_loss_cond"] = loss_fn(g_out_cond, g_label)
    
    if use_uncond_loss:
        g_out_uncond, _ = model_D(img=fake_image, condition=None,)
        loss_g_comp["g_loss_uncond"] = loss_fn(g_out_uncond, g_label)

    if use_contrastive_loss:
        if min(fake_image.shape[-2:]) >= CLIPConfig.MIN_QUALITY_SIZE: 
            loss_g_comp['g_loss_cond_contrastive'] = lam * contrastive_loss_G(fake_image, clip_model, txt_feature, device)
        loss_g_comp['d_loss_cond_contrastive'] = gamma * contrastive_loss_D(g_out_align, txt_feature)
    
    # Add EIGGAN mixed loss
    if use_mixed_loss:
        mixed_loss_weight = 0.1
        perceptual_loss_fn = VGGPerceptualLoss(device).to(device)
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