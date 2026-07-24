import torch
from utils.utils import normalize
from config.config import CLIPConfig
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def to_uint8(images):
    """Convert images from [-1, 1] float to uint8 [0, 255].

    This is the format torchmetrics expects when normalize=False (normalize=True would
    instead expect float in [0, 1]). Passing uint8 with normalize=True corrupts the input.
    """
    return ((images.clamp(-1, 1) + 1) * 127.5).clamp(0, 255).to(torch.uint8)


def build_fid(device):
    """Single FID metric object; update across all batches, then compute once."""
    return FrechetInceptionDistance(normalize=False).to(device)


def build_inception_score(device, splits=10):
    """Single Inception Score metric object; update across all batches, then compute once."""
    return InceptionScore(normalize=False, splits=splits).to(device)


@torch.no_grad()
def calculate_clip_score(images, text_features, clip_model, return_features=False):
    """CLIP cosine similarity between generated images (in [-1, 1]) and text features.

    ``return_features=True`` additionally returns the L2-normalized per-image CLIP
    features (shape ``[B, D]``), so a caller that also needs image-feature-derived
    statistics (e.g. a diversity metric) can reuse this single encode_image() pass
    instead of running CLIP over the same images a second time.
    """
    processed_images = CLIPConfig.preprocess_image(images)

    image_features = clip_model.encode_image(processed_images).float()
    image_features = normalize(image_features, dim=-1)
    text_features = normalize(text_features.float(), dim=-1)

    similarity = torch.sum(image_features * text_features, dim=-1).mean()
    if return_features:
        return similarity.item(), image_features
    return similarity.item()
