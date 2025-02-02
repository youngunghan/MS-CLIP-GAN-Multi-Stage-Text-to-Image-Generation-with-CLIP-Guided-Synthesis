import torch
from utils.utils import normalize
from config.config import CLIPConfig
from torchmetrics.image.fid import FrechetInceptionDistance
from ignite.metrics import FID as IgniteFID
from ignite.metrics import InceptionScore

def calculate_clip_score(images, text_features, clip_model):
    """Calculate CLIP score"""
    processed_images = CLIPConfig.preprocess_image(images)
    
    # Calculate features and similarity
    with torch.no_grad():
        image_features = clip_model.encode_image(processed_images)
        image_features = normalize(image_features, dim=-1)
        similarity = torch.sum(image_features * text_features, dim=-1).mean()
    
    return similarity.item()

def calculate_fid_score(real_images, fake_images, device):
    """Calculate FID score"""
    try:
        # Check and normalize image range
        for images, name in [(real_images, 'real'), (fake_images, 'fake')]:
            if images.min() < -1 or images.max() > 1:
                print(f"Warning: {name} images out of range [-1, 1]. Min: {images.min()}, Max: {images.max()}")
                images = torch.clamp(images, -1, 1)
        
        # [-1, 1] -> [0, 255]
        real_images = ((real_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        fake_images = ((fake_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        
        # Calculate FID
        fid = FrechetInceptionDistance(normalize=True).to(device)
        with torch.no_grad():
            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)
        
        return fid.compute().item()
    except Exception as e:
        print(f"Error in FID calculation: {str(e)}")
        return float('inf')

def calculate_ignite_fid_score(real_images, fake_images, device):
    """Calculate FID score using ignite implementation"""
    try:
        # [-1, 1] -> [0, 255]
        real_images = ((real_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        fake_images = ((fake_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        
        # Calculate FID using ignite
        fid_metric = IgniteFID().to(device)
        fid_metric.update((real_images, fake_images))
        
        return fid_metric.compute()
    except Exception as e:
        print(f"Error in Ignite FID calculation: {str(e)}")
        return float('inf')

def calculate_inception_score(fake_images, device, n_split=10):
    """Calculate Inception Score"""
    try:
        # [-1, 1] -> [0, 255]
        fake_images = ((fake_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        
        # Calculate Inception Score
        is_metric = InceptionScore(n_split=n_split).to(device)
        is_metric.update(fake_images)
        
        return is_metric.compute()
    except Exception as e:
        print(f"Error in Inception Score calculation: {str(e)}")
        return (float('inf'), float('inf'))  # Returns (mean, std)