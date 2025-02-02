from typing import Tuple
import clip
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class CLIPConfig:
    """CLIP model configuration and transforms"""
    
    # CLIP 모델 기본 설정
    IMAGE_SIZE: int = 224  # CLIP 입력 크기
    MIN_QUALITY_SIZE: int = 256  # 최소 품질 기준
    FEATURE_DIM: int = 512  # CLIP 특징 벡터 차원
    
    # 정규화 상수
    MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    
    @staticmethod
    def load_clip(model, device):
        ''' 'B/32', 'L/14', 'B/16' '''
        return clip.load(model, device=device)
    
    @staticmethod
    def denormalize_image(x):
        """Denormalize image from [-1,1] to [0,1] range"""
        return (x + 1) / 2
    
    @classmethod
    def get_transform(cls) -> transforms.Compose:
        """Get CLIP image transform"""
        return transforms.Compose([
            transforms.Resize(cls.IMAGE_SIZE, 
                            interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(cls.IMAGE_SIZE),
            transforms.Normalize(mean=cls.MEAN, std=cls.STD)
        ])
    
    @staticmethod
    def preprocess_image(image):
        """Preprocess images for CLIP model"""
        image = CLIPConfig.denormalize_image( torch.clamp(image, -1, 1) )
        return CLIPConfig.get_transform()(image)
    
    
