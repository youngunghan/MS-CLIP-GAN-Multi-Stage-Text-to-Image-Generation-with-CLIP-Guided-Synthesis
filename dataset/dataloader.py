import json
import zipfile
import PIL.Image
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from typing import Union, Optional
from pathlib import Path
from utils.utils import normalize

class MM_CelebA(Dataset):
    ''' 
    LAFITE Paper Dataset 
    Loading images and CLIP embeddings from ZIP file
    '''
    BASE_SIZE = 64
    VALID_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    
    def __init__(self, data_path: Union[str, Path], num_stage: int):
        self.data_path = Path(data_path)
        self.num_stage = num_stage
        self.img_sizes = [self.BASE_SIZE * (2 ** i) for i in range(num_stage)]
        
        self.images = {}
        self.clip_img_embs = {}
        self.clip_txt_embs = {}
        self.idx_to_filename = {}
        
        self._load_dataset()

    @staticmethod
    def get_transform(size: Optional[int] = None) -> T.Compose:
        transforms = [
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            #T.RandomHorizontalFlip(p=0.5),
            #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            #T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),  # PIL Image를 Tensor로 변환
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        return T.Compose(transforms)
    
    def _is_image_file(self, filename: Union[str, Path]) -> bool:
        """Check if file is an image"""
        return Path(filename).suffix.lower().lstrip('.') in self.VALID_EXTENSIONS
    
    def _load_dataset(self):
        """Load all data from ZIP file"""
        with zipfile.ZipFile(self.data_path, mode='r') as zf:
            image_files = sorted( f for f in zf.namelist() if self._is_image_file(f) )
            transform = self.get_transform(self.BASE_SIZE)
            for idx, fname in enumerate(image_files):
                self.idx_to_filename[idx] = fname
                with zf.open(fname) as f:
                    img = PIL.Image.open(f).convert('RGB')
                    self.images[fname] = transform(img)
            print('finished img files')
            
            if 'dataset.json' not in zf.namelist():
                raise ValueError("dataset.json not found in ZIP file")
            
            with zf.open('dataset.json') as f:
                data = json.load(f)
                
                for fname, embedding in data['clip_img_features']:
                    self.clip_img_embs[fname] = normalize(torch.tensor(embedding, dtype=torch.float32), dim=0)
                print('finished img features')
                
                for fname, embedding in data['clip_txt_features']:
                    if isinstance(embedding[0], list):
                        # 2D 리스트인 경우 (여러 텍스트 임베딩)
                        self.clip_txt_embs[fname] = normalize(torch.tensor(embedding, dtype=torch.float32), dim=1)
                    else:
                        # 1D 리스트인 경우 (단일 텍스트 임베딩)
                        self.clip_img_embs[fname] = normalize(torch.tensor(embedding, dtype=torch.float32), dim=0)                        
                print('finished txt features')
    
    def __len__(self) -> int:
        return len(self.idx_to_filename)
    
    def __getitem__(self, idx: int):
        """Get dataset item
        
        Args:
            idx: Index of item
            
        Returns:
            tuple of:
                - list of images at different resolutions
                - CLIP image embedding
                - randomly selected CLIP text embedding
        """
        filename = self.idx_to_filename[idx]
        base_img = self.images[filename]
        
        # 다중 해상도 이미지 생성
        imgs = [base_img]
        for size in self.img_sizes[1:]:
            imgs.append(F.interpolate(
                base_img.unsqueeze(0),
                size=(size, size),
                mode='bicubic',
                align_corners=True
            ).squeeze(0))
        
        # CLIP 임베딩 가져오기
        img_embedding = self.clip_img_embs[filename]
        txt_embeddings = self.clip_txt_embs[filename]
        
        # 랜덤 텍스트 임베딩 선택
        txt_idx = random.randint(0, len(txt_embeddings) - 1)
        
        return imgs, img_embedding, txt_embeddings[txt_idx]
    
def get_dataloader(args, dataset, is_train=True):
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset=dataset, 
                            sampler=sampler,  
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    return dataloader