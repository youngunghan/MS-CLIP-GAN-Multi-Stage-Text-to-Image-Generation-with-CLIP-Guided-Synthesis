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
    Loading images and CLIP embeddings from ZIP file.

    Images are decoded lazily (per __getitem__) at the dataset's full resolution
    (= the largest stage size) and then *down*-sampled to each stage resolution.
    This preserves the genuine high-resolution content stored by the preprocessing
    step instead of discarding it by resizing everything to 64px first.
    '''
    BASE_SIZE = 64
    VALID_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

    def __init__(self, data_path: Union[str, Path], num_stage: int):
        self.data_path = Path(data_path)
        self.num_stage = num_stage
        self.img_sizes = [self.BASE_SIZE * (2 ** i) for i in range(num_stage)]
        self.full_size = self.img_sizes[-1]
        self.transform = self.get_transform(self.full_size)

        self.idx_to_filename = {}
        self.clip_img_embs = {}
        self.clip_txt_embs = {}
        self._zip = None  # opened lazily, once per worker process

        self._load_metadata()

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

    def _load_metadata(self):
        """Read the file index and CLIP embeddings (small) from the ZIP. Images are NOT decoded here."""
        with zipfile.ZipFile(self.data_path, mode='r') as zf:
            image_files = sorted(f for f in zf.namelist() if self._is_image_file(f))

            if 'dataset.json' not in zf.namelist():
                raise ValueError("dataset.json not found in ZIP file")
            with zf.open('dataset.json') as f:
                data = json.load(f)

        for fname, embedding in data['clip_img_features']:
            self.clip_img_embs[fname] = normalize(torch.tensor(embedding, dtype=torch.float32), dim=0)
        print('finished img features')

        for fname, embedding in data['clip_txt_features']:
            if not embedding:
                # no caption embeddings for this image -> it will be dropped below
                continue
            if isinstance(embedding[0], list):
                # 2D list (multiple text embeddings per image) -> [N, D]
                self.clip_txt_embs[fname] = normalize(torch.tensor(embedding, dtype=torch.float32), dim=1)
            else:
                # 1D list (single text embedding) -> store as [1, D] so __getitem__ indexing works
                self.clip_txt_embs[fname] = normalize(
                    torch.tensor(embedding, dtype=torch.float32), dim=0
                ).unsqueeze(0)
        print('finished txt features')

        # Index only images that have BOTH an image and a (non-empty) text embedding,
        # so __getitem__ can never hit a missing key / empty caption list.
        valid = [f for f in image_files if f in self.clip_img_embs and f in self.clip_txt_embs]
        dropped = len(image_files) - len(valid)
        if dropped:
            print(f'Warning: dropped {dropped} image(s) without matching img/txt embeddings')
        if len(valid) == 0:
            raise ValueError("No samples with both image and text embeddings were found")
        self.idx_to_filename = {idx: fname for idx, fname in enumerate(valid)}
        print(f'indexed {len(valid)} samples')

    def _get_zip(self) -> zipfile.ZipFile:
        """Return a per-process ZipFile handle (opened lazily so DataLoader workers each get their own)."""
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.data_path, mode='r')
        return self._zip

    def __len__(self) -> int:
        return len(self.idx_to_filename)

    def __getitem__(self, idx: int):
        """Get dataset item

        Returns:
            tuple of:
                - list of images at each stage resolution (ascending, e.g. 64/128/256)
                - CLIP image embedding
                - randomly selected CLIP text embedding
        """
        filename = self.idx_to_filename[idx]
        zf = self._get_zip()
        with zf.open(filename) as f:
            base_img = self.transform(PIL.Image.open(f).convert('RGB'))  # [3, full, full] in [-1, 1]

        # Build the multi-resolution list by *down*-sampling the full-res image.
        imgs = []
        for size in self.img_sizes:
            if size == self.full_size:
                imgs.append(base_img)
            else:
                imgs.append(
                    F.interpolate(base_img.unsqueeze(0), size=(size, size), mode='area').squeeze(0)
                )

        img_embedding = self.clip_img_embs[filename]
        txt_embeddings = self.clip_txt_embs[filename]

        # randomly select one caption embedding
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
