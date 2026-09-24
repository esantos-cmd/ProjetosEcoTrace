"""Dataset e aumentos de dados.
 
Estrutura esperada:
    data/
      train/images/xxx.jpg   train/masks/xxx.png
      val/images/yyy.jpg     val/masks/yyy.png
 
Cada máscara é um PNG em tons de cinza onde o valor do pixel é o índice
da classe (0 = fundo, 1 = carne_magra, 2 = gordura, 3 = osso).
"""
from pathlib import Path
 
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = (0.485, 0.456, 0.406)  # ImageNet
STD = (0.229, 0.224, 0.225)
 
 
def get_train_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        # Cor é informação importante para carne/gordura: variações pequenas.
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.HueSaturationValue(5, 10, 10, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
 
 
def get_val_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
 
 
class MeatDataset(Dataset):
    def __init__(self, root: str, transform: A.Compose | None = None):
        root = Path(root)
        self.images = sorted(p for p in (root / "images").iterdir()
                             if p.suffix.lower() in IMG_EXTS)
        self.masks_dir = root / "masks"
        self.transform = transform
        if not self.images:
            raise FileNotFoundError(f"Nenhuma imagem em {root / 'images'}")
        missing = [p.name for p in self.images
                   if not (self.masks_dir / f"{p.stem}.png").exists()]
        if missing:
            raise FileNotFoundError(f"Máscaras ausentes para: {missing[:5]}...")
 
    def __len__(self) -> int:
        return len(self.images)
 
    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / f"{img_path.stem}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
            mask = torch.from_numpy(mask)
        return image, mask.long()
 
 
def preprocess(image_bgr: np.ndarray, size: int) -> torch.Tensor:
    """Prepara um frame BGR do OpenCV para inferência (1, 3, H, W)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = get_val_transform(size)(image=rgb)["image"]
    return tensor.unsqueeze(0)
 