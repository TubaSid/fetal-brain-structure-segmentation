"""Data utilities: preprocessing, augmentation, and dataset definitions."""

import os
import glob
import cv2
import numpy as np
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# Color to class mapping
BGR2IDX = {
    (0, 0, 0): 0,      # Black -> background
    (0, 0, 255): 1,    # Red (BGR) -> parenchyma
    (0, 255, 0): 2,    # Green -> CSP
    (255, 0, 0): 3,    # Blue (BGR) -> LV
}

IDX2RGB = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}


def color_mask_to_index(mask_bgr: np.ndarray) -> np.ndarray:
    """
    Convert color mask to single-channel index map.
    
    Args:
        mask_bgr: BGR color mask or single-channel mask
        
    Returns:
        Single-channel array with values 0..3 (class indices)
    """
    if mask_bgr.ndim == 2:
        # Already single-channel
        idx = mask_bgr.astype(np.int64)
        return np.clip(idx, 0, 3)
    
    # Color mask: convert BGR to indices
    h, w = mask_bgr.shape[:2]
    out = np.zeros((h, w), dtype=np.int64)
    for bgr, idx in BGR2IDX.items():
        match = np.all(mask_bgr == np.array(bgr, dtype=np.uint8)[None, None, :], axis=2)
        out[match] = idx
    return out


def index_to_rgb(idx_map: np.ndarray) -> np.ndarray:
    """Convert class indices to RGB color map."""
    rgb = np.zeros((*idx_map.shape, 3), dtype=np.uint8)
    for class_idx, rgb_color in IDX2RGB.items():
        rgb[idx_map == class_idx] = rgb_color
    return rgb


def list_pairs(
    img_dir: str, 
    mask_dir: str, 
    exts: Tuple[str] = (".png", ".jpg", ".jpeg")
) -> List[Tuple[str, str]]:
    """
    List paired image-mask files.
    
    Args:
        img_dir: Directory containing images
        mask_dir: Directory containing masks
        exts: File extensions to search
        
    Returns:
        List of (image_path, mask_path) tuples
    """
    imgs = []
    for ext in exts:
        imgs.extend(sorted(glob.glob(os.path.join(img_dir, f"*{ext}"))))
    
    pairs = []
    for img_path in imgs:
        name = os.path.splitext(os.path.basename(img_path))[0]
        for ext in exts:
            mask_path = os.path.join(mask_dir, name + ext)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                break
    
    return pairs


def build_transforms(img_size: int, is_train: bool) -> A.Compose:
    """Build data augmentation pipeline."""
    if is_train:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size, 
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])


class SegDataset(Dataset):
    """Segmentation dataset with mask-based oversampling."""
    
    def __init__(self, pairs: List[Tuple[str, str]], transform, read_gray: bool = True):
        """
        Args:
            pairs: List of (image_path, mask_path) tuples
            transform: Albumentations transform pipeline
            read_gray: Read images as grayscale
        """
        self.pairs = pairs
        self.transform = transform
        self.read_gray = read_gray
        
        # Precompute which images contain small structures (CSP/LV) for oversampling
        self.contains_small = []
        for _, mask_path in self.pairs:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                self.contains_small.append(False)
                continue
            
            idx = color_mask_to_index(mask if mask.ndim == 3 else mask)
            has_small = np.any(idx == 2) or np.any(idx == 3)
            self.contains_small.append(bool(has_small))
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, i: int) -> Tuple:
        img_path, mask_path = self.pairs[i]
        
        # Read image
        if self.read_gray:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # Read mask
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask_raw is None or img is None:
            raise FileNotFoundError(f"Missing image or mask at {img_path} / {mask_path}")
        
        # Convert mask to indices
        mask_idx = color_mask_to_index(mask_raw)
        
        # Prepare for augmentation
        img_hwc = img if img.ndim == 3 else img[..., None]
        if img_hwc.shape[2] != 1:
            img_hwc = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2GRAY)[..., None]
        
        # Apply augmentation
        augmented = self.transform(image=img_hwc, mask=mask_idx)
        img_t = augmented['image']  # (C, H, W)
        mask_t = augmented['mask'].long()  # (H, W)
        
        # Ensure single channel
        if img_t.shape[0] != 1:
            img_t = img_t[:1, ...]
        
        return img_t, mask_t, os.path.basename(img_path)
