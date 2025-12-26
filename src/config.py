"""Configuration for fetal brain segmentation model."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    """Configuration for training and inference."""
    
    # Data paths
    DATA_ROOT: str = "./data"
    IMG_DIR: str = "images"
    MASK_DIR: str = "masks"
    OUT_DIR: str = "./outputs"
    
    # Model architecture
    img_size: int = 512
    in_channels: int = 1
    n_classes: int = 4  # 0=background, 1=parenchyma, 2=CSP, 3=LV
    
    # Training hyperparameters
    batch_size: int = 8
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 2
    amp: bool = True  # Automatic Mixed Precision
    
    # Loss function weights
    dice_weight: float = 0.4
    focal_weight: float = 0.2
    ce_weight_smoothing: float = 0.0  # Smooth class weights (0..0.1)
    
    # Imbalance handling
    oversample_factor: float = 3.0  # Weight boost for CSP/LV samples
    
    # Checkpointing metric (emphasizes small structures)
    ckpt_metric_weights: Dict[int, float] = field(
        default_factory=lambda: {0: 0.0, 1: 1.0, 2: 2.0, 3: 2.0}
    )
    
    # Post-processing parameters
    dilate_px: int = 10
    par_kernel: int = 7
    small_min_area: int = 15
    small_keep_k: int = 2
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.img_size > 0, "img_size must be positive"
        assert self.n_classes >= 2, "n_classes must be >= 2"
        assert 0 <= self.ce_weight_smoothing <= 1, "ce_weight_smoothing must be in [0, 1]"
        assert 0 < self.dice_weight + self.focal_weight < 1, "Loss weights must sum to < 1"
