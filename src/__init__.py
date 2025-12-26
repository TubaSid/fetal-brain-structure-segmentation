"""Fetal Brain Ultrasound Segmentation - Main Package."""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "U-Net with Attention Gates for fetal brain segmentation"

from .config import Config
from .data import SegDataset, list_pairs, build_transforms, color_mask_to_index, index_to_rgb
from .model import UNet, AttentionBlock, DoubleConv
from .losses import FocalLoss, SoftDiceLoss, combined_loss
from .metrics import confusion_matrix, iou_from_cm, dice_from_cm, metric_score
from .postprocess import postprocess_pred, refine_lv_safe

__all__ = [
    'Config',
    'SegDataset',
    'list_pairs',
    'build_transforms',
    'color_mask_to_index',
    'index_to_rgb',
    'UNet',
    'AttentionBlock',
    'DoubleConv',
    'FocalLoss',
    'SoftDiceLoss',
    'combined_loss',
    'confusion_matrix',
    'iou_from_cm',
    'dice_from_cm',
    'metric_score',
    'postprocess_pred',
    'refine_lv_safe',
]
