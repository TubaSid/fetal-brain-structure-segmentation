#!/usr/bin/env python
"""
Inference script for fetal brain segmentation.

Usage:
    python infer.py --image data/images/sample.png --mask data/masks/sample.png --model outputs/best_unet.pt
"""

import sys
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src import (
    UNet,
    build_transforms,
    color_mask_to_index,
    index_to_rgb,
    postprocess_pred,
    iou_from_cm,
    dice_from_cm,
    confusion_matrix,
)


def infer(image_path: str, mask_path: str, model_path: str, device='cuda'):
    """Run inference on a single image."""
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, n_classes=4).to(device)
    
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model from {model_path}")
    print(f"Model score: {ckpt.get('score', 'N/A'):.4f}")
    
    # Load and preprocess
    tf = build_transforms(img_size=512, is_train=False)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    mask_gt = color_mask_to_index(mask_raw)
    
    # Transform
    aug = tf(image=img[..., None], mask=mask_gt)
    x = aug['image'].unsqueeze(0).to(device).float()
    y_gt = aug['mask'].cpu().numpy()
    
    # Inference
    with torch.no_grad():
        logits = model(x)
    
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_raw = torch.argmax(logits, dim=1)[0].cpu().numpy()
    
    # Post-process
    pred_refined = postprocess_pred(pred_raw)
    
    # Metrics
    pred_t = torch.from_numpy(pred_raw).long()
    gt_t = torch.from_numpy(y_gt).long()
    
    cm = confusion_matrix(pred_t, gt_t, n_classes=4)
    iou = iou_from_cm(cm)
    dice = dice_from_cm(cm)
    
    iou_refined = None
    dice_refined = None
    if not np.array_equal(pred_raw, pred_refined):
        pred_ref_t = torch.from_numpy(pred_refined).long()
        cm_ref = confusion_matrix(pred_ref_t, gt_t, n_classes=4)
        iou_refined = iou_from_cm(cm_ref)
        dice_refined = dice_from_cm(cm_ref)
    
    # Print results
    print("\n" + "="*50)
    print("METRICS")
    print("="*50)
    class_names = ['Background', 'Parenchyma', 'CSP', 'LV']
    
    print("\nRaw Prediction:")
    print(f"{'Class':<15} {'IoU':<10} {'Dice':<10}")
    print("-" * 35)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {iou[i]:<10.4f} {dice[i]:<10.4f}")
    
    if iou_refined is not None:
        print("\nPost-Processed Prediction:")
        print(f"{'Class':<15} {'IoU':<10} {'Dice':<10}")
        print("-" * 35)
        for i, name in enumerate(class_names):
            print(f"{name:<15} {iou_refined[i]:<10.4f} {dice_refined[i]:<10.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original image, GT mask, Probabilities
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(index_to_rgb(y_gt))
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(probs[2], cmap='hot')  # CSP probabilities
    axes[0, 2].set_title('CSP Probability')
    axes[0, 2].axis('off')
    
    # Row 2: Raw prediction, refined prediction
    axes[1, 0].imshow(index_to_rgb(pred_raw))
    axes[1, 0].set_title('Raw Prediction')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(index_to_rgb(pred_refined))
    axes[1, 1].set_title('Post-Processed')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(probs[3], cmap='hot')  # LV probabilities
    axes[1, 2].set_title('LV Probability')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to inference_result.png")
    plt.show()
    
    return {
        'pred_raw': pred_raw,
        'pred_refined': pred_refined,
        'probs': probs,
        'iou': iou,
        'dice': dice,
        'iou_refined': iou_refined,
        'dice_refined': dice_refined,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for fetal brain segmentation')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to ground truth mask')
    parser.add_argument('--model', type=str, default='outputs/best_unet.pt', help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    results = infer(args.image, args.mask, args.model, args.device)
