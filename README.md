# Fetal Brain Structure Segmentation

Semantic segmentation of key fetal brain structures (CSP, LV, parenchyma) in trans-thalamic ultrasound images using deep learning with U-Net and Attention Gates.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Master's Thesis Project**  
Data Science · Sapienza Università di Roma

---

## About

This project presents a deep learning approach for automated segmentation of fetal brain structures in trans-thalamic ultrasound images. The system identifies and segments three clinically important anatomical regions:

- **Parenchyma** (brain tissue)
- **CSP** (Cavum Septum Pellucidum) - small midline structure
- **LV** (Lateral Ventricle) - fluid-filled chambers

The model combines U-Net architecture with attention gates to improve feature extraction, particularly for small anatomical structures that are critical for prenatal diagnosis. This work addresses class imbalance through weighted sampling and multi-objective loss functions, achieving robust segmentation performance on challenging ultrasound data.

---

## Overview

This project implements a deep learning-based segmentation system for analyzing fetal brain ultrasound images. The model identifies and segments three key brain structures:

- **Parenchyma** (brain tissue)
- **CSP** (Cavum Septum Pellucidum) - small midline structure
- **LV** (Lateral Ventricle) - fluid-filled chambers

### Key Features

- U-Net with Attention Gates — improved feature extraction with attention mechanisms  
- Multi-Loss Training — combined Dice + Focal + Cross-Entropy losses for robust learning  
- Class Imbalance Handling — weighted sampling and loss functions for small structures  
- Smart Post-Processing — morphological operations to refine predictions  
- Production-Ready — model checkpointing, metrics tracking, inference pipeline  

### Performance Metrics

- **IoU for Small Structures**: Focus on CSP and LV segmentation accuracy
- **Weighted Dice Loss**: Emphasizes rare but clinically important structures
- **Evaluation on Test Set**: IoU/Dice scores for each class

---

## Project Structure

```
fetal-brain-seg/
├── src/                          # Source code
│   ├── config.py                 # Configuration and hyperparameters
│   ├── data.py                   # Dataset and data augmentation
│   ├── model.py                  # U-Net with Attention Gates
│   ├── losses.py                 # Focal, Dice, and combined losses
│   ├── metrics.py                # Evaluation metrics (IoU, Dice)
│   └── postprocess.py            # Post-processing utilities
│
├── scripts/                      # Utility scripts
│   ├── download_dataset.ps1      # Download and extract Zenodo dataset
│   └── smoke.py                  # Quick import and forward pass test
│
├── configs/                      # Configuration files
│   └── default_config.yaml       # Default hyperparameters
│
├── data/                         # Data directory (create symbolic link or copy)
│   ├── images/                   # Input ultrasound images
│   └── masks/                    # Segmentation masks
│
├── outputs/                      # Model checkpoints and results
│   └── best_unet.pt             # Best model weights
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, CPU-only mode supported)
- 8 GB RAM minimum

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TubaSid/fetal-brain-structure-segmentation.git
   cd fetal-brain-structure-segmentation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place training images in `data/images/`
   - Place corresponding masks in `data/masks/`
   - Ensure image-mask pairs have matching filenames

### Training

```python
from src.config import Config
from src.data import list_pairs, SegDataset, build_transforms
from src.model import UNet
import torch
from torch.utils.data import DataLoader

# Setup
cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
pairs = list_pairs(cfg.IMG_DIR, cfg.MASK_DIR)
train_tf = build_transforms(cfg.img_size, is_train=True)
dataset = SegDataset(pairs, transform=train_tf)
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# Model
model = UNet(in_channels=cfg.in_channels, n_classes=cfg.n_classes).to(device)

# Training loop (run train.py for complete pipeline)
```

### Inference

```python
import torch
import cv2
from src.model import UNet
from src.data import build_transforms, color_mask_to_index
from src.postprocess import postprocess_pred

# Load model
model = UNet(in_channels=1, n_classes=4).to(device)
ckpt = torch.load('outputs/best_unet.pt', map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()

# Prepare image
tf = build_transforms(img_size=512, is_train=False)
img = cv2.imread('data/images/sample.png', cv2.IMREAD_GRAYSCALE)
aug = tf(image=img[..., None], mask=np.zeros((img.shape[:2])))
x = aug['image'].unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    logits = model(x)
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

# Post-process
pred_refined = postprocess_pred(pred)
```

---

## Dataset Information

### Image Format
- **Type**: Grayscale ultrasound images
- **Size**: 512×512 pixels (configurable)
- **Format**: PNG or JPEG

### Mask Format (Color-Coded)
- **Background**: Black (0, 0, 0)
- **Parenchyma**: Red (255, 0, 0) in RGB
- **CSP**: Green (0, 255, 0)
- **LV**: Blue (0, 0, 255)

### Class Distribution
The dataset exhibits class imbalance with background and parenchyma being dominant. The model uses:
- Weighted random sampling during training
- Focal loss to down-weight easy examples
- Dice loss to focus on small structures

---

## Data Availability

This repository does not include the dataset because it is large and inappropriate to store in Git. Please obtain the fetal head ultrasound dataset from the peer‑reviewed source below and place images and masks under the local `data/` directory (which is gitignored by default):

- Images: `data/images/`
- Masks: `data/masks/`

The dataset described in the paper is publicly available on Zenodo:

- Alzubaidi M.H., Agus M., Makhlouf M., Anver F., Alyafei K. Large‑scale annotation dataset for fetal head biometry in ultrasound images. Zenodo; 2023. doi:10.5281/zenodo.8265464. https://doi.org/10.5281/zenodo.8265464

If your dataset paths differ, update `src/config.py` (`DATA_ROOT`, `IMG_DIR`, `MASK_DIR`) accordingly.

---

## Getting the Dataset (Windows PowerShell)

You need direct file URLs from the Zenodo record (DOI: 10.5281/zenodo.8265464). On the Zenodo page, right‑click each file’s download button and copy the link address. Then use one of the options below.

Option A: Use the helper script

```powershell
# From repo root
pwsh -File scripts/download_dataset.ps1 -OutputDir data `
   -Urls "https://zenodo.org/records/8265464/files/TT.zip?download=1" `
            "https://zenodo.org/records/8265464/files/TV.zip?download=1"

# After extraction, place images under data/images and masks under data/masks
# or point src/config.py to the extracted structure.
```

Option B: Manual download (example)

```powershell
# Create folders
New-Item -ItemType Directory -Force -Path data | Out-Null
New-Item -ItemType Directory -Force -Path data\archives | Out-Null

# Replace with actual direct file URLs from Zenodo
$urls = @(
   "https://zenodo.org/records/8265464/files/TT.zip?download=1",
   "https://zenodo.org/records/8265464/files/TV.zip?download=1"
)

# Download
$urls | ForEach-Object {
   $name = ([System.Uri]$_).Segments[-1]
   if ($name.Contains('?')) { $name = $name.Split('?')[0] }
   Invoke-WebRequest -Uri $_ -OutFile ("data/archives/" + $name) -UseBasicParsing
}

# Extract
Get-ChildItem data/archives/*.zip | ForEach-Object {
   Expand-Archive -LiteralPath $_.FullName -DestinationPath data -Force
}
```

---

## Citation

If you use this code or reproduce results with the referenced dataset, please cite the following article:

Alzubaidi M., Agus M., Makhlouf M., Anver F., Alyafei K., Househ M. Large‑scale annotation dataset for fetal head biometry in ultrasound images. Data in Brief. 2023;51:109708. doi:10.1016/j.dib.2023.109708. PMCID: PMC10630602. https://pmc.ncbi.nlm.nih.gov/articles/PMC10630602/

Additionally, cite the dataset record where appropriate:

Alzubaidi M.H., Agus M., Makhlouf M., Anver F., Alyafei K. Large‑scale annotation dataset for fetal head biometry in ultrasound images. Zenodo; 2023. doi:10.5281/zenodo.8265464. https://doi.org/10.5281/zenodo.8265464

---

## Why `data/` exists

- `data/`: A local, user‑owned staging area for images and masks used in examples and scripts. It is intentionally ignored by Git (see `.gitignore`) to avoid uploading large or sensitive datasets. You should download data from the cited source and place it here, or point the configuration to your own dataset location.

---

## Model Architecture

### U-Net with Attention Gates

```
Input (B, 1, 512, 512)
    ↓
Encoder (4 levels with MaxPool)
    ↓
Bottleneck (16× filters)
    ↓
Decoder (4 levels) + Attention Gates
    ↓
Output (B, 4, 512, 512) - logits per class
```

**Key Components**:
- **Encoder**: Progressive downsampling with skip connections
- **Attention Gates**: Weight skip connections based on decoder features
- **Decoder**: Transpose convolutions with attention-weighted concatenation
- **Output**: 4-channel logits for softmax classification

---

## Loss Functions

The model uses a **combined loss** function:

```
Loss = (1 - dice_weight - focal_weight) × CE_Loss
       + focal_weight × Focal_Loss
       + dice_weight × Dice_Loss
```

**Default Weights**:
- Cross-Entropy: 40%
- Focal Loss: 20% (for hard negatives)
- Dice Loss: 40% (for boundary alignment)

**Class Weights**:
- Background: 0.0 (ignored in checkpoint metric)
- Parenchyma: 1.0
- CSP: 2.0 (emphasize small structures)
- LV: 2.0 (emphasize small structures)

---

## Metrics

### Per-Class Metrics
- **IoU (Intersection over Union)**: Strict boundary alignment
- **Dice Coefficient**: F1-like score for segmentation
- **Specificity/Sensitivity**: Class-wise confusion matrix analysis

### Model Selection
The model is selected based on a **weighted IoU score**:
```
Checkpoint Metric = 0×bg_iou + 1×par_iou + 2×csp_iou + 2×lv_iou
```
This emphasizes CSP and LV segmentation quality.

---

## Configuration

Edit `src/config.py` to customize:

```python
cfg = Config()
cfg.img_size = 512          # Input image size
cfg.batch_size = 8          # Batch size for training
cfg.epochs = 60             # Training epochs
cfg.lr = 3e-4               # Learning rate
cfg.dice_weight = 0.4       # Dice loss weight
cfg.focal_weight = 0.2      # Focal loss weight
cfg.oversample_factor = 3.0 # Oversample ratio for CSP/LV images
```

---

## Training Tips

1. **Data Augmentation**: Horizontal flip, rotation, brightness/contrast adjustments
2. **Class Imbalance**: Use weighted sampling + focal loss
3. **Early Stopping**: Monitor validation loss/metric every epoch
4. **Learning Rate Schedule**: Optional: reduce LR on plateau
5. **Mixed Precision**: Enable AMP for faster training on GPUs

---

## Post-Processing

The inference pipeline includes:

1. **Parenchyma Cleaning**
   - Morphological closing to fill holes
   - Keep only the largest connected component

2. **CSP/LV Refinement**
   - Keep components > min_area pixels
   - Constrain to parenchyma region
   - Limit to top-K largest components

3. **Probability-Based Filtering** (optional)
   - Keep components with high confidence
   - Use margin between predicted class and alternatives

---

## Related Work

- **Original Paper**: U-Net (Ronneberger et al., 2015)
- **Attention Mechanisms**: Attention U-Net (Oktay et al., 2018)
- **Focal Loss**: Focal Loss for Dense Object Detection (Lin et al., 2017)
- **Dice Loss**: Dice Loss for Medical Image Segmentation

---

## License

MIT License - Feel free to use this project for research and commercial purposes.

---

## Author

**Tuba Siddiqui**

Master's Thesis Project  
Data Science Degree  
Sapienza Università di Roma

For questions or contributions, please open an issue or submit a pull request.

---

## Acknowledgments

- Master's thesis research conducted at Sapienza Università di Roma
- Dataset sourced from fetal ultrasound clinical studies
- Implementation inspired by community-driven medical imaging projects
- Thanks to PyTorch and the computer vision community

---

## Contact

- **Email**: tubaasid@gmail.com
- **Likedin**: linkedin.com/in/tubasid

---

Built with PyTorch
