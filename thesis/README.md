# Master's Thesis

**Title:** Semantic Segmentation of Key Fetal Brain Structures in Trans-Thalamic Ultrasound Images Using Deep Learning

**Author:** Tuba Siddiqui  
**Institution:** Sapienza Università di Roma  
**Degree:** Master of Science in Data Science  
**Year:** 2025

---

## Contents

This directory contains the complete thesis documentation:

- **`thesis.pdf`** - Full thesis document
- **`thesis_latex_source.zip`** - LaTeX source files for reproducibility
- **`presentation.ppt`** - Thesis defense presentation

---

## Abstract

This thesis presents a deep learning approach for automated segmentation of fetal brain structures in trans-thalamic ultrasound images. Using a U-Net architecture enhanced with attention gates, the model achieves robust segmentation of three key anatomical regions: brain parenchyma, Cavum Septum Pellucidum (CSP), and Lateral Ventricles (LV).

The research addresses class imbalance through:
- Weighted random sampling
- Combined loss functions (Cross-Entropy, Focal, and Dice losses)
- Post-processing refinements

Performance is evaluated using IoU and Dice metrics, with emphasis on small structure segmentation quality critical for prenatal diagnosis.

---

## Key Contributions

1. Implementation of attention-augmented U-Net for medical ultrasound segmentation
2. Multi-objective loss function combining CE, Focal, and Dice losses
3. Class imbalance handling through weighted sampling and loss weighting
4. Post-processing pipeline for morphological refinement
5. Comprehensive evaluation on large-scale fetal ultrasound dataset (3,832 images)

---

## Dataset

The model is trained and evaluated on the publicly available dataset:

**Alzubaidi et al. (2023)** - Large-scale annotation dataset for fetal head biometry in ultrasound images  
DOI: [10.5281/zenodo.8265464](https://doi.org/10.5281/zenodo.8265464)

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{siddiqui2025fetal,
  author  = {Siddiqui, Tuba},
  title   = {Semantic Segmentation of Key Fetal Brain Structures in Trans-Thalamic Ultrasound Images Using Deep Learning},
  school  = {Sapienza Università di Roma},
  year    = {2025},
  type    = {Master's Thesis},
  note    = {Data Science}
}
```

---

## Compiling LaTeX Source

To compile the thesis from source:

1. Extract `thesis_latex_source.zip`
2. Ensure you have a LaTeX distribution installed (TeX Live, MiKTeX, or Overleaf)
3. Compile the main `.tex` file:
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

---

## License

The thesis document and associated materials are © 2025 Tuba Siddiqui. The accompanying code implementation is licensed under MIT License (see parent directory).
