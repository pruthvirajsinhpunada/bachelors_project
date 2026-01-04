# SVM Satellite Image Classification

## 🛰️ Land Cover Classification using Support Vector Machine

Bachelor's Thesis Project - Data Analytics

---

## 📋 Project Overview

This project applies **Support Vector Machine (SVM)** for classifying satellite images into different land cover types using the **EuroSAT dataset**.

### Key Features
- ✅ SVM classification with multiple kernels (Linear, RBF, Polynomial)
- ✅ Hyperplane/decision boundary visualization
- ✅ HOG (Histogram of Oriented Gradients) feature extraction
- ✅ Comprehensive evaluation metrics and visualizations

### Dataset: EuroSAT
- **Source**: Zenodo (European Research Repository)
- **URL**: https://zenodo.org/record/7711810
- **Images**: 27,000 satellite images (64x64 pixels)
- **Classes**: 10 land cover types

| Class | Description |
|-------|-------------|
| AnnualCrop | Annual crop fields |
| Forest | Forested areas |
| HerbaceousVegetation | Herbaceous vegetation |
| Highway | Roads and highways |
| Industrial | Industrial areas |
| Pasture | Pasture lands |
| PermanentCrop | Permanent crop fields |
| Residential | Residential areas |
| River | Rivers and streams |
| SeaLake | Seas and lakes |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Train the Model
```bash
# Standard training
python3 main.py --train --evaluate --subset 2000

# With kernel comparison (recommended)
python3 main.py --train --evaluate --compare-kernels --subset 2000
```

### 3. View Results
Check `outputs/figures/` for all generated visualizations.

---

## 📊 Project Results

| Kernel | Accuracy | Training Time |
|--------|----------|---------------|
| **RBF** | ~54% | 1-2s |
| Linear | ~45% | 1s |
| Polynomial | ~35% | 1-2s |

**Best Kernel**: RBF (Radial Basis Function)

---

## 📁 Project Structure

```
thesis/
├── main.py                  # Main entry point
├── src/
│   ├── satellite_loader.py  # EuroSAT dataset loader
│   ├── feature_extraction.py # HOG features
│   ├── svm_classifier.py    # SVM model
│   ├── visualization.py     # Plotting functions
│   └── kernel_visualization.py # Kernel comparison
├── notebooks/
│   └── SVM_Object_Recognition.ipynb
├── outputs/
│   ├── models/              # Saved models
│   └── figures/             # Visualizations
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Feature Extraction: HOG
Histogram of Oriented Gradients captures shape information:
- Gradient magnitude: |G| = √(Gx² + Gy²)
- Gradient direction: θ = arctan(Gy / Gx)

### 2. Classification: SVM
Support Vector Machine with kernel functions:
- **Linear**: K(x,y) = x·y
- **RBF**: K(x,y) = exp(-γ||x-y||²)
- **Polynomial**: K(x,y) = (γx·y + r)^d

### 3. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves

---

## 📈 Generated Visualizations

| File | Description |
|------|-------------|
| 00_satellite_samples.png | Sample satellite images |
| 01_sample_images.png | Preprocessed samples |
| 02_class_distribution.png | Class balance |
| 03_confusion_matrix.png | Classification results |
| 04_roc_curves.png | ROC curves |
| 09_kernel_comparison.png | Kernel comparison |
| 10_hyperplane_decision_boundaries.png | Hyperplane visualization |
| 11_kernel_theory.png | SVM theory |

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **scikit-learn**: SVM, metrics
- **scikit-image**: HOG features
- **Matplotlib/Seaborn**: Visualizations
- **NumPy/Pandas**: Data processing
- **Jupyter Notebook**: Interactive analysis

---

## 📚 References

1. EuroSAT Dataset: Helber et al., 2019
2. HOG Features: Dalal & Triggs, 2005
3. SVM: Cortes & Vapnik, 1995

---

---

## 👨‍🎓 Author

**Punada Pruthvirajsinh**  
Bachelor's Thesis - Data Analytics

**GitHub**: https://github.com/pruthvirajsinhpunada/bachelors_project

---

## 📓 Interactive Notebook

You can explore the complete analysis, theory, and live code in the interactive Jupyter Notebook.

### How to Run Locally:
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/SVM_Object_Recognition.ipynb`
3. **IMPORTANT**: If the plots don't show, run the **"Setup & Imports"** cell at the top first!

### Online Preview:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pruthvirajsinhpunada/bachelors_project/main?labpath=notebooks%2FSVM_Object_Recognition.ipynb)

- **Interactive Jupyter Notebook**: [Launch on Binder](https://mybinder.org/v2/gh/pruthvirajsinhpunada/bachelors_project/main?labpath=notebooks%2FSVM_Object_Recognition_Fixed.ipynb)
- **Google Colab (Recommended)**: [Open in Colab](https://colab.research.google.com/github/pruthvirajsinhpunada/bachelors_project/blob/main/notebooks/SVM_Object_Recognition_Fixed.ipynb)
- **Static Preview**: [View on nbviewer](https://nbviewer.org/github/pruthvirajsinhpunada/bachelors_project/blob/main/notebooks/SVM_Object_Recognition_Fixed.ipynb) (Fixed Version)
