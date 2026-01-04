# 🎯 SVM Object Recognition

## Bachelor's Thesis - Data Analytics Final Project

A professional implementation of **Support Vector Machine (SVM)** for image classification and object recognition using the CIFAR-10 dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Project Overview

| Component | Description |
|-----------|-------------|
| **Algorithm** | Support Vector Machine (Supervised Learning) |
| **Application** | Image Segmentation & Object Recognition |
| **Dataset** | CIFAR-10 (60,000 images, 10 classes) |
| **Features** | HOG (Histogram of Oriented Gradients) |
| **Language** | Python |

### CIFAR-10 Classes
| ID | Class |
|----|-------|
| 0 | ✈️ Airplane |
| 1 | 🚗 Automobile |
| 2 | 🐦 Bird |
| 3 | 🐱 Cat |
| 4 | 🦌 Deer |
| 5 | 🐕 Dog |
| 6 | 🐸 Frog |
| 7 | 🐴 Horse |
| 8 | 🚢 Ship |
| 9 | 🚛 Truck |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd thesis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Quick training with subset (recommended for first run)
python main.py --train --evaluate --subset 10000

# Full training (takes longer)
python main.py --train --evaluate

# With hyperparameter tuning
python main.py --train --tune --evaluate --subset 5000
```

### 3. Make Predictions

```bash
# Predict class for a new image
python main.py --predict path/to/your/image.jpg
```

---

## 📁 Project Structure

```
thesis/
├── data/                         # CIFAR-10 dataset (auto-downloaded)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── feature_extraction.py     # HOG feature extraction
│   ├── svm_classifier.py         # SVM model implementation
│   └── visualization.py          # Plotting functions
├── notebooks/
│   └── SVM_Object_Recognition.ipynb  # Jupyter analysis notebook
├── outputs/
│   ├── models/                   # Saved trained models
│   └── figures/                  # Generated visualizations
├── main.py                       # CLI entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 🔬 Methodology

### 1. Feature Extraction: HOG

**Histogram of Oriented Gradients (HOG)** captures edge orientations and local shape information:

```
Input Image (32×32) → Gradient Calculation → Cell Histograms → Block Normalization → Feature Vector
```

| Parameter | Value |
|-----------|-------|
| Orientations | 9 |
| Pixels per Cell | 4×4 |
| Cells per Block | 2×2 |
| Features per Image | 1,764 |

### 2. SVM Classification

We use a **Radial Basis Function (RBF)** kernel for non-linear classification:

- **One-vs-Rest (OvR)** strategy for multi-class
- **Probability estimates** enabled for confidence scores
- **GridSearchCV** for hyperparameter optimization

### 3. Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class performance
- **ROC Curves**: AUC for each class

---

## 📊 Sample Results

After training with default settings, you can expect:

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 55-65% |
| Precision | 55-65% |
| Recall | 55-65% |
| F1-Score | 55-65% |

> **Note**: Deep learning models achieve 90%+ on CIFAR-10, but SVM demonstrates classical ML fundamentals effectively.

---

## 📈 Generated Visualizations

After running with `--evaluate`, the following plots are generated in `outputs/figures/`:

| File | Description |
|------|-------------|
| `01_sample_images.png` | Sample images from dataset |
| `02_class_distribution.png` | Class balance in training set |
| `03_confusion_matrix.png` | Classification confusion matrix |
| `04_roc_curves.png` | ROC curves for each class |
| `05_prediction_samples.png` | Sample predictions (correct/incorrect) |
| `06_metrics_comparison.png` | Accuracy, Precision, Recall, F1 comparison |
| `07_per_class_accuracy.png` | Per-class accuracy breakdown |
| `08_hog_visualization.png` | HOG feature visualization |

---

## 🛠️ Command-Line Options

```bash
python main.py [OPTIONS]

Training:
  --train               Train the SVM classifier
  --tune                Enable hyperparameter tuning (GridSearchCV)
  --evaluate            Evaluate on test set and generate visualizations

Model Parameters:
  --kernel {rbf,linear,poly}    SVM kernel type (default: rbf)
  --C FLOAT                     Regularization parameter (default: 10.0)
  --gamma {scale,auto,FLOAT}    Kernel coefficient (default: scale)

Feature Extraction:
  --use-pca             Apply PCA dimensionality reduction
  --pca-components N    Number of PCA components (default: 100)

Data:
  --subset N            Use a subset of N samples (faster training)

Prediction:
  --predict IMAGE_PATH  Classify a single image
```

---

## 📚 Theoretical Background

### Support Vector Machine (SVM)

SVM finds the optimal hyperplane that separates classes with maximum margin:

- **Linear SVM**: Works when data is linearly separable
- **Kernel Trick**: Maps data to higher dimensions for non-linear boundaries
- **RBF Kernel**: `K(x, y) = exp(-γ||x-y||²)` - creates circular decision boundaries

### Why HOG + SVM?

1. **HOG** captures edge and gradient structure (object shapes)
2. **SVM** excels at finding optimal decision boundaries
3. **Combination** is robust and interpretable (unlike black-box deep learning)

---

## 📝 Files Description

| File | Lines | Description |
|------|-------|-------------|
| `data_loader.py` | ~230 | CIFAR-10 download, loading, preprocessing |
| `feature_extraction.py` | ~220 | HOG feature extraction with PCA option |
| `svm_classifier.py` | ~280 | SVM training, tuning, evaluation |
| `visualization.py` | ~350 | 8 visualization functions |
| `main.py` | ~260 | CLI interface and training pipeline |

---

## 🎓 Author

**Bachelor's Thesis Project**  
Data Analytics Program  
2026

---

## 📄 License

This project is for educational purposes as part of a Bachelor's thesis in Data Analytics.
