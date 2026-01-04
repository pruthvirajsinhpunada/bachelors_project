#!/usr/bin/env python3
"""
SVM Satellite Image Classification
===================================
Bachelor's Thesis: Land Cover Classification using SVM

This project applies Support Vector Machine (SVM) for satellite image 
classification using the EuroSAT dataset with 10 land cover classes.

Features:
- EuroSAT satellite image classification
- SVM kernel comparison (Linear, RBF, Polynomial)
- Hyperplane/decision boundary visualization
- HOG feature extraction
- Comprehensive evaluation and visualizations

Usage:
    python main.py --train --evaluate
    python main.py --train --evaluate --compare-kernels
    python main.py --compare-kernels-only
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from satellite_loader import load_satellite_dataset, plot_satellite_samples, SATELLITE_CLASS_NAMES, preprocess_images
from feature_extraction import HOGFeatureExtractor, visualize_hog_features
from svm_classifier import SVMClassifier
from visualization import (
    generate_all_visualizations,
    plot_sample_images,
    plot_class_distribution,
    plot_hog_visualization,
    setup_figure_dir
)


def print_banner():
    """Print project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║      🛰️  SVM SATELLITE IMAGE CLASSIFICATION                          ║
║         Bachelor's Thesis - Data Analytics                           ║
║                                                                      ║
║         Using Support Vector Machine for Land Cover Classification   ║
║         Dataset: EuroSAT (10 Land Cover Classes)                     ║
║                                                                      ║
║         Features: Kernel Comparison | Hyperplane Visualization       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_kernel_comparison(X_train, y_train, X_test, y_test, class_names, output_dir='outputs/figures'):
    """
    Compare all SVM kernels and generate visualizations.
    """
    from kernel_visualization import (
        compare_kernels, plot_kernel_comparison, 
        visualize_all_kernel_boundaries, plot_kernel_theory
    )
    
    print("\n" + "=" * 70)
    print("KERNEL COMPARISON & HYPERPLANE VISUALIZATION")
    print("=" * 70)
    
    # Compare kernels
    results = compare_kernels(X_train, y_train, X_test, y_test, class_names)
    
    # Generate kernel comparison plot
    plot_kernel_comparison(
        results, 
        save_path=os.path.join(output_dir, '09_kernel_comparison.png')
    )
    
    # Generate hyperplane/decision boundary visualization
    visualize_all_kernel_boundaries(
        X_train, y_train, class_names,
        save_path=os.path.join(output_dir, '10_hyperplane_decision_boundaries.png')
    )
    
    # Generate kernel theory educational plot
    plot_kernel_theory(
        save_path=os.path.join(output_dir, '11_kernel_theory.png')
    )
    
    return results


def train_pipeline(args):
    """
    Complete training pipeline for satellite image classification.
    """
    print_banner()
    
    # ========================================
    # Step 1: Load Satellite Dataset
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading EuroSAT Satellite Dataset")
    print("=" * 60)
    
    data = load_satellite_dataset(
        data_dir='data',
        subset_size=args.subset_size
    )
    class_names = data['class_names']
    
    # Save satellite samples visualization
    os.makedirs('outputs/figures', exist_ok=True)
    plot_satellite_samples(
        data['X_train'], data['y_train'], class_names,
        save_path='outputs/figures/00_satellite_samples.png'
    )
    
    # ========================================
    # Step 2: Preprocess Images
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Satellite Images")
    print("=" * 60)
    
    X_train_processed = preprocess_images(data['X_train'], grayscale=False, normalize=True)
    X_test_processed = preprocess_images(data['X_test'], grayscale=False, normalize=True)
    
    print(f"✓ Preprocessed training images: {X_train_processed.shape}")
    print(f"✓ Preprocessed test images: {X_test_processed.shape}")
    
    # ========================================
    # Step 3: Extract HOG Features
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: Extracting HOG Features")
    print("=" * 60)
    
    # Adjust for satellite image size (64x64)
    img_size = X_train_processed.shape[1]
    ppc = (8, 8) if img_size >= 64 else (4, 4)
    
    extractor = HOGFeatureExtractor(
        orientations=9,
        pixels_per_cell=ppc,
        cells_per_block=(2, 2)
    )
    
    X_train_features = extractor.fit_transform(
        X_train_processed,
        apply_pca=args.use_pca,
        n_components=args.pca_components
    )
    
    X_test_features = extractor.transform(X_test_processed)
    
    print(f"✓ Training features shape: {X_train_features.shape}")
    print(f"✓ Test features shape: {X_test_features.shape}")
    
    # Save feature extractor
    os.makedirs('outputs/models', exist_ok=True)
    extractor.save('outputs/models/feature_extractor.joblib')
    
    # ========================================
    # Step 4: Kernel Comparison (if requested)
    # ========================================
    if args.compare_kernels:
        kernel_results = run_kernel_comparison(
            X_train_features, data['y_train'],
            X_test_features, data['y_test'],
            class_names,
            output_dir='outputs/figures'
        )
    
    # ========================================
    # Step 5: Train SVM Classifier
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 5: Training SVM Classifier")
    print("=" * 60)
    
    classifier = SVMClassifier(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma
    )
    
    if args.tune:
        classifier.tune_hyperparameters(
            X_train_features, 
            data['y_train'],
            cv=3
        )
    else:
        classifier.train(X_train_features, data['y_train'])
    
    # Save model
    classifier.save('outputs/models/svm_classifier.joblib')
    
    # ========================================
    # Step 6: Evaluate Model
    # ========================================
    if args.evaluate:
        print("\n" + "=" * 60)
        print("STEP 6: Evaluating Model")
        print("=" * 60)
        
        results = classifier.evaluate(
            X_test_features,
            data['y_test'],
            class_names=class_names
        )
        
        # ========================================
        # Step 7: Generate Visualizations
        # ========================================
        print("\n" + "=" * 60)
        print("STEP 7: Generating Visualizations")
        print("=" * 60)
        
        data['class_names'] = class_names
        generate_all_visualizations(data, results, output_dir='outputs/figures')
        
        # Generate HOG visualization sample
        sample_idx = np.random.randint(0, len(X_train_processed))
        sample_image = X_train_processed[sample_idx]
        _, hog_image = visualize_hog_features(sample_image, extractor)
        
        plot_hog_visualization(
            data['X_train'][sample_idx],
            hog_image,
            class_name=class_names[data['y_train'][sample_idx]],
            save_path='outputs/figures/08_hog_visualization.png'
        )
        
        # ========================================
        # Summary
        # ========================================
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE - SUMMARY")
        print("=" * 70)
        print(f"\n📊 Dataset: EuroSAT Satellite Images")
        print(f"   Land Cover Classes: {len(class_names)}")
        print(f"\n📈 Results:")
        print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"   Precision: {results['precision']*100:.2f}%")
        print(f"   Recall:    {results['recall']*100:.2f}%")
        print(f"   F1-Score:  {results['f1_score']*100:.2f}%")
        print(f"\n🔧 Model Configuration:")
        print(f"   Kernel: {args.kernel.upper()}")
        print(f"   C: {args.C}")
        print(f"\n📁 Saved artifacts:")
        print(f"   Model:    outputs/models/svm_classifier.joblib")
        print(f"   Features: outputs/models/feature_extractor.joblib")
        print(f"   Figures:  outputs/figures/")
        
        if args.compare_kernels:
            print(f"\n🔬 Kernel Comparison:")
            print(f"   See: outputs/figures/09_kernel_comparison.png")
            print(f"   See: outputs/figures/10_hyperplane_decision_boundaries.png")
            print(f"   See: outputs/figures/11_kernel_theory.png")
        
        print("\n" + "=" * 70)
        
        return results
    
    return None


def run_standalone_kernel_comparison(args):
    """Run standalone kernel comparison visualization."""
    from kernel_visualization import (
        visualize_all_kernel_boundaries, 
        plot_kernel_theory,
        compare_kernels,
        plot_kernel_comparison
    )
    
    print_banner()
    
    print("\nRunning Kernel Comparison on Satellite Data...")
    
    # Load dataset
    data = load_satellite_dataset(data_dir='data', subset_size=args.subset_size or 2000)
    
    X_train_processed = preprocess_images(data['X_train'], grayscale=False, normalize=True)
    X_test_processed = preprocess_images(data['X_test'], grayscale=False, normalize=True)
    
    extractor = HOGFeatureExtractor(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    X_train_features = extractor.fit_transform(X_train_processed)
    X_test_features = extractor.transform(X_test_processed)
    
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Compare kernels
    results = compare_kernels(
        X_train_features, data['y_train'],
        X_test_features, data['y_test'],
        data['class_names']
    )
    
    # Plot comparison
    plot_kernel_comparison(results, save_path='outputs/figures/09_kernel_comparison.png')
    
    # Generate hyperplane viz
    visualize_all_kernel_boundaries(
        X_train_features, data['y_train'], data['class_names'],
        save_path='outputs/figures/10_hyperplane_decision_boundaries.png'
    )
    
    # Generate theory plot
    plot_kernel_theory(save_path='outputs/figures/11_kernel_theory.png')
    
    print("\n✓ Kernel comparison visualizations saved to outputs/figures/")


def predict_image(image_path, show_result=True):
    """Predict the land cover class of a satellite image."""
    import cv2
    
    classifier = SVMClassifier.load('outputs/models/svm_classifier.joblib')
    extractor = HOGFeatureExtractor.load('outputs/models/feature_extractor.joblib')
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image / 255.0
    
    features = extractor.transform(np.array([image_rgb]), show_progress=False)
    
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    predicted_class = SATELLITE_CLASS_NAMES[prediction]
    
    if show_result:
        print(f"\n�️ Satellite Image Prediction")
        print(f"   Image: {image_path}")
        print(f"   Predicted Land Cover: {predicted_class}")
        print(f"   Confidence: {probabilities[prediction]*100:.2f}%")
        print(f"\n   All probabilities:")
        for i, (name, prob) in enumerate(zip(SATELLITE_CLASS_NAMES, probabilities)):
            bar = '█' * int(prob * 20)
            print(f"      {name:22s}: {bar:20s} {prob*100:5.2f}%")
    
    return predicted_class, probabilities


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SVM Satellite Image Classification - Bachelor\'s Thesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard training
    python main.py --train --evaluate --subset 2000
    
    # Training with kernel comparison (recommended)
    python main.py --train --evaluate --compare-kernels --subset 2000
    
    # With hyperparameter tuning
    python main.py --train --tune --evaluate --subset 1000
    
    # Just kernel comparison visualization
    python main.py --compare-kernels-only
    
    # Predict land cover for an image
    python main.py --predict path/to/satellite_image.jpg
        """
    )
    
    # Training options
    parser.add_argument('--train', action='store_true',
                        help='Train the SVM classifier')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on test set')
    parser.add_argument('--compare-kernels', action='store_true',
                        help='Compare Linear, RBF, and Polynomial kernels')
    
    # Standalone visualization options
    parser.add_argument('--compare-kernels-only', action='store_true',
                        help='Only run kernel comparison without full training')
    
    # Model parameters
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=['rbf', 'linear', 'poly'],
                        help='SVM kernel type (default: rbf)')
    parser.add_argument('--C', type=float, default=10.0,
                        help='SVM regularization parameter (default: 10.0)')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='Kernel coefficient (default: scale)')
    
    # Feature extraction
    parser.add_argument('--use-pca', action='store_true',
                        help='Apply PCA dimensionality reduction')
    parser.add_argument('--pca-components', type=int, default=100,
                        help='Number of PCA components (default: 100)')
    
    # Data options
    parser.add_argument('--subset', type=int, dest='subset_size',
                        help='Use a subset of data (for faster training)')
    
    # Prediction
    parser.add_argument('--predict', type=str, metavar='IMAGE_PATH',
                        help='Predict land cover class for a satellite image')
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.predict:
        predict_image(args.predict)
    elif args.compare_kernels_only:
        run_standalone_kernel_comparison(args)
    elif args.train:
        train_pipeline(args)
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("💡 QUICK START COMMANDS:")
        print("=" * 60)
        print("\n  Standard training:")
        print("    python main.py --train --evaluate --subset 2000")
        print("\n  With kernel comparison (recommended):")
        print("    python main.py --train --evaluate --compare-kernels --subset 2000")
        print("\n  Just visualizations:")
        print("    python main.py --compare-kernels-only")
        print()


if __name__ == "__main__":
    main()
