#!/usr/bin/env python3
"""
SVM Object Recognition - Main Entry Point
==========================================
Bachelor's Thesis: SVM for Object Recognition

This script provides a command-line interface for:
- Training the SVM classifier
- Evaluating model performance
- Generating visualizations
- Making predictions on new images

Usage:
    python main.py --train --evaluate
    python main.py --tune --evaluate
    python main.py --predict path/to/image.jpg
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_cifar10, preprocess_images, CLASS_NAMES
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
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         SVM OBJECT RECOGNITION PROJECT                           ║
║         Bachelor's Thesis - Data Analytics                       ║
║                                                                  ║
║         Using Support Vector Machine for Image Classification    ║
║         Dataset: CIFAR-10 (10 Object Classes)                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def train_pipeline(args):
    """
    Complete training pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print_banner()
    
    # ========================================
    # Step 1: Load Data
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading CIFAR-10 Dataset")
    print("=" * 60)
    
    data = load_cifar10(
        data_dir='data',
        subset_size=args.subset_size
    )
    
    # ========================================
    # Step 2: Preprocess Images
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Images")
    print("=" * 60)
    
    X_train_processed = preprocess_images(data['X_train'], grayscale=True, normalize=True)
    X_test_processed = preprocess_images(data['X_test'], grayscale=True, normalize=True)
    
    print(f"✓ Preprocessed training images: {X_train_processed.shape}")
    print(f"✓ Preprocessed test images: {X_test_processed.shape}")
    
    # ========================================
    # Step 3: Extract HOG Features
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: Extracting HOG Features")
    print("=" * 60)
    
    extractor = HOGFeatureExtractor(
        orientations=9,
        pixels_per_cell=(4, 4),
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
    extractor.save('outputs/models/feature_extractor.joblib')
    
    # ========================================
    # Step 4: Train SVM Classifier
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: Training SVM Classifier")
    print("=" * 60)
    
    classifier = SVMClassifier(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma
    )
    
    if args.tune:
        # Hyperparameter tuning
        classifier.tune_hyperparameters(
            X_train_features, 
            data['y_train'],
            cv=3
        )
    else:
        # Direct training
        classifier.train(X_train_features, data['y_train'])
    
    # Save model
    classifier.save('outputs/models/svm_classifier.joblib')
    
    # ========================================
    # Step 5: Evaluate Model
    # ========================================
    if args.evaluate:
        print("\n" + "=" * 60)
        print("STEP 5: Evaluating Model")
        print("=" * 60)
        
        results = classifier.evaluate(
            X_test_features,
            data['y_test'],
            class_names=CLASS_NAMES
        )
        
        # Update data dict for visualization
        data['X_test'] = data['X_test']  # Keep original for visualization
        
        # ========================================
        # Step 6: Generate Visualizations
        # ========================================
        print("\n" + "=" * 60)
        print("STEP 6: Generating Visualizations")
        print("=" * 60)
        
        generate_all_visualizations(data, results, output_dir='outputs/figures')
        
        # Generate HOG visualization sample
        sample_idx = np.random.randint(0, len(X_train_processed))
        sample_image = X_train_processed[sample_idx]
        _, hog_image = visualize_hog_features(sample_image, extractor)
        
        plot_hog_visualization(
            data['X_train'][sample_idx],
            hog_image,
            class_name=CLASS_NAMES[data['y_train'][sample_idx]],
            save_path='outputs/figures/08_hog_visualization.png'
        )
        
        # ========================================
        # Summary
        # ========================================
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"\n📊 Results:")
        print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"   Precision: {results['precision']*100:.2f}%")
        print(f"   Recall:    {results['recall']*100:.2f}%")
        print(f"   F1-Score:  {results['f1_score']*100:.2f}%")
        print(f"\n📁 Saved artifacts:")
        print(f"   Model:    outputs/models/svm_classifier.joblib")
        print(f"   Features: outputs/models/feature_extractor.joblib")
        print(f"   Figures:  outputs/figures/")
        print("\n" + "=" * 60)
        
        return results
    
    return None


def predict_image(image_path, show_result=True):
    """
    Predict the class of a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    show_result : bool
        Print the result
        
    Returns:
    --------
    tuple : (predicted_class_name, probabilities)
    """
    import cv2
    
    # Load model and extractor
    classifier = SVMClassifier.load('outputs/models/svm_classifier.joblib')
    extractor = HOGFeatureExtractor.load('outputs/models/feature_extractor.joblib')
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to CIFAR-10 size and convert to RGB
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale and normalize
    image_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]) / 255.0
    
    # Extract features
    features = extractor.transform(np.array([image_gray]), show_progress=False)
    
    # Predict
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    predicted_class = CLASS_NAMES[prediction]
    
    if show_result:
        print(f"\n🔮 Prediction for: {image_path}")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {probabilities[prediction]*100:.2f}%")
        print(f"\n   All probabilities:")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
            bar = '█' * int(prob * 20)
            print(f"      {name:12s}: {bar:20s} {prob*100:5.2f}%")
    
    return predicted_class, probabilities


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SVM Object Recognition - Bachelor\'s Thesis Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings
    python main.py --train --evaluate
    
    # Train with hyperparameter tuning
    python main.py --train --tune --evaluate
    
    # Train on a smaller subset (faster)
    python main.py --train --evaluate --subset 5000
    
    # Predict on a new image
    python main.py --predict path/to/image.jpg
        """
    )
    
    # Training options
    parser.add_argument('--train', action='store_true',
                        help='Train the SVM classifier')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on test set')
    
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
                        help='Predict class for a single image')
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.predict:
        predict_image(args.predict)
    elif args.train:
        train_pipeline(args)
    else:
        parser.print_help()
        print("\n💡 Quick start: python main.py --train --evaluate --subset 10000")


if __name__ == "__main__":
    main()
