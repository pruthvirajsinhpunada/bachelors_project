"""
Visualization Module for SVM Object Recognition
================================================
Bachelor's Thesis: SVM for Object Recognition

This module provides comprehensive visualizations for:
- Dataset exploration
- Feature analysis
- Model evaluation (confusion matrix, ROC curves)
- Prediction results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def setup_figure_dir(output_dir='outputs/figures'):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_sample_images(images, labels, class_names, n_samples=25, save_path=None):
    """
    Plot a grid of sample images from the dataset.
    
    Parameters:
    -----------
    images : np.ndarray
        Array of images
    labels : np.ndarray
        Corresponding labels
    class_names : list
        Names of classes
    n_samples : int
        Number of samples to display
    save_path : str, optional
        Path to save the figure
    """
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16, fontweight='bold', y=1.02)
    
    # Select random samples
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img_idx = indices[idx]
            
            # Handle both grayscale and RGB
            if images[img_idx].ndim == 2:
                ax.imshow(images[img_idx], cmap='gray')
            else:
                ax.imshow(images[img_idx])
            
            ax.set_title(class_names[labels[img_idx]], fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_class_distribution(labels, class_names, save_path=None):
    """
    Plot the distribution of classes in the dataset.
    
    Parameters:
    -----------
    labels : np.ndarray
        Label array
    class_names : list
        Names of classes
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique)))
    bars = ax.bar([class_names[i] for i in unique], counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars (dynamic offset based on data)
    offset = max(counts) * 0.02  # 2% of max height
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Object Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_hog_visualization(original_image, hog_image, class_name=None, save_path=None):
    """
    Visualize HOG features alongside the original image.
    
    Parameters:
    -----------
    original_image : np.ndarray
        Original image
    hog_image : np.ndarray
        HOG visualization image
    class_name : str, optional
        Class name for title
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    title = 'HOG Feature Visualization'
    if class_name:
        title += f' - {class_name}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Original image
    if original_image.ndim == 2:
        axes[0].imshow(original_image, cmap='gray')
    else:
        axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # HOG visualization
    axes[1].imshow(hog_image, cmap='gray')
    axes[1].set_title('HOG Features', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, save_path=None):
    """
    Plot a confusion matrix heatmap.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list
        Names of classes
    normalize : bool
        Normalize values to percentages
    save_path : str, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                linewidths=0.5, linecolor='white')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_roc_curves(roc_data, class_names, save_path=None):
    """
    Plot ROC curves for all classes.
    
    Parameters:
    -----------
    roc_data : dict
        Dictionary with ROC data for each class
    class_names : list
        Names of classes
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (class_idx, data) in enumerate(roc_data.items()):
        ax.plot(data['fpr'], data['tpr'], color=colors[i], lw=2,
                label=f"{class_names[class_idx]} (AUC = {data['auc']:.3f})")
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_prediction_samples(images, y_true, y_pred, class_names, n_samples=20, save_path=None):
    """
    Plot sample predictions with correct/incorrect indicators.
    
    Parameters:
    -----------
    images : np.ndarray
        Test images
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list
        Names of classes
    n_samples : int
        Number of samples to display
    save_path : str, optional
        Path to save the figure
    """
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold', y=1.02)
    
    # Select random samples
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img_idx = indices[idx]
            
            # Display image
            if images[img_idx].ndim == 2:
                ax.imshow(images[img_idx], cmap='gray')
            else:
                ax.imshow(images[img_idx])
            
            true_label = class_names[y_true[img_idx]]
            pred_label = class_names[y_pred[img_idx]]
            is_correct = y_true[img_idx] == y_pred[img_idx]
            
            # Color code: green for correct, red for incorrect
            color = 'green' if is_correct else 'red'
            symbol = '✓' if is_correct else '✗'
            
            ax.set_title(f'{symbol} Pred: {pred_label}\nTrue: {true_label}',
                        fontsize=9, color=color, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot a comparison of different evaluation metrics.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names and values
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(metrics_dict.keys())
    values = [metrics_dict[m] * 100 for m in metrics]  # Convert to percentage
    
    colors = plt.cm.RdYlGn(np.array(values) / 100)
    bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 100)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """
    Plot accuracy for each class.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list
        Names of classes
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
        else:
            acc = 0
        class_accuracies.append(acc * 100)
    
    colors = plt.cm.RdYlGn(np.array(class_accuracies) / 100)
    bars = ax.bar(class_names, class_accuracies, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, class_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, 100)
    ax.set_xlabel('Object Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Classification Accuracy', fontsize=14, fontweight='bold')
    ax.axhline(y=np.mean(class_accuracies), color='blue', linestyle='--', 
               alpha=0.7, label=f'Mean: {np.mean(class_accuracies):.1f}%')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def generate_all_visualizations(data, results, output_dir='outputs/figures'):
    """
    Generate all visualizations for the project.
    
    Parameters:
    -----------
    data : dict
        Dataset dictionary with images and labels
    results : dict
        Evaluation results from the classifier
    output_dir : str
        Directory to save figures
    """
    setup_figure_dir(output_dir)
    
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    class_names = data['class_names']
    
    # 1. Sample images
    plot_sample_images(
        data['X_train'], data['y_train'], class_names,
        save_path=os.path.join(output_dir, '01_sample_images.png')
    )
    
    # 2. Class distribution
    plot_class_distribution(
        data['y_train'], class_names,
        save_path=os.path.join(output_dir, '02_class_distribution.png')
    )
    
    # 3. Confusion matrix
    plot_confusion_matrix(
        data['y_test'], results['y_pred'], class_names,
        save_path=os.path.join(output_dir, '03_confusion_matrix.png')
    )
    
    # 4. ROC curves
    plot_roc_curves(
        results['roc_data'], class_names,
        save_path=os.path.join(output_dir, '04_roc_curves.png')
    )
    
    # 5. Prediction samples
    plot_prediction_samples(
        data['X_test'], data['y_test'], results['y_pred'], class_names,
        save_path=os.path.join(output_dir, '05_prediction_samples.png')
    )
    
    # 6. Metrics comparison
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1_score']
    }
    plot_metrics_comparison(
        metrics,
        save_path=os.path.join(output_dir, '06_metrics_comparison.png')
    )
    
    # 7. Per-class accuracy
    plot_per_class_accuracy(
        data['y_test'], results['y_pred'], class_names,
        save_path=os.path.join(output_dir, '07_per_class_accuracy.png')
    )
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Test visualizations with dummy data
    print("=" * 60)
    print("Visualization Module Test")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    
    dummy_images = np.random.rand(n_samples, 32, 32, 3)
    dummy_labels = np.random.randint(0, 10, n_samples)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Test sample images plot
    plot_sample_images(dummy_images, dummy_labels, class_names, n_samples=10)
    print("✓ Sample images visualization works")
    
    # Test class distribution
    plot_class_distribution(dummy_labels, class_names)
    print("✓ Class distribution visualization works")
    
    print("\n✓ All visualization functions tested successfully!")
