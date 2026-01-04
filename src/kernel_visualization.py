"""
Kernel Comparison and Hyperplane Visualization Module
======================================================
Bachelor's Thesis: SVM for Object Recognition

This module provides:
- SVM kernel comparison (Linear, RBF, Polynomial)
- Decision boundary/hyperplane visualization
- 2D projection of high-dimensional feature space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import os


# Professional color palettes
KERNEL_COLORS = {
    'linear': '#2ecc71',    # Green
    'rbf': '#3498db',       # Blue  
    'poly': '#e74c3c'       # Red
}

CLASS_COLORS = plt.cm.tab10.colors


def compare_kernels(X_train, y_train, X_test, y_test, class_names=None):
    """
    Compare different SVM kernels on the same dataset.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    class_names : List of class names
    
    Returns:
    --------
    dict : Results for each kernel
    """
    kernels = {
        'linear': {'kernel': 'linear', 'C': 1.0},
        'rbf': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
        'poly': {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'gamma': 'scale'}
    }
    
    results = {}
    
    print("\n" + "=" * 70)
    print("SVM KERNEL COMPARISON")
    print("=" * 70)
    print(f"{'Kernel':<12} {'Accuracy':>10} {'Training Time':>15} {'Support Vectors':>18}")
    print("-" * 70)
    
    for name, params in kernels.items():
        start_time = time.time()
        
        # Train SVM
        svm = SVC(**params, random_state=42)
        svm.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        n_sv = svm.n_support_.sum()
        
        results[name] = {
            'model': svm,
            'accuracy': accuracy,
            'training_time': train_time,
            'n_support_vectors': n_sv,
            'y_pred': y_pred
        }
        
        print(f"{name.upper():<12} {accuracy*100:>9.2f}% {train_time:>14.2f}s {n_sv:>18}")
    
    print("=" * 70)
    
    # Find best kernel
    best_kernel = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\n🏆 Best Kernel: {best_kernel.upper()} ({results[best_kernel]['accuracy']*100:.2f}%)")
    
    return results


def plot_kernel_comparison(results, save_path=None):
    """
    Create a visualization comparing kernel performance.
    
    Parameters:
    -----------
    results : dict
        Results from compare_kernels()
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    kernels = list(results.keys())
    colors = [KERNEL_COLORS[k] for k in kernels]
    
    # 1. Accuracy comparison
    accuracies = [results[k]['accuracy'] * 100 for k in kernels]
    bars = axes[0].bar([k.upper() for k in kernels], accuracies, color=colors, edgecolor='white', linewidth=2)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    
    # 2. Training time comparison
    times = [results[k]['training_time'] for k in kernels]
    bars = axes[1].bar([k.upper() for k in kernels], times, color=colors, edgecolor='white', linewidth=2)
    time_offset = max(times) * 0.05  # Dynamic offset
    for bar, t in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + time_offset,
                    f'{t:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Training Time', fontsize=14, fontweight='bold', pad=15)
    
    # 3. Support vectors comparison
    svs = [results[k]['n_support_vectors'] for k in kernels]
    bars = axes[2].bar([k.upper() for k in kernels], svs, color=colors, edgecolor='white', linewidth=2)
    sv_offset = max(svs) * 0.02  # Dynamic offset
    for bar, sv in zip(bars, svs):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + sv_offset,
                    f'{sv}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('Number of Support Vectors', fontsize=14, fontweight='bold', pad=15)
    
    plt.suptitle('SVM Kernel Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    return fig


def visualize_decision_boundary(X, y, kernel='rbf', class_names=None, 
                                 n_components=2, resolution=100, save_path=None):
    """
    Visualize SVM decision boundaries in 2D using PCA projection.
    
    This demonstrates the HYPERPLANE SEGREGATION concept - how SVM
    creates decision boundaries to separate different classes.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    kernel : str
        SVM kernel type
    class_names : list
        Names of classes
    n_components : int
        PCA components (must be 2 for visualization)
    resolution : int
        Grid resolution for decision boundary
    save_path : str, optional
        Path to save the figure
    """
    print(f"\nVisualizing decision boundary for {kernel.upper()} kernel...")
    
    # Reduce to 2D using PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA variance explained: {variance_explained:.1f}%")
    
    # Train SVM on 2D data
    if kernel == 'linear':
        svm = SVC(kernel='linear', C=1.0, random_state=42)
    elif kernel == 'rbf':
        svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    else:  # poly
        svm = SVC(kernel='poly', C=1.0, degree=3, gamma='scale', random_state=42)
    
    svm.fit(X_2d, y)
    
    # Create mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Get predictions for mesh
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Create colormap
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    cmap_light = ListedColormap([(*c[:3], 0.3) for c in colors])
    cmap_bold = ListedColormap(colors)
    
    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light, levels=np.arange(-0.5, n_classes, 1))
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
    
    # Plot training points
    for i, cls in enumerate(unique_classes):
        mask = y == cls
        label = class_names[cls] if class_names else f'Class {cls}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=[colors[i]], label=label, edgecolors='white', 
                  s=60, linewidth=1, alpha=0.8)
    
    # Plot support vectors
    sv_indices = svm.support_
    ax.scatter(X_2d[sv_indices, 0], X_2d[sv_indices, 1],
              s=150, facecolors='none', edgecolors='black', linewidths=2,
              label=f'Support Vectors (n={len(sv_indices)})')
    
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'SVM Decision Boundary - {kernel.upper()} Kernel\n(Hyperplane Segregation Visualization)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def visualize_all_kernel_boundaries(X, y, class_names=None, save_path=None):
    """
    Create a side-by-side comparison of decision boundaries for all kernels.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    class_names : list
        Names of classes
    save_path : str, optional
        Path to save the figure
    """
    print("\nGenerating decision boundaries for all kernels...")
    
    # Reduce to 2D using PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    kernels = [
        ('linear', 'Linear Kernel', {'kernel': 'linear', 'C': 1.0}),
        ('rbf', 'RBF (Radial Basis Function) Kernel', {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}),
        ('poly', 'Polynomial Kernel (degree=3)', {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'gamma': 'scale'})
    ]
    
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    cmap_light = ListedColormap([(*c[:3], 0.3) for c in colors])
    
    # Create mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    for ax, (name, title, params) in zip(axes, kernels):
        # Train SVM
        svm = SVC(**params, random_state=42)
        svm.fit(X_2d, y)
        
        # Get predictions
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # Plot decision regions
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light, levels=np.arange(-0.5, n_classes, 1))
        ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
        
        # Plot points
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], 
                      edgecolors='white', s=40, linewidth=0.5, alpha=0.7)
        
        # Plot support vectors
        sv = svm.support_
        ax.scatter(X_2d[sv, 0], X_2d[sv, 1], s=100, facecolors='none', 
                  edgecolors='black', linewidths=1.5)
        
        accuracy = svm.score(X_2d, y) * 100
        ax.set_title(f'{title}\nAccuracy: {accuracy:.1f}% | SVs: {len(sv)}', 
                    fontsize=11, fontweight='bold', color=KERNEL_COLORS[name])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('SVM Kernel Comparison: Hyperplane Decision Boundaries', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


def plot_kernel_theory(save_path=None):
    """
    Create educational visualization explaining SVM kernel concepts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    np.random.seed(42)
    
    # 1. Linear separable data
    ax = axes[0, 0]
    n = 50
    X1 = np.random.randn(n, 2) + np.array([2, 2])
    X2 = np.random.randn(n, 2) + np.array([-2, -2])
    ax.scatter(X1[:, 0], X1[:, 1], c='#3498db', label='Class A', s=80, edgecolors='white')
    ax.scatter(X2[:, 0], X2[:, 1], c='#e74c3c', label='Class B', s=80, edgecolors='white')
    
    # Draw separating hyperplane
    x_line = np.linspace(-5, 5, 100)
    ax.plot(x_line, -x_line, 'k-', linewidth=2, label='Hyperplane')
    ax.fill_between(x_line, -x_line, 10, alpha=0.1, color='#3498db')
    ax.fill_between(x_line, -x_line, -10, alpha=0.1, color='#e74c3c')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title('Linear Kernel\n(Linearly Separable Data)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    
    # 2. Non-linear data (circles)
    ax = axes[0, 1]
    theta = np.linspace(0, 2*np.pi, n)
    X1 = np.column_stack([np.cos(theta) + np.random.randn(n)*0.1, 
                          np.sin(theta) + np.random.randn(n)*0.1])
    X2 = np.column_stack([3*np.cos(theta) + np.random.randn(n)*0.2, 
                          3*np.sin(theta) + np.random.randn(n)*0.2])
    ax.scatter(X1[:, 0], X1[:, 1], c='#3498db', label='Class A (inner)', s=80, edgecolors='white')
    ax.scatter(X2[:, 0], X2[:, 1], c='#e74c3c', label='Class B (outer)', s=80, edgecolors='white')
    
    # Draw circular boundary
    circle = plt.Circle((0, 0), 2, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title('RBF Kernel\n(Non-linearly Separable Data)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Maximum margin concept
    ax = axes[1, 0]
    X1 = np.random.randn(n, 2) * 0.5 + np.array([1.5, 1.5])
    X2 = np.random.randn(n, 2) * 0.5 + np.array([-1.5, -1.5])
    ax.scatter(X1[:, 0], X1[:, 1], c='#3498db', s=80, edgecolors='white')
    ax.scatter(X2[:, 0], X2[:, 1], c='#e74c3c', s=80, edgecolors='white')
    
    # Hyperplane and margins
    x_line = np.linspace(-4, 4, 100)
    ax.plot(x_line, -x_line, 'k-', linewidth=3, label='Hyperplane')
    ax.plot(x_line, -x_line + 1.5, 'k--', linewidth=1.5, alpha=0.7)
    ax.plot(x_line, -x_line - 1.5, 'k--', linewidth=1.5, alpha=0.7)
    ax.fill_between(x_line, -x_line - 1.5, -x_line + 1.5, alpha=0.2, color='yellow', label='Margin')
    
    # Mark support vectors
    sv1 = X1[np.argmin(X1[:, 0] + X1[:, 1])]
    sv2 = X2[np.argmax(X2[:, 0] + X2[:, 1])]
    ax.scatter([sv1[0], sv2[0]], [sv1[1], sv2[1]], s=200, facecolors='none', 
              edgecolors='black', linewidths=3, label='Support Vectors')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title('Maximum Margin Principle\n(Support Vectors Define the Boundary)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    
    # 4. Kernel trick illustration
    ax = axes[1, 1]
    ax.text(0.5, 0.85, 'THE KERNEL TRICK', fontsize=16, fontweight='bold', 
           ha='center', va='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.65, 'φ: Low-dim → High-dim space', fontsize=14, 
           ha='center', va='center', transform=ax.transAxes, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.text(0.5, 0.45, 'K(x, y) = φ(x) · φ(y)', fontsize=16, style='italic',
           ha='center', va='center', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.text(0.5, 0.25, 'Linear:  K(x,y) = x·y\n\n'
                       'RBF:     K(x,y) = exp(-γ||x-y||²)\n\n'
                       'Poly:    K(x,y) = (γx·y + r)^d', 
           fontsize=12, ha='center', va='center', transform=ax.transAxes,
           family='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Kernel Functions\n(Mathematical Formulas)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Understanding SVM Kernels & Hyperplane Segregation', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Kernel Comparison Module Test")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=20, n_classes=5,
                               n_informative=10, random_state=42)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compare kernels
    results = compare_kernels(X_train, y_train, X_test, y_test)
    
    # Generate theory plot
    plot_kernel_theory(save_path='outputs/figures/kernel_theory.png')
    
    print("\n✓ All tests passed!")
