"""
Data Loader Module for CIFAR-10 Dataset
========================================
Bachelor's Thesis: SVM for Object Recognition
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import urllib.request
import tarfile


# CIFAR-10 Class Names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def download_cifar10(data_dir='data'):
    """
    Download CIFAR-10 dataset if not already present.
    
    Parameters:
    -----------
    data_dir : str
        Directory to save the dataset
        
    Returns:
    --------
    str : Path to the extracted dataset
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(extract_dir):
        print("✓ CIFAR-10 dataset already exists.")
        return extract_dir
    
    print(f"Downloading CIFAR-10 dataset...")
    
    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        print(f"\rProgress: {percent}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, filename, show_progress)
    print("\n✓ Download complete!")
    
    # Extract
    print("Extracting dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(data_dir)
    
    # Clean up tar file
    os.remove(filename)
    print("✓ Extraction complete!")
    
    return extract_dir


def load_cifar10_batch(file_path):
    """
    Load a single CIFAR-10 batch file.
    
    Parameters:
    -----------
    file_path : str
        Path to the batch file
        
    Returns:
    --------
    tuple : (images, labels)
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    # Images are stored as 3072 values (32x32x3 flattened)
    images = batch[b'data']
    labels = np.array(batch[b'labels'])
    
    # Reshape to (N, 32, 32, 3) - CIFAR stores as (N, 3, 32, 32)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images, labels


def load_cifar10(data_dir='data', subset_size=None):
    """
    Load the complete CIFAR-10 dataset.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the dataset
    subset_size : int, optional
        If specified, use only a subset of data (useful for faster training)
        
    Returns:
    --------
    dict : Dictionary containing train/test images and labels
    """
    # Download if necessary
    extract_dir = download_cifar10(data_dir)
    
    print("\nLoading CIFAR-10 dataset...")
    
    # Load training batches
    train_images = []
    train_labels = []
    
    for i in tqdm(range(1, 6), desc="Loading training batches"):
        batch_file = os.path.join(extract_dir, f'data_batch_{i}')
        images, labels = load_cifar10_batch(batch_file)
        train_images.append(images)
        train_labels.append(labels)
    
    X_train = np.concatenate(train_images)
    y_train = np.concatenate(train_labels)
    
    # Load test batch
    test_file = os.path.join(extract_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_file)
    
    # Apply subset if specified
    if subset_size is not None:
        print(f"\nUsing subset of {subset_size} samples for faster training...")
        
        # Stratified sampling for training
        indices = np.random.choice(len(X_train), min(subset_size, len(X_train)), replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Proportional test set
        test_subset = subset_size // 5
        indices = np.random.choice(len(X_test), min(test_subset, len(X_test)), replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Training set: {X_train.shape[0]} images")
    print(f"  Test set: {X_test.shape[0]} images")
    print(f"  Image shape: {X_train.shape[1:]}")
    print(f"  Number of classes: {len(CLASS_NAMES)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': CLASS_NAMES
    }


def preprocess_images(images, grayscale=True, normalize=True):
    """
    Preprocess images for feature extraction.
    
    Parameters:
    -----------
    images : np.ndarray
        Array of images (N, H, W, C)
    grayscale : bool
        Convert to grayscale if True
    normalize : bool
        Normalize pixel values to [0, 1]
        
    Returns:
    --------
    np.ndarray : Preprocessed images
    """
    processed = images.copy().astype(np.float32)
    
    if normalize:
        processed = processed / 255.0
    
    if grayscale:
        # Convert to grayscale using luminosity method
        # Y = 0.299*R + 0.587*G + 0.114*B
        processed = np.dot(processed[..., :3], [0.299, 0.587, 0.114])
    
    return processed


def create_data_splits(X, y, val_size=0.2, random_state=42):
    """
    Create train/validation splits from the data.
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    val_size : float
        Proportion of data for validation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_val, y_train, y_val)
    """
    return train_test_split(
        X, y, 
        test_size=val_size, 
        random_state=random_state,
        stratify=y
    )


def get_class_distribution(y):
    """
    Get the distribution of classes in the dataset.
    
    Parameters:
    -----------
    y : np.ndarray
        Label array
        
    Returns:
    --------
    dict : Class distribution {class_name: count}
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = {}
    
    for cls_idx, count in zip(unique, counts):
        distribution[CLASS_NAMES[cls_idx]] = count
    
    return distribution


if __name__ == "__main__":
    # Test the data loader
    print("=" * 60)
    print("CIFAR-10 Data Loader Test")
    print("=" * 60)
    
    data = load_cifar10(subset_size=5000)
    
    print("\n" + "=" * 60)
    print("Class Distribution (Training Set):")
    print("=" * 60)
    
    distribution = get_class_distribution(data['y_train'])
    for cls_name, count in distribution.items():
        print(f"  {cls_name:12s}: {count:5d} images")
