"""
Satellite Image Loader - EuroSAT Dataset
=========================================
Bachelor's Thesis: SVM for Object Recognition

EuroSAT is a dataset of satellite images covering 10 land use/land cover classes.
This demonstrates real-world application of SVM for satellite image classification.

Classes:
- AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
- Pasture, PermanentCrop, Residential, River, SeaLake
"""

import os
import numpy as np
from tqdm import tqdm
import urllib.request
import zipfile
from PIL import Image


# EuroSAT Class Names (Land Cover Types)
SATELLITE_CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]


def download_eurosat(data_dir='data'):
    """
    Download EuroSAT satellite image dataset.
    
    Note: EuroSAT is ~90MB. For demo purposes, we'll create a synthetic
    satellite-like dataset if download fails or for faster testing.
    
    Parameters:
    -----------
    data_dir : str
        Directory to save the dataset
        
    Returns:
    --------
    str : Path to the dataset
    """
    eurosat_dir = os.path.join(data_dir, 'EuroSAT')
    
    # Check if already exists
    if os.path.exists(eurosat_dir) and len(os.listdir(eurosat_dir)) > 0:
        print("✓ EuroSAT dataset already exists.")
        return eurosat_dir
    
    os.makedirs(data_dir, exist_ok=True)
    
    # EuroSAT URL (RGB version)
    url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
    zip_path = os.path.join(data_dir, "EuroSAT_RGB.zip")
    
    try:
        print("Downloading EuroSAT dataset (this may take a few minutes)...")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(url, zip_path, show_progress)
        print("\n✓ Download complete!")
        
        # Extract
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up
        os.remove(zip_path)
        print("✓ Extraction complete!")
        
        return eurosat_dir
        
    except Exception as e:
        print(f"\n⚠ Could not download EuroSAT: {e}")
        print("Creating synthetic satellite dataset for demonstration...")
        return create_synthetic_satellite_data(data_dir)


def create_synthetic_satellite_data(data_dir='data', n_samples_per_class=100):
    """
    Create synthetic satellite-like images for demonstration.
    These simulate different land cover types with characteristic colors.
    
    Parameters:
    -----------
    data_dir : str
        Directory to save synthetic data
    n_samples_per_class : int
        Number of samples per class
        
    Returns:
    --------
    str : Path to synthetic dataset
    """
    synthetic_dir = os.path.join(data_dir, 'SatelliteSynthetic')
    os.makedirs(synthetic_dir, exist_ok=True)
    
    # Define color characteristics for each land cover type
    # (RGB means and variations to simulate satellite imagery)
    land_cover_colors = {
        'AnnualCrop': {'mean': [139, 195, 74], 'std': 30},      # Light green
        'Forest': {'mean': [27, 94, 32], 'std': 20},            # Dark green
        'HerbaceousVegetation': {'mean': [76, 175, 80], 'std': 25},  # Medium green
        'Highway': {'mean': [158, 158, 158], 'std': 15},        # Gray
        'Industrial': {'mean': [120, 120, 150], 'std': 20},     # Gray-blue
        'Pasture': {'mean': [129, 199, 132], 'std': 25},        # Light green
        'PermanentCrop': {'mean': [85, 139, 47], 'std': 25},    # Olive green
        'Residential': {'mean': [161, 136, 127], 'std': 30},    # Brown-gray
        'River': {'mean': [25, 118, 210], 'std': 20},           # Blue
        'SeaLake': {'mean': [30, 100, 180], 'std': 15}          # Deep blue
    }
    
    print(f"Creating synthetic satellite images...")
    
    for class_idx, class_name in enumerate(tqdm(SATELLITE_CLASS_NAMES, desc="Generating classes")):
        class_dir = os.path.join(synthetic_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        colors = land_cover_colors[class_name]
        
        for i in range(n_samples_per_class):
            # Create 64x64 image with texture
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Base color with noise
            for c in range(3):
                channel = np.random.normal(colors['mean'][c], colors['std'], (64, 64))
                channel = np.clip(channel, 0, 255).astype(np.uint8)
                img[:, :, c] = channel
            
            # Add some texture patterns
            if class_name in ['Forest', 'HerbaceousVegetation', 'Pasture']:
                # Add speckle for vegetation
                noise = np.random.randint(-20, 20, (64, 64))
                img[:, :, 1] = np.clip(img[:, :, 1].astype(int) + noise, 0, 255).astype(np.uint8)
            
            elif class_name in ['Highway', 'Residential', 'Industrial']:
                # Add linear patterns
                for y in range(0, 64, np.random.randint(8, 16)):
                    thickness = np.random.randint(1, 3)
                    img[y:y+thickness, :, :] = np.clip(img[y:y+thickness, :, :].astype(int) - 30, 0, 255).astype(np.uint8)
            
            elif class_name in ['River', 'SeaLake']:
                # Add wave-like patterns
                for x in range(64):
                    shift = int(3 * np.sin(x / 5))
                    img[:, x, :] = np.roll(img[:, x, :], shift, axis=0)
            
            # Save image
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(class_dir, f'{class_name}_{i:04d}.png'))
    
    print(f"✓ Created {n_samples_per_class * len(SATELLITE_CLASS_NAMES)} synthetic satellite images")
    return synthetic_dir


def load_satellite_dataset(data_dir='data', subset_size=None, image_size=64):
    """
    Load satellite images from EuroSAT or synthetic dataset.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the dataset
    subset_size : int, optional
        Number of samples to load (None for all)
    image_size : int
        Size to resize images to (default 64x64)
        
    Returns:
    --------
    dict : Dataset with images, labels, and class names
    """
    # Try to get dataset directory
    eurosat_dir = os.path.join(data_dir, 'EuroSAT')
    eurosat_rgb_dir = os.path.join(data_dir, 'EuroSAT_RGB')  # Alternative extraction name
    synthetic_dir = os.path.join(data_dir, 'SatelliteSynthetic')
    
    if os.path.exists(eurosat_rgb_dir) and len(os.listdir(eurosat_rgb_dir)) > 0:
        dataset_dir = eurosat_rgb_dir
        print("Loading EuroSAT RGB satellite dataset...")
    elif os.path.exists(eurosat_dir) and len(os.listdir(eurosat_dir)) > 0:
        dataset_dir = eurosat_dir
        print("Loading EuroSAT satellite dataset...")
    elif os.path.exists(synthetic_dir) and len(os.listdir(synthetic_dir)) > 0:
        dataset_dir = synthetic_dir
        print("Loading synthetic satellite dataset...")
    else:
        # Download/create dataset
        dataset_dir = download_eurosat(data_dir)
    
    images = []
    labels = []
    
    # Check which classes exist
    available_classes = [c for c in SATELLITE_CLASS_NAMES if os.path.exists(os.path.join(dataset_dir, c))]
    
    if not available_classes:
        # Dataset might be in a subdirectory
        for subdir in os.listdir(dataset_dir):
            subdir_path = os.path.join(dataset_dir, subdir)
            if os.path.isdir(subdir_path):
                available_classes = [c for c in SATELLITE_CLASS_NAMES if os.path.exists(os.path.join(subdir_path, c))]
                if available_classes:
                    dataset_dir = subdir_path
                    break
    
    if not available_classes:
        print("⚠ No satellite classes found. Creating synthetic data...")
        dataset_dir = create_synthetic_satellite_data(data_dir)
        available_classes = SATELLITE_CLASS_NAMES
    
    print(f"Loading from: {dataset_dir}")
    print(f"Found {len(available_classes)} classes")
    
    for class_idx, class_name in enumerate(tqdm(available_classes, desc="Loading images")):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        # Get image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        # Apply subset if specified
        if subset_size:
            max_per_class = subset_size // len(available_classes)
            image_files = image_files[:max_per_class]
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size))
                images.append(np.array(img))
                labels.append(class_idx)
            except Exception as e:
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    # Split train/test (80/20)
    split_idx = int(0.8 * len(images))
    
    print(f"\n✓ Satellite dataset loaded!")
    print(f"  Total images: {len(images)}")
    print(f"  Training: {split_idx}")
    print(f"  Testing: {len(images) - split_idx}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Classes: {len(available_classes)}")
    
    return {
        'X_train': images[:split_idx],
        'y_train': labels[:split_idx],
        'X_test': images[split_idx:],
        'y_test': labels[split_idx:],
        'class_names': available_classes
    }


def plot_satellite_samples(images, labels, class_names, n_samples=20, save_path=None):
    """
    Plot sample satellite images.
    
    Parameters:
    -----------
    images : np.ndarray
        Image array
    labels : np.ndarray
        Label array
    class_names : list
        Class names
    n_samples : int
        Number of samples to show
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    fig.suptitle('🛰️ Satellite Image Samples (Land Cover Classification)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            img_idx = indices[idx]
            ax.imshow(images[img_idx])
            ax.set_title(class_names[labels[img_idx]], fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Satellite Image Loader Test")
    print("=" * 60)
    
    # Load dataset
    data = load_satellite_dataset(subset_size=1000)
    
    print(f"\nTrain shape: {data['X_train'].shape}")
    print(f"Test shape: {data['X_test'].shape}")
    print(f"Classes: {data['class_names']}")
    
    # Plot samples
    plot_satellite_samples(
        data['X_train'], data['y_train'], data['class_names'],
        save_path='outputs/figures/satellite_samples.png'
    )
