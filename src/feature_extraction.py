"""
Feature Extraction Module using HOG (Histogram of Oriented Gradients)
=====================================================================
Bachelor's Thesis: SVM for Object Recognition

HOG Features capture edge orientations and local shape information,
making them excellent for object recognition tasks with SVM classifiers.
"""

import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os


class HOGFeatureExtractor:
    """
    Combined Feature Extractor (HOG + Color Histograms) for image classification.
    
    Combines:
    1. HOG: Captures texture and shape (edges)
    2. Color Histograms: Captures spectral information (land cover colors)
    
    This aligns with standard methodology for satellite image analysis.
    
    Parameters:
    -----------
    orientations : int
        Number of HOG orientation bins (default: 9)
    pixels_per_cell : tuple
        Size of each HOG cell in pixels (default: (4, 4))
    cells_per_block : tuple
        Number of HOG cells in each block (default: (2, 2))
    block_norm : str
        HOG block normalization method (default: 'L2-Hys')
    use_color_hist : bool
        Whether to include color histogram features (default: True)
    color_bins : int
        Number of bins per color channel (default: 32)
    resize_shape : tuple, optional
        Resize images to this shape before feature extraction
    """
    
    def __init__(self, 
                 orientations=9,
                 pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2),
                 block_norm='L2-Hys',
                 use_color_hist=True,
                 color_bins=32,
                 resize_shape=None):
        
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.use_color_hist = use_color_hist
        self.color_bins = color_bins
        self.resize_shape = resize_shape
        
        # Scalers and transformers (fitted during transform)
        self.scaler = StandardScaler()
        self.pca = None
        self._is_fitted = False
        
    def _extract_color_histogram(self, image):
        """
        Extract color histogram features.
        
        Parameters:
        -----------
        image : np.ndarray
            RGB image (H, W, 3)
            
        Returns:
        --------
        np.ndarray : Color coordinates/histogram vector
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return np.array([])
            
        # Calculate histogram for each channel
        hist_features = []
        for channel in range(3):
            hist, _ = np.histogram(image[:, :, channel], bins=self.color_bins, range=(0, 1))
            # Normalize
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            hist_features.extend(hist)
            
        return np.array(hist_features)

    def _extract_single(self, image):
        """
        Extract combined features from a single image.
        
        Parameters:
        -----------
        image : np.ndarray
            Image (H, W, 3) or (H, W)
            
        Returns:
        --------
        np.ndarray : Combined feature vector
        """
        # Resize if specified
        if self.resize_shape is not None:
            image = resize(image, self.resize_shape, anti_aliasing=True)
            
        # Prepare for HOG (needs greyscale)
        if image.ndim == 3:
            image_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            image_rgb = image # Keep original for color features
        else:
            image_gray = image
            image_rgb = None # No color features if input is grayscale
            
        # 1. Extract HOG Features
        hog_features = hog(
            image_gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True
        )
        
        # 2. Extract Color features (if enabled and image is RGB)
        if self.use_color_hist and image_rgb is not None:
            color_features = self._extract_color_histogram(image_rgb)
            combined_features = np.hstack([hog_features, color_features])
        else:
            combined_features = hog_features
            
        return combined_features
    
    def extract_features(self, images, show_progress=True):
        """
        Extract features from multiple images.
        """
        features_list = []
        iterator = tqdm(images, desc="Extracting features (HOG+Color)") if show_progress else images
        
        for image in iterator:
            features = self._extract_single(image)
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit_transform(self, images, apply_pca=False, n_components=100, show_progress=True):
        """
        Extract features and fit the scaler (and optionally PCA).
        """
        # Extract features
        features = self.extract_features(images, show_progress)
        
        print(f"\n✓ Extracted {features.shape[1]} features per image (HOG + Color)")
        
        # Fit and transform with scaler
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA if requested
        if apply_pca:
            # Limit components to min(samples, features, requested)
            n_components = min(n_components, features.shape[0], features.shape[1])
            print(f"Applying PCA (reducing to {n_components} components)...")
            self.pca = PCA(n_components=n_components, random_state=42)
            features_scaled = self.pca.fit_transform(features_scaled)
            
            variance_retained = np.sum(self.pca.explained_variance_ratio_) * 100
            print(f"✓ PCA complete. Variance retained: {variance_retained:.2f}%")
        
        self._is_fitted = True
        return features_scaled
    
    def transform(self, images, show_progress=True):
        """Transform images using the fitted scaler."""
        if not self._is_fitted:
            raise ValueError("Extractor must be fitted first.")
        
        features = self.extract_features(images, show_progress)
        features_scaled = self.scaler.transform(features)
        
        if self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)
        
        return features_scaled
    
    def save(self, filepath):
        """Save the fitted extractor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'block_norm': self.block_norm,
            'resize_shape': self.resize_shape,
            'use_color_hist':  getattr(self, 'use_color_hist', True),
            'color_bins': getattr(self, 'color_bins', 32),
            'scaler': self.scaler,
            'pca': self.pca,
            '_is_fitted': self._is_fitted
        }, filepath)
        print(f"✓ Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a fitted extractor from disk."""
        data = joblib.load(filepath)
        
        extractor = cls(
            orientations=data['orientations'],
            pixels_per_cell=data['pixels_per_cell'],
            cells_per_block=data['cells_per_block'],
            block_norm=data['block_norm'],
            resize_shape=data['resize_shape'],
            use_color_hist=data.get('use_color_hist', True),
            color_bins=data.get('color_bins', 32)
        )
        extractor.scaler = data['scaler']
        extractor.pca = data['pca']
        extractor._is_fitted = data['_is_fitted']
        
        print(f"✓ Feature extractor loaded from {filepath}")
        return extractor
    
    def get_feature_info(self):
        """Get information about the extracted features."""
        info = {
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'use_color_hist': self.use_color_hist,
            'color_bins': self.color_bins,
            'is_fitted': self._is_fitted,
            'uses_pca': self.pca is not None
        }
        
        if self.pca is not None:
            info['pca_components'] = self.pca.n_components_
            info['variance_retained'] = np.sum(self.pca.explained_variance_ratio_)
        
        return info


def visualize_hog_features(image, extractor=None):
    """
    Visualize HOG features for a single image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (grayscale)
    extractor : HOGFeatureExtractor, optional
        Extractor with parameters to use
        
    Returns:
    --------
    tuple : (HOG features, HOG image visualization)
    """
    if extractor is None:
        extractor = HOGFeatureExtractor()
    
    # Ensure grayscale
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    
    features, hog_image = hog(
        image,
        orientations=extractor.orientations,
        pixels_per_cell=extractor.pixels_per_cell,
        cells_per_block=extractor.cells_per_block,
        block_norm=extractor.block_norm,
        visualize=True,
        feature_vector=True
    )
    
    return features, hog_image


if __name__ == "__main__":
    # Test feature extraction
    print("=" * 60)
    print("HOG Feature Extraction Test")
    print("=" * 60)
    
    # Create dummy images for testing
    np.random.seed(42)
    dummy_images = np.random.rand(100, 32, 32)  # 100 grayscale 32x32 images
    
    # Create extractor and extract features
    extractor = HOGFeatureExtractor()
    features = extractor.fit_transform(dummy_images, apply_pca=False)
    
    print(f"\nInput shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"\nFeature info: {extractor.get_feature_info()}")
