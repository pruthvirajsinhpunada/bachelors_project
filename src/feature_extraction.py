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
    HOG Feature Extractor for image classification.
    
    Histogram of Oriented Gradients (HOG) is a feature descriptor used
    for object detection. It captures edge directions and their distributions.
    
    Parameters:
    -----------
    orientations : int
        Number of orientation bins (default: 9)
    pixels_per_cell : tuple
        Size of each cell in pixels (default: (4, 4))
    cells_per_block : tuple
        Number of cells in each block (default: (2, 2))
    block_norm : str
        Block normalization method (default: 'L2-Hys')
    resize_shape : tuple, optional
        Resize images to this shape before feature extraction
    """
    
    def __init__(self, 
                 orientations=9,
                 pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2),
                 block_norm='L2-Hys',
                 resize_shape=None):
        
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.resize_shape = resize_shape
        
        # Scalers and transformers (fitted during transform)
        self.scaler = StandardScaler()
        self.pca = None
        self._is_fitted = False
        
    def _extract_single_hog(self, image):
        """
        Extract HOG features from a single image.
        
        Parameters:
        -----------
        image : np.ndarray
            Grayscale image (H, W)
            
        Returns:
        --------
        np.ndarray : HOG feature vector
        """
        # Resize if specified
        if self.resize_shape is not None:
            image = resize(image, self.resize_shape, anti_aliasing=True)
        
        # Ensure image is 2D (grayscale)
        if image.ndim == 3:
            # Convert to grayscale
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        
        # Extract HOG features
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True
        )
        
        return features
    
    def extract_features(self, images, show_progress=True):
        """
        Extract HOG features from multiple images.
        
        Parameters:
        -----------
        images : np.ndarray
            Array of images (N, H, W) or (N, H, W, C)
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        np.ndarray : Feature matrix (N, n_features)
        """
        features_list = []
        
        iterator = tqdm(images, desc="Extracting HOG features") if show_progress else images
        
        for image in iterator:
            features = self._extract_single_hog(image)
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit_transform(self, images, apply_pca=False, n_components=100, show_progress=True):
        """
        Extract features and fit the scaler (and optionally PCA).
        
        Parameters:
        -----------
        images : np.ndarray
            Training images
        apply_pca : bool
            Apply PCA dimensionality reduction
        n_components : int
            Number of PCA components
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        np.ndarray : Scaled (and optionally reduced) features
        """
        # Extract HOG features
        features = self.extract_features(images, show_progress)
        
        print(f"\n✓ Extracted {features.shape[1]} HOG features per image")
        
        # Fit and transform with scaler
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA if requested
        if apply_pca:
            print(f"Applying PCA (reducing to {n_components} components)...")
            self.pca = PCA(n_components=n_components, random_state=42)
            features_scaled = self.pca.fit_transform(features_scaled)
            
            variance_retained = np.sum(self.pca.explained_variance_ratio_) * 100
            print(f"✓ PCA complete. Variance retained: {variance_retained:.2f}%")
        
        self._is_fitted = True
        return features_scaled
    
    def transform(self, images, show_progress=True):
        """
        Transform images using the fitted scaler (and PCA if applicable).
        
        Parameters:
        -----------
        images : np.ndarray
            Images to transform
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        np.ndarray : Transformed features
        """
        if not self._is_fitted:
            raise ValueError("Extractor must be fitted first. Call fit_transform() on training data.")
        
        # Extract HOG features
        features = self.extract_features(images, show_progress)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply PCA if fitted
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
            resize_shape=data['resize_shape']
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
            'block_norm': self.block_norm,
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
