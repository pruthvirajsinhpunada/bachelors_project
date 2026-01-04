"""
SVM Classifier Module for Object Recognition
=============================================
Bachelor's Thesis: SVM for Object Recognition

This module implements the Support Vector Machine classifier with:
- Multiple kernel options (RBF, Linear, Polynomial)
- Hyperparameter tuning via GridSearchCV
- Comprehensive evaluation metrics
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
import os
import time


class SVMClassifier:
    """
    Support Vector Machine Classifier for multi-class object recognition.
    
    Parameters:
    -----------
    kernel : str
        Kernel type: 'rbf', 'linear', 'poly' (default: 'rbf')
    C : float
        Regularization parameter (default: 1.0)
    gamma : str or float
        Kernel coefficient for 'rbf' and 'poly' (default: 'scale')
    degree : int
        Degree for polynomial kernel (default: 3)
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, 
                 kernel='rbf',
                 C=1.0,
                 gamma='scale',
                 degree=3,
                 random_state=42):
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            random_state=random_state,
            probability=True,  # Enable probability estimates for ROC curves
            decision_function_shape='ovr'  # One-vs-Rest for multi-class
        )
        
        self._is_fitted = False
        self.training_time = None
        self.best_params = None
        
    def train(self, X_train, y_train):
        """
        Train the SVM classifier.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
            
        Returns:
        --------
        self
        """
        print("\n" + "=" * 60)
        print("Training SVM Classifier")
        print("=" * 60)
        print(f"Kernel: {self.kernel}")
        print(f"C: {self.C}")
        print(f"Gamma: {self.gamma}")
        print(f"Training samples: {len(X_train)}")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        self._is_fitted = True
        
        print(f"\n✓ Training complete!")
        print(f"  Training time: {self.training_time:.2f} seconds")
        print(f"  Support vectors: {self.model.n_support_.sum()}")
        
        return self
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=3, n_jobs=-1):
        """
        Tune hyperparameters using GridSearchCV.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_grid : dict, optional
            Parameter grid for search
        cv : int
            Number of cross-validation folds
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
            
        Returns:
        --------
        dict : Best parameters found
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        
        print("\n" + "=" * 60)
        print("Hyperparameter Tuning with GridSearchCV")
        print("=" * 60)
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: {cv}")
        
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self._is_fitted = True
        
        print(f"\n✓ Grid search complete!")
        print(f"  Search time: {search_time:.2f} seconds")
        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict
            
        Returns:
        --------
        np.ndarray : Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained first. Call train() or tune_hyperparameters().")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict
            
        Returns:
        --------
        np.ndarray : Class probabilities (N, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            True labels
        class_names : list, optional
            Names of classes for the report
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        # ROC curves (one-vs-rest for multi-class)
        n_classes = len(np.unique(y_test))
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        roc_data = {}
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[i] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_data': roc_data,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print("\n" + "=" * 60)
        print("Model Evaluation Results")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv : int
            Number of folds
            
        Returns:
        --------
        dict : Cross-validation results
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        
        print(f"✓ CV Accuracy: {results['mean']:.4f} (+/- {results['std']*2:.4f})")
        
        return results
    
    def save(self, filepath):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'model': self.model,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'degree': self.degree,
            'random_state': self.random_state,
            '_is_fitted': self._is_fitted,
            'training_time': self.training_time,
            'best_params': self.best_params
        }
        
        joblib.dump(data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a trained model from disk."""
        data = joblib.load(filepath)
        
        classifier = cls(
            kernel=data['kernel'],
            C=data['C'],
            gamma=data['gamma'],
            degree=data['degree'],
            random_state=data['random_state']
        )
        
        classifier.model = data['model']
        classifier._is_fitted = data['_is_fitted']
        classifier.training_time = data['training_time']
        classifier.best_params = data['best_params']
        
        print(f"✓ Model loaded from {filepath}")
        return classifier
    
    def get_model_info(self):
        """Get information about the model."""
        info = {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'is_fitted': self._is_fitted,
            'training_time': self.training_time,
            'best_params': self.best_params
        }
        
        if self._is_fitted:
            info['n_support_vectors'] = self.model.n_support_.sum()
            info['n_classes'] = len(self.model.classes_)
        
        return info


if __name__ == "__main__":
    # Test the classifier
    print("=" * 60)
    print("SVM Classifier Test")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.rand(500, 100)
    y_train = np.random.randint(0, 5, 500)
    X_test = np.random.rand(100, 100)
    y_test = np.random.randint(0, 5, 100)
    
    # Train and evaluate
    clf = SVMClassifier(kernel='rbf', C=1.0)
    clf.train(X_train, y_train)
    results = clf.evaluate(X_test, y_test, class_names=['A', 'B', 'C', 'D', 'E'])
    
    print(f"\nModel info: {clf.get_model_info()}")
