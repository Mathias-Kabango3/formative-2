"""
Training Pipeline for HMM Activity Recognition.

This module handles:
- Data preparation and splitting
- Feature scaling
- Model training
- Model persistence
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import pickle
import os
import json
from datetime import datetime

from config import (
    ACTIVITIES, ACTIVITY_TO_ID, ID_TO_ACTIVITY,
    TEST_SIZE, RANDOM_STATE, MODELS_DIR, RESULTS_DIR, N_STATES
)
from data_processing import prepare_dataset, segment_into_windows, generate_synthetic_data
from feature_extraction import extract_features_from_windows, get_feature_names
from hmm_model import GaussianHMM, HMMLearnWrapper, create_hmm_model


class ActivityRecognitionPipeline:
    """
    Complete pipeline for activity recognition using HMM.
    """
    
    def __init__(
        self,
        use_hmmlearn: bool = False,
        n_iter: int = 100,
        random_state: int = RANDOM_STATE
    ):
        """
        Initialize the pipeline.
        
        Args:
            use_hmmlearn: Whether to use hmmlearn library
            n_iter: Number of EM iterations
            random_state: Random seed
        """
        self.use_hmmlearn = use_hmmlearn
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = get_feature_names()
        
        self.is_trained = False
        self.training_info = {}
    
    def prepare_data(
        self,
        use_synthetic: bool = True,
        n_synthetic_samples: int = 1000,
        test_size: float = TEST_SIZE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and split data for training.
        
        Args:
            use_synthetic: Whether to use synthetic data
            n_synthetic_samples: Number of synthetic samples per activity
            test_size: Fraction of data for testing
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("=" * 50)
        print("Preparing Data")
        print("=" * 50)
        
        # Load and segment data
        windows, labels = prepare_dataset(
            use_synthetic=use_synthetic,
            n_synthetic_samples=n_synthetic_samples
        )
        
        print(f"\nTotal windows: {len(windows)}")
        
        # Extract features
        print("\nExtracting features...")
        X = extract_features_from_windows(windows)
        y = np.array(labels)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Class distribution
        print("\nClass distribution (training):")
        for activity, idx in ACTIVITY_TO_ID.items():
            count = np.sum(y_train == idx)
            print(f"  {activity}: {count} ({100*count/len(y_train):.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Train the HMM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Print progress
        
        Returns:
            Training info dictionary
        """
        print("\n" + "=" * 50)
        print("Training HMM Model")
        print("=" * 50)
        
        # Scale features
        print("\nScaling features...")
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create HMM model
        self.model = create_hmm_model(
            use_hmmlearn=self.use_hmmlearn,
            n_states=N_STATES,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        print(f"\nTraining HMM with {N_STATES} states...")
        print(f"Using {'hmmlearn' if self.use_hmmlearn else 'custom'} implementation")
        
        # Organize data by sequences
        # For HMM training, we need sequences of observations
        # Here we train on the entire feature matrix
        start_time = datetime.now()
        self.model.fit(X_scaled, verbose=verbose)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Store training info
        self.training_info = {
            'n_training_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_states': N_STATES,
            'training_time': training_time,
            'use_hmmlearn': self.use_hmmlearn,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get transition matrix
        A = self.model.get_transition_matrix()
        if A is not None:
            print("\nLearned Transition Matrix:")
            print("   ", "  ".join([ID_TO_ACTIVITY[i][:4] for i in range(N_STATES)]))
            for i in range(N_STATES):
                row = " ".join([f"{A[i,j]:.2f}" for j in range(N_STATES)])
                print(f"{ID_TO_ACTIVITY[i][:4]}: {row}")
        
        self.is_trained = True
        return self.training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict activity labels using Viterbi decoding.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability matrix (n_samples, n_states)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, model_name: str = 'hmm_activity_model'):
        """
        Save trained model and scaler.
        
        Args:
            model_name: Base name for saved files
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, f'{model_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")
        
        # Save training info
        info_path = os.path.join(MODELS_DIR, f'{model_name}_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.training_info, f, indent=2)
        print(f"Training info saved to {info_path}")
    
    def load_model(self, model_name: str = 'hmm_activity_model'):
        """
        Load trained model and scaler.
        
        Args:
            model_name: Base name for saved files
        """
        # Load model
        model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
        self.model = create_hmm_model(use_hmmlearn=self.use_hmmlearn)
        self.model.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, f'{model_name}_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {scaler_path}")
        
        # Load training info
        info_path = os.path.join(MODELS_DIR, f'{model_name}_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
        
        self.is_trained = True


def train_per_activity_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 50
) -> Dict[str, object]:
    """
    Train separate HMM models for each activity.
    This can improve recognition for specific activities.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_iter: Number of EM iterations
    
    Returns:
        Dictionary mapping activity names to trained models
    """
    models = {}
    
    for activity, activity_id in ACTIVITY_TO_ID.items():
        print(f"\nTraining model for '{activity}'...")
        
        # Get data for this activity
        mask = y_train == activity_id
        X_activity = X_train[mask]
        
        if len(X_activity) < 10:
            print(f"  Skipping - not enough samples ({len(X_activity)})")
            continue
        
        # Train a simple HMM (could use more states for complex activities)
        model = GaussianHMM(n_states=1, n_iter=n_iter)
        model.fit(X_activity, verbose=False)
        models[activity] = model
        
        print(f"  Trained with {len(X_activity)} samples")
    
    return models


def cross_validate_hmm(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_iter: int = 50
) -> Dict[str, float]:
    """
    Perform cross-validation for HMM model.
    
    Args:
        X: Feature matrix
        y: Labels
        n_folds: Number of cross-validation folds
        n_iter: EM iterations per fold
    
    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import StratifiedKFold
    
    print(f"\nPerforming {n_folds}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train
        model = GaussianHMM(n_states=N_STATES, n_iter=n_iter)
        model.fit(X_train_scaled, verbose=False)
        
        # Predict
        y_pred = model.predict(X_val_scaled)
        
        # Accuracy (note: HMM states may not align with original labels)
        # This is a simplified accuracy - proper evaluation needs state mapping
        from sklearn.metrics import accuracy_score
        
        # Simple accuracy (after finding best state mapping)
        accuracy = compute_accuracy_with_mapping(y_val, y_pred)
        accuracies.append(accuracy)
        
        print(f"  Fold {fold + 1}: Accuracy = {accuracy:.2%}")
    
    results = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'fold_accuracies': accuracies
    }
    
    print(f"\nCV Results: {results['mean_accuracy']:.2%} (+/- {results['std_accuracy']:.2%})")
    
    return results


def compute_accuracy_with_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy by finding optimal mapping between predicted and true states.
    
    HMM states may not correspond directly to our activity labels,
    so we find the mapping that maximizes accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted states
    
    Returns:
        Accuracy with best mapping
    """
    from scipy.optimize import linear_sum_assignment
    
    n_classes = len(ACTIVITIES)
    
    # Build confusion matrix
    confusion = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        if t < n_classes and p < n_classes:
            confusion[t, p] += 1
    
    # Hungarian algorithm to find best mapping
    row_ind, col_ind = linear_sum_assignment(-confusion)
    
    # Create mapping
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    # Apply mapping and compute accuracy
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
    accuracy = np.mean(y_pred_mapped == y_true)
    
    return accuracy


if __name__ == "__main__":
    # Test training pipeline
    print("=" * 60)
    print("Testing Training Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ActivityRecognitionPipeline(
        use_hmmlearn=False,  # Use custom implementation
        n_iter=50
    )
    
    # Prepare data (using synthetic for testing)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        use_synthetic=True,
        n_synthetic_samples=500
    )
    
    # Train model
    training_info = pipeline.train(X_train, y_train)
    
    # Make predictions
    print("\n" + "=" * 50)
    print("Making Predictions")
    print("=" * 50)
    
    y_pred = pipeline.predict(X_test)
    
    # Compute accuracy with mapping
    accuracy = compute_accuracy_with_mapping(y_test, y_pred)
    print(f"\nTest Accuracy (with state mapping): {accuracy:.2%}")
    
    # Save model
    pipeline.save_model('test_hmm_model')
    
    # Cross-validation
    print("\n" + "=" * 50)
    print("Cross-Validation")
    print("=" * 50)
    
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_results = cross_validate_hmm(X_all, y_all, n_folds=5, n_iter=30)
