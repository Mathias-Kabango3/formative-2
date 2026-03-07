"""
Evaluation and Visualization Module for HMM Activity Recognition.

This module provides:
- Model evaluation metrics (sensitivity, specificity, accuracy)
- Visualization of transition matrices
- Activity sequence plots
- Confusion matrix visualization
- Feature importance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

from config import ACTIVITIES, ID_TO_ACTIVITY, ACTIVITY_TO_ID, RESULTS_DIR, N_STATES


def find_best_state_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, int]:
    """
    Find optimal mapping between HMM states and activity labels.
    Uses Hungarian algorithm for optimal assignment.
    
    Args:
        y_true: True activity labels
        y_pred: Predicted HMM states
    
    Returns:
        Dictionary mapping predicted states to true labels
    """
    n_classes = len(ACTIVITIES)
    
    # Build confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        if t < n_classes and p < n_classes:
            conf_matrix[t, p] += 1
    
    # Hungarian algorithm (maximize agreement = minimize negative)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    
    # Create mapping: predicted_state -> true_label
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    return mapping


def apply_state_mapping(y_pred: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Apply state mapping to predictions."""
    return np.array([mapping.get(p, p) for p in y_pred])


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_mapping: Dict[int, int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute sensitivity, specificity, and other metrics per class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        state_mapping: Optional mapping to apply to predictions
    
    Returns:
        Dictionary with metrics per activity
    """
    if state_mapping is not None:
        y_pred = apply_state_mapping(y_pred, state_mapping)
    
    n_classes = len(ACTIVITIES)
    metrics = {}
    
    for activity, activity_id in ACTIVITY_TO_ID.items():
        # Binary classification for this class
        y_true_binary = (y_true == activity_id).astype(int)
        y_pred_binary = (y_pred == activity_id).astype(int)
        
        # True positives, false positives, etc.
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Sensitivity (True Positive Rate / Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Precision (Positive Predictive Value)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        # Number of samples
        n_samples = int(np.sum(y_true_binary))
        
        metrics[activity] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'n_samples': n_samples,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    metrics['overall'] = {'accuracy': accuracy}
    
    return metrics


def print_evaluation_table(metrics: Dict[str, Dict[str, float]]):
    """
    Print evaluation metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics per activity
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Header
    print(f"\n{'Activity':<15} {'Samples':>10} {'Sensitivity':>12} {'Specificity':>12} {'Accuracy':>12}")
    print("-" * 65)
    
    overall_accuracy = metrics.get('overall', {}).get('accuracy', 0)
    
    for activity in ACTIVITIES:
        if activity in metrics:
            m = metrics[activity]
            print(f"{activity:<15} {m['n_samples']:>10} {m['sensitivity']:>12.2%} {m['specificity']:>12.2%} {'-':>12}")
    
    print("-" * 65)
    print(f"{'Overall':<15} {'-':>10} {'-':>12} {'-':>12} {overall_accuracy:>12.2%}")
    print("=" * 80)


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_mapping: Dict[int, int] = None
) -> str:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        state_mapping: Optional state mapping
    
    Returns:
        Report as string
    """
    if state_mapping is not None:
        y_pred = apply_state_mapping(y_pred, state_mapping)
    
    # Classification report
    target_names = [ID_TO_ACTIVITY[i] for i in range(len(ACTIVITIES))]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    return report


# ===== Visualization Functions =====

def plot_transition_matrix(
    transition_matrix: np.ndarray,
    state_names: List[str] = None,
    title: str = "HMM Transition Matrix",
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot transition matrix as a heatmap.
    
    Args:
        transition_matrix: Transition probability matrix (n_states x n_states)
        state_names: Names for each state
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if state_names is None:
        state_names = ACTIVITIES[:transition_matrix.shape[0]]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=state_names,
        yticklabels=state_names,
        ax=ax,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Transition Probability'}
    )
    
    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Transition matrix heatmap saved to {save_path}")
    
    plt.show()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_mapping: Dict[int, int] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        state_mapping: Optional state mapping
        normalize: Whether to normalize by true label count
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if state_mapping is not None:
        y_pred = apply_state_mapping(y_pred, state_mapping)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt if not normalize else '.2f',
        cmap='Blues',
        xticklabels=ACTIVITIES,
        yticklabels=ACTIVITIES,
        ax=ax,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Activity', fontsize=12)
    ax.set_ylabel('True Activity', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return fig


def plot_activity_sequence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_mapping: Dict[int, int] = None,
    n_samples: int = 200,
    title: str = "Activity Sequence Comparison",
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot decoded activity sequence vs true sequence.
    
    Args:
        y_true: True activity labels
        y_pred: Predicted activity labels
        state_mapping: Optional state mapping
        n_samples: Number of samples to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if state_mapping is not None:
        y_pred = apply_state_mapping(y_pred, state_mapping)
    
    # Limit samples
    n = min(n_samples, len(y_true))
    y_true_plot = y_true[:n]
    y_pred_plot = y_pred[:n]
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Color map for activities
    colors = plt.cm.Set2(np.linspace(0, 1, len(ACTIVITIES)))
    
    # True sequence
    for i, activity in enumerate(ACTIVITIES):
        mask = y_true_plot == i
        axes[0].fill_between(
            range(n), 0, 1,
            where=mask,
            color=colors[i],
            alpha=0.7,
            label=activity
        )
    axes[0].set_ylabel('True Activity', fontsize=12)
    axes[0].set_yticks([])
    axes[0].legend(loc='upper right', ncol=len(ACTIVITIES))
    axes[0].set_title('True Activity Sequence', fontsize=12)
    
    # Predicted sequence
    for i, activity in enumerate(ACTIVITIES):
        mask = y_pred_plot == i
        axes[1].fill_between(
            range(n), 0, 1,
            where=mask,
            color=colors[i],
            alpha=0.7
        )
    axes[1].set_ylabel('Predicted', fontsize=12)
    axes[1].set_xlabel('Time Window', fontsize=12)
    axes[1].set_yticks([])
    axes[1].set_title('Predicted Activity Sequence (Viterbi Decoding)', fontsize=12)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Activity sequence plot saved to {save_path}")
    
    plt.show()
    return fig


def plot_state_probabilities(
    state_probs: np.ndarray,
    y_true: np.ndarray = None,
    n_samples: int = 100,
    title: str = "State Probabilities Over Time",
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot state probabilities over time.
    
    Args:
        state_probs: State probability matrix (n_samples, n_states)
        y_true: Optional true labels for comparison
        n_samples: Number of samples to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    n = min(n_samples, len(state_probs))
    probs = state_probs[:n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(ACTIVITIES)))
    
    for i, activity in enumerate(ACTIVITIES):
        ax.plot(probs[:, i], label=activity, color=colors[i], linewidth=2)
    
    if y_true is not None:
        # Add markers for true labels
        for i in range(n):
            ax.axvline(x=i, color=colors[y_true[i]], alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('Time Window', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"State probabilities plot saved to {save_path}")
    
    plt.show()
    return fig


def plot_training_convergence(
    log_likelihoods: List[float],
    title: str = "Training Convergence",
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot log-likelihood convergence during training.
    
    Args:
        log_likelihoods: List of log-likelihood values per iteration
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(log_likelihoods, marker='o', markersize=4, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Log-Likelihood', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    plt.show()
    return fig


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_k: int = 20,
    title: str = "Feature Importance (Variance across States)",
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot feature importance based on variance across HMM states.
    
    Args:
        model: Trained HMM model
        feature_names: Names of features
        top_k: Number of top features to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get emission means for each state
    if hasattr(model, 'means'):
        means = model.means
    elif hasattr(model, 'means_'):
        means = model.means_
    else:
        print("Cannot extract means from model")
        return None
    
    # Compute variance of means across states (importance)
    importance = np.var(means, axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1][:top_k]
    top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in sorted_idx]
    top_importance = importance[sorted_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importance, align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Variance across States)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()
    return fig


def create_evaluation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transition_matrix: np.ndarray,
    state_mapping: Dict[int, int] = None,
    save_dir: str = RESULTS_DIR
) -> Dict:
    """
    Create comprehensive evaluation summary with all plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        transition_matrix: HMM transition matrix
        state_mapping: Optional state mapping
        save_dir: Directory to save results
    
    Returns:
        Dictionary with evaluation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find mapping if not provided
    if state_mapping is None:
        state_mapping = find_best_state_mapping(y_true, y_pred)
    
    # Compute metrics
    metrics = compute_per_class_metrics(y_true, y_pred, state_mapping)
    
    # Print results
    print_evaluation_table(metrics)
    
    # Generate report
    report = generate_report(y_true, y_pred, state_mapping)
    print("\nClassification Report:")
    print(report)
    
    # Create plots
    print("\nGenerating visualizations...")
    
    # 1. Transition matrix heatmap
    plot_transition_matrix(
        transition_matrix,
        title="Learned Transition Probabilities",
        save_path=os.path.join(save_dir, f"transition_matrix_{timestamp}.png")
    )
    
    # 2. Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, state_mapping,
        title="Activity Recognition Confusion Matrix",
        save_path=os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
    )
    
    # 3. Activity sequence
    plot_activity_sequence(
        y_true, y_pred, state_mapping,
        title="Decoded Activity Sequence",
        save_path=os.path.join(save_dir, f"activity_sequence_{timestamp}.png")
    )
    
    # Package results
    results = {
        'metrics': metrics,
        'state_mapping': state_mapping,
        'report': report,
        'timestamp': timestamp
    }
    
    return results


if __name__ == "__main__":
    # Test evaluation and visualization
    print("=" * 60)
    print("Testing Evaluation and Visualization Module")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    
    # Create realistic test data with some errors
    y_true = np.repeat(np.arange(4), n_samples // 4)
    y_pred = y_true.copy()
    
    # Introduce some prediction errors (~15%)
    error_mask = np.random.random(n_samples) < 0.15
    y_pred[error_mask] = np.random.randint(0, 4, np.sum(error_mask))
    
    # Create sample transition matrix
    transition_matrix = np.array([
        [0.85, 0.10, 0.02, 0.03],  # standing
        [0.08, 0.82, 0.05, 0.05],  # walking
        [0.05, 0.15, 0.75, 0.05],  # jumping
        [0.02, 0.03, 0.03, 0.92],  # still
    ])
    
    print(f"\nTest data: {n_samples} samples")
    print(f"True label distribution: {np.bincount(y_true)}")
    print(f"Predicted label distribution: {np.bincount(y_pred)}")
    
    # Compute metrics
    metrics = compute_per_class_metrics(y_true, y_pred)
    print_evaluation_table(metrics)
    
    # Test visualizations
    print("\n" + "-" * 40)
    print("Testing Visualizations")
    print("-" * 40)
    
    # Transition matrix
    plot_transition_matrix(
        transition_matrix,
        title="Test Transition Matrix"
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        title="Test Confusion Matrix"
    )
    
    # Activity sequence
    plot_activity_sequence(
        y_true, y_pred,
        n_samples=100,
        title="Test Activity Sequence"
    )
    
    print("\nAll tests completed!")
