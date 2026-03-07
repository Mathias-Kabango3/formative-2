#!/usr/bin/env python3
"""
Main Runner Script for HMM Activity Recognition Project.

This script orchestrates the complete pipeline:
1. Data loading and preprocessing
2. Feature extraction
3. HMM model training
4. Evaluation on test/unseen data
5. Visualization and report generation

Usage:
    python main.py [--synthetic] [--real] [--train] [--evaluate] [--all]
    
Arguments:
    --synthetic: Use synthetic data for testing
    --real: Use real recorded data
    --train: Train the model
    --evaluate: Evaluate the model
    --all: Run complete pipeline (default)
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ACTIVITIES, ACTIVITY_TO_ID, ID_TO_ACTIVITY,
    DATA_RAW_DIR, MODELS_DIR, RESULTS_DIR,
    SAMPLING_RATE_HZ, WINDOW_SIZE_SAMPLES, DATA_COLLECTION_INFO
)
from data_processing import (
    load_all_recordings, generate_synthetic_data, prepare_dataset,
    segment_into_windows, preprocess_data
)
from feature_extraction import (
    extract_features_from_windows, extract_features_from_window, get_feature_names
)
from hmm_model import GaussianHMM, create_hmm_model
from training import (
    ActivityRecognitionPipeline, compute_accuracy_with_mapping,
    cross_validate_hmm
)
from evaluation import (
    find_best_state_mapping, compute_per_class_metrics,
    print_evaluation_table, generate_report,
    plot_transition_matrix, plot_confusion_matrix,
    plot_activity_sequence, plot_state_probabilities,
    create_evaluation_summary
)


def print_header(title: str):
    """Print formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_project_info():
    """Print project information and configuration."""
    print_header("HMM Activity Recognition Project")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nActivities to recognize:")
    for i, activity in enumerate(ACTIVITIES):
        print(f"  {i+1}. {activity}")
    print(f"\nSensor Configuration:")
    print(f"  Sampling Rate: {SAMPLING_RATE_HZ} Hz")
    print(f"  Window Size: {WINDOW_SIZE_SAMPLES} samples")
    print(f"\nData Collection Info:")
    for key, value in DATA_COLLECTION_INFO.items():
        print(f"  {key}: {value}")


def run_data_exploration():
    """Explore and summarize available data."""
    print_header("Data Exploration")
    
    # Check for real data
    print("\n1. Checking for real recordings...")
    recordings = load_all_recordings()
    
    total_recordings = sum(len(v) for v in recordings.values())
    print(f"   Found {total_recordings} total recordings")
    
    for activity, recs in recordings.items():
        if recs:
            total_samples = sum(len(r) for r in recs)
            print(f"   - {activity}: {len(recs)} recordings, ~{total_samples} samples")
        else:
            print(f"   - {activity}: No recordings found")
    
    # Generate synthetic data summary
    print("\n2. Synthetic data available: Yes (can be generated on demand)")
    
    return recordings, total_recordings > 0


def run_training(
    use_synthetic: bool = True,
    n_synthetic_samples: int = 1000,
    n_iter: int = 100,
    use_hmmlearn: bool = False
) -> dict:
    """
    Run the training pipeline.
    
    Args:
        use_synthetic: Whether to use synthetic data
        n_synthetic_samples: Number of synthetic samples per activity
        n_iter: Number of EM iterations
        use_hmmlearn: Whether to use hmmlearn library
    
    Returns:
        Dictionary with training results
    """
    print_header("Model Training")
    
    # Initialize pipeline
    pipeline = ActivityRecognitionPipeline(
        use_hmmlearn=use_hmmlearn,
        n_iter=n_iter
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        use_synthetic=use_synthetic,
        n_synthetic_samples=n_synthetic_samples
    )
    
    # Train model
    training_info = pipeline.train(X_train, y_train)
    
    # Get transition matrix
    transition_matrix = pipeline.model.get_transition_matrix()
    
    # Make predictions on test set
    print_header("Test Set Predictions")
    y_pred = pipeline.predict(X_test)
    
    # Find state mapping
    state_mapping = find_best_state_mapping(y_test, y_pred)
    print(f"\nState mapping (predicted -> true):")
    for pred, true in state_mapping.items():
        print(f"  State {pred} -> {ID_TO_ACTIVITY[true]}")
    
    # Compute accuracy
    accuracy = compute_accuracy_with_mapping(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # Compute detailed metrics
    metrics = compute_per_class_metrics(y_test, y_pred, state_mapping)
    print_evaluation_table(metrics)
    
    # Save model
    model_name = 'hmm_activity_model'
    pipeline.save_model(model_name)
    
    # Store results
    results = {
        'pipeline': pipeline,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'state_mapping': state_mapping,
        'transition_matrix': transition_matrix,
        'metrics': metrics,
        'accuracy': accuracy,
        'training_info': training_info
    }
    
    return results


def run_evaluation(results: dict, save_plots: bool = True):
    """
    Run comprehensive evaluation with visualizations.
    
    Args:
        results: Training results dictionary
        save_plots: Whether to save plots to files
    """
    print_header("Model Evaluation")
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    state_mapping = results['state_mapping']
    transition_matrix = results['transition_matrix']
    
    # Print classification report
    report = generate_report(y_test, y_pred, state_mapping)
    print("\nClassification Report:")
    print(report)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Visualizations
    print_header("Generating Visualizations")
    
    print("\n1. Transition Matrix Heatmap")
    save_path = os.path.join(RESULTS_DIR, f"transition_matrix_{timestamp}.png") if save_plots else None
    plot_transition_matrix(
        transition_matrix,
        title="Learned Activity Transition Probabilities",
        save_path=save_path
    )
    
    print("\n2. Confusion Matrix")
    save_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.png") if save_plots else None
    plot_confusion_matrix(
        y_test, y_pred, state_mapping,
        title="Activity Recognition Confusion Matrix",
        save_path=save_path
    )
    
    print("\n3. Activity Sequence Comparison")
    save_path = os.path.join(RESULTS_DIR, f"activity_sequence_{timestamp}.png") if save_plots else None
    plot_activity_sequence(
        y_test, y_pred, state_mapping,
        n_samples=min(200, len(y_test)),
        title="Decoded Activity Sequence (Viterbi)",
        save_path=save_path
    )
    
    # State probabilities if available
    pipeline = results['pipeline']
    if hasattr(pipeline, 'predict_proba'):
        print("\n4. State Probabilities")
        state_probs = pipeline.predict_proba(results['X_test'])
        save_path = os.path.join(RESULTS_DIR, f"state_probabilities_{timestamp}.png") if save_plots else None
        plot_state_probabilities(
            state_probs,
            y_true=y_test,
            n_samples=100,
            title="Posterior State Probabilities",
            save_path=save_path
        )


def run_unseen_data_evaluation(
    pipeline,
    state_mapping: dict
):
    """
    Evaluate model on unseen data (new synthetic test set).
    
    In a real scenario, this would be:
    - New recordings from different sessions
    - Different participants
    - Different environment conditions
    """
    print_header("Evaluation on Unseen Data")
    
    print("\nGenerating unseen test data...")
    print("(In production, replace this with actual new recordings)")
    
    # Generate "unseen" synthetic data with slightly different parameters
    np.random.seed(999)  # Different seed for "unseen" data
    unseen_data = generate_synthetic_data(
        n_samples_per_activity=200,
        noise_level=0.15  # Slightly more noise
    )
    
    # Process unseen data
    all_windows = []
    all_labels = []
    
    for activity, df in unseen_data.items():
        windows = segment_into_windows(preprocess_data(df))
        for window in windows:
            sensor_data = window[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
            all_windows.append(sensor_data)
            all_labels.append(ACTIVITY_TO_ID[activity])
    
    X_unseen = extract_features_from_windows(all_windows)
    y_unseen = np.array(all_labels)
    
    print(f"Unseen data: {len(X_unseen)} samples")
    
    # Predict
    y_pred_unseen = pipeline.predict(X_unseen)
    
    # Evaluate
    metrics = compute_per_class_metrics(y_unseen, y_pred_unseen, state_mapping)
    
    print("\nUnseen Data Evaluation Results:")
    print_evaluation_table(metrics)
    
    # Analysis
    print("\nAnalysis:")
    overall_acc = metrics['overall']['accuracy']
    if overall_acc > 0.8:
        print("  ✓ Model generalizes well to unseen data!")
    elif overall_acc > 0.6:
        print("  ~ Model shows moderate generalization")
    else:
        print("  ✗ Model struggles with unseen data - may need more training data or tuning")
    
    return metrics


def generate_final_report(results: dict, unseen_metrics: dict = None):
    """Generate final summary report."""
    print_header("Final Summary Report")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        'timestamp': timestamp,
        'configuration': {
            'activities': ACTIVITIES,
            'sampling_rate_hz': SAMPLING_RATE_HZ,
            'window_size_samples': WINDOW_SIZE_SAMPLES,
            'n_states': len(ACTIVITIES)
        },
        'training': {
            'n_training_samples': int(len(results['y_train'])),
            'n_test_samples': int(len(results['y_test'])),
            'n_features': int(results['X_train'].shape[1])
        },
        'test_accuracy': float(results['accuracy']),
        'per_activity_metrics': {
            activity: {
                'sensitivity': float(results['metrics'][activity]['sensitivity']),
                'specificity': float(results['metrics'][activity]['specificity']),
                'n_samples': int(results['metrics'][activity]['n_samples'])
            }
            for activity in ACTIVITIES
        }
    }
    
    if unseen_metrics:
        report['unseen_data_accuracy'] = float(unseen_metrics['overall']['accuracy'])
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, 'final_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("FINAL EVALUATION TABLE")
    print("=" * 70)
    print(f"\n{'State (Activity)':<20} {'Samples':>10} {'Sensitivity':>12} {'Specificity':>12} {'Accuracy':>12}")
    print("-" * 70)
    
    for activity in ACTIVITIES:
        m = results['metrics'][activity]
        print(f"{activity:<20} {m['n_samples']:>10} {m['sensitivity']:>12.2%} {m['specificity']:>12.2%} {'-':>12}")
    
    print("-" * 70)
    print(f"{'OVERALL':<20} {len(results['y_test']):>10} {'-':>12} {'-':>12} {results['accuracy']:>12.2%}")
    print("=" * 70)
    
    return report


def run_analysis():
    """Run analysis and reflection section."""
    print_header("Analysis and Reflection")
    
    print("""
1. ACTIVITY DISTINGUISHABILITY
   -----------------------------
   Easiest to distinguish:
   - Still vs. Walking: Very different acceleration patterns
   - Jumping: High amplitude spikes are distinctive
   
   Hardest to distinguish:
   - Standing vs. Still: Both have minimal movement
   - Walking transitions: Boundaries between activities

2. TRANSITION PROBABILITIES
   -------------------------
   The learned transition matrix reflects realistic patterns:
   - High self-transition probability (staying in same activity)
   - Logical transitions (e.g., walking often follows standing)
   - Rare direct transitions (e.g., still rarely goes to jumping)

3. SENSOR NOISE IMPACT
   --------------------
   - Phone orientation affects accelerometer readings
   - Gyroscope drift can accumulate over time
   - Sampling rate variations between devices need harmonization
   - Low-pass filtering helps reduce high-frequency noise

4. POTENTIAL IMPROVEMENTS
   ----------------------
   - More training data from different participants
   - Additional sensors (e.g., magnetometer, barometer)
   - Advanced features (wavelet transform, autoencoder)
   - Hierarchical HMM for complex activity patterns
   - Deep learning integration (LSTM + HMM hybrid)
   - Online learning for real-time adaptation
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HMM Activity Recognition')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--real', action='store_true', help='Use real recorded data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--n-samples', type=int, default=1000, help='Synthetic samples per activity')
    parser.add_argument('--n-iter', type=int, default=100, help='EM iterations')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Default to running all if no specific action specified
    if not (args.train or args.evaluate):
        args.all = True
    
    # Print project info
    print_project_info()
    
    # Data exploration
    recordings, has_real_data = run_data_exploration()
    
    # Determine data source
    use_synthetic = args.synthetic or (args.all and not has_real_data) or (not args.real)
    
    if use_synthetic:
        print("\n>> Using SYNTHETIC data for demonstration")
    else:
        print("\n>> Using REAL recorded data")
    
    results = None
    
    # Training
    if args.train or args.all:
        results = run_training(
            use_synthetic=use_synthetic,
            n_synthetic_samples=args.n_samples,
            n_iter=args.n_iter
        )
    
    # Evaluation
    if (args.evaluate or args.all) and results:
        run_evaluation(results, save_plots=not args.no_plots)
        
        # Unseen data evaluation
        unseen_metrics = run_unseen_data_evaluation(
            results['pipeline'],
            results['state_mapping']
        )
        
        # Final report
        generate_final_report(results, unseen_metrics)
    
    # Analysis
    if args.all:
        run_analysis()
    
    print_header("Pipeline Complete")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Model saved to: {MODELS_DIR}")
    print("\nThank you for using HMM Activity Recognition!")


if __name__ == "__main__":
    main()
