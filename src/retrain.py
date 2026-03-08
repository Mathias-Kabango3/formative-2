#!/usr/bin/env python3
"""
Quick Retrain Script for HMM Activity Recognition.

Run this script whenever you add new recordings to data/raw/

Usage:
    python3 retrain.py
    
    # With options:
    python3 retrain.py --show-plots      # Show visualizations
    python3 retrain.py --iterations 200  # More training iterations
    python3 retrain.py --test-files 2    # Number of files to hold out for testing
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ACTIVITIES, ACTIVITY_TO_ID, DATA_RAW_DIR, MODELS_DIR, RESULTS_DIR
from data_processing import (
    load_all_recordings, load_recordings_with_holdout,
    segment_into_windows, preprocess_data, prepare_dataset_with_holdout,
    save_train_test_split
)
from feature_extraction import extract_features_from_windows, get_feature_names
from training import ActivityRecognitionPipeline, compute_accuracy_with_mapping
from evaluation import (
    find_best_state_mapping, compute_per_class_metrics, print_evaluation_table,
    plot_transition_matrix, plot_confusion_matrix, plot_activity_sequence,
    plot_emission_probabilities
)
from generate_report import generate_report
import numpy as np
from sklearn.model_selection import train_test_split


def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def scan_recordings():
    """Scan and summarize all recordings in data/raw."""
    print_header("Scanning Recordings")
    
    recordings = load_all_recordings()
    
    total = 0
    summary = {}
    
    for activity in ACTIVITIES:
        recs = recordings.get(activity, [])
        count = len(recs)
        samples = sum(len(r) for r in recs) if recs else 0
        summary[activity] = {'recordings': count, 'samples': samples}
        total += count
        
        status = "✓" if count >= 5 else "⚠" if count >= 1 else "✗"
        print(f"  {status} {activity}: {count} recordings ({samples} samples)")
    
    print(f"\n  Total: {total} recordings")
    
    return recordings, summary


def prepare_features(recordings):
    """Extract features from all recordings."""
    print_header("Feature Extraction")
    
    all_windows = []
    all_labels = []
    
    for activity in ACTIVITIES:
        recs = recordings.get(activity, [])
        activity_windows = 0
        
        for recording in recs:
            processed = preprocess_data(recording)
            windows = segment_into_windows(processed)
            
            for window in windows:
                sensor_data = window[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
                all_windows.append(sensor_data)
                all_labels.append(ACTIVITY_TO_ID[activity])
                activity_windows += 1
        
        print(f"  {activity}: {activity_windows} windows")
    
    X = extract_features_from_windows(all_windows)
    y = np.array(all_labels)
    
    print(f"\n  Total: {len(X)} windows, {X.shape[1]} features")
    
    return X, y


def train_model(X, y, n_iter=100):
    """Train HMM model."""
    print_header("Training Model")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train
    pipeline = ActivityRecognitionPipeline(use_hmmlearn=False, n_iter=n_iter)
    pipeline.scaler.fit(X_train)
    X_train_scaled = pipeline.scaler.transform(X_train)
    
    from hmm_model import GaussianHMM
    pipeline.model = GaussianHMM(n_states=len(ACTIVITIES), n_iter=n_iter)
    pipeline.model.fit(X_train_scaled, verbose=True)
    pipeline.is_trained = True
    
    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(pipeline, X_test, y_test, show_plots=False, test_label="Test"):
    """Evaluate trained model."""
    print_header(f"Evaluation Results ({test_label})")
    
    X_test_scaled = pipeline.scaler.transform(X_test)
    y_pred = pipeline.model.predict(X_test_scaled)
    
    # Find state mapping
    state_mapping = find_best_state_mapping(y_test, y_pred)
    
    # Compute metrics
    metrics = compute_per_class_metrics(y_test, y_pred, state_mapping)
    print_evaluation_table(metrics)
    
    accuracy = metrics['overall']['accuracy']
    
    # Get transition matrix
    transition_matrix = pipeline.model.get_transition_matrix()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if show_plots:
        # Transition probabilities
        plot_transition_matrix(
            transition_matrix,
            title="Learned Transition Probabilities",
            save_path=os.path.join(RESULTS_DIR, f"transition_matrix_{timestamp}.png")
        )
        
        # Emission probabilities
        feature_names = get_feature_names()
        plot_emission_probabilities(
            pipeline.model,
            feature_names=feature_names,
            title="Emission Probability Parameters",
            save_path=os.path.join(RESULTS_DIR, f"emission_probs_{timestamp}.png")
        )
        
        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred, state_mapping,
            title=f"Confusion Matrix ({test_label})",
            save_path=os.path.join(RESULTS_DIR, f"confusion_matrix_{test_label.lower()}_{timestamp}.png")
        )
        
        # Activity sequence
        plot_activity_sequence(
            y_test, y_pred, state_mapping,
            title=f"Activity Sequence ({test_label})",
            save_path=os.path.join(RESULTS_DIR, f"activity_sequence_{test_label.lower()}_{timestamp}.png")
        )
    
    return accuracy, metrics, state_mapping, y_pred


def save_model(pipeline):
    """Save trained model."""
    print_header("Saving Model")
    pipeline.save_model('hmm_activity_model')


def prepare_features_from_windows(windows, labels):
    """Extract features from pre-segmented windows."""
    X = extract_features_from_windows(windows)
    y = np.array(labels)
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Retrain HMM model with current data')
    parser.add_argument('--show-plots', action='store_true', help='Show visualizations')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--test-files', type=int, default=2, help='Number of files to hold out for unseen testing')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" HMM Activity Recognition - Retrain with File-Level Holdout")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Load data with file-level holdout
    print_header("Loading Data with File-Level Holdout")
    
    train_windows, train_labels, test_windows, test_labels, test_file_names = \
        prepare_dataset_with_holdout(n_test_files=args.test_files, random_state=42)
    
    if len(train_windows) == 0:
        print("\n❌ No training data found! Add data to data/raw/ first.")
        return
    
    # Save train/test split to data/processed/
    save_train_test_split(train_windows, train_labels, test_windows, test_labels, test_file_names)
    
    print(f"\n  Held-out test files ({len(test_file_names)}):")
    for fname in test_file_names:
        print(f"    - {fname}")
    
    # Step 2: Extract features
    print_header("Feature Extraction")
    
    X_train_full, y_train_full = prepare_features_from_windows(train_windows, train_labels)
    X_unseen, y_unseen = prepare_features_from_windows(test_windows, test_labels)
    
    print(f"  Training data: {len(X_train_full)} windows, {X_train_full.shape[1]} features")
    print(f"  Unseen test data: {len(X_unseen)} windows")
    
    # Step 3: Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\n  Train split: {len(X_train)} windows")
    print(f"  Validation split: {len(X_val)} windows")
    
    # Step 4: Train model
    print_header("Training HMM Model")
    
    pipeline = ActivityRecognitionPipeline(use_hmmlearn=False, n_iter=args.iterations)
    pipeline.scaler.fit(X_train)
    X_train_scaled = pipeline.scaler.transform(X_train)
    
    from hmm_model import GaussianHMM
    pipeline.model = GaussianHMM(n_states=len(ACTIVITIES), n_iter=args.iterations)
    pipeline.model.fit(X_train_scaled, verbose=True)
    pipeline.is_trained = True
    
    # Step 5: Evaluate on validation set
    val_accuracy, val_metrics, val_mapping, _ = evaluate_model(
        pipeline, X_val, y_val, show_plots=False, test_label="Validation"
    )
    
    # Step 6: Evaluate on UNSEEN test files (the main evaluation for rubric)
    if len(X_unseen) > 0:
        print_header(f"UNSEEN TEST DATA EVALUATION ({len(test_file_names)} held-out files)")
        print(f"\nThese files were COMPLETELY EXCLUDED from training:")
        for fname in test_file_names:
            print(f"  - {fname}")
        
        unseen_accuracy, unseen_metrics, unseen_mapping, y_pred_unseen = evaluate_model(
            pipeline, X_unseen, y_unseen, show_plots=args.show_plots, test_label="Unseen"
        )
        
        # Print detailed metrics for unseen data
        print("\n" + "=" * 60)
        print(" DETAILED UNSEEN TEST METRICS")
        print("=" * 60)
        print(f"\n{'Activity':<12} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-" * 65)
        
        for activity in ACTIVITIES:
            if activity in unseen_metrics:
                m = unseen_metrics[activity]
                print(f"{activity:<12} {m['sensitivity']:>8.2%} {m['specificity']:>8.2%} "
                      f"{m['precision']:>8.2%} {m['f1_score']:>8.2%} "
                      f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
        
        print("-" * 65)
        print(f"{'OVERALL':<12} {'Accuracy:':>8} {unseen_accuracy:>26.2%}")
        print("=" * 60)
    else:
        print("\n⚠ No unseen test data available. Add more recordings.")
        unseen_accuracy = 0.0
    
    # Step 7: Save model
    save_model(pipeline)
    
    # Step 8: Generate Word report
    print_header("Generating Word Report")
    try:
        report_path = generate_report(
            test_file_names=test_file_names,
            unseen_metrics=unseen_metrics if len(X_unseen) > 0 else None,
            val_metrics=val_metrics,
            author_name="Mathias"  # Change this to your name
        )
        print(f"  Report saved to: {report_path}")
    except ImportError:
        print("  ⚠ python-docx not installed. Skipping report generation.")
        print("  Run: pip install python-docx")
    
    # Summary
    print_header("Training Complete - Summary")
    total_windows = len(train_windows) + len(test_windows)
    print(f"  Total windows: {total_windows}")
    print(f"  Training windows: {len(train_windows)}")
    print(f"  Unseen test windows: {len(test_windows)}")
    print(f"\n  Validation Accuracy: {val_accuracy:.1%}")
    if len(X_unseen) > 0:
        print(f"  Unseen Test Accuracy: {unseen_accuracy:.1%}")
    print(f"\n  Model saved to: {MODELS_DIR}")
    
    if args.show_plots:
        print(f"  Visualizations saved to: {RESULTS_DIR}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
