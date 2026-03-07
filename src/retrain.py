#!/usr/bin/env python3
"""
Quick Retrain Script for HMM Activity Recognition.

Run this script whenever you add new recordings to data/raw/

Usage:
    python3 retrain.py
    
    # With options:
    python3 retrain.py --show-plots      # Show visualizations
    python3 retrain.py --iterations 200  # More training iterations
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ACTIVITIES, ACTIVITY_TO_ID, DATA_RAW_DIR, MODELS_DIR, RESULTS_DIR
from data_processing import load_all_recordings, segment_into_windows, preprocess_data
from feature_extraction import extract_features_from_windows
from training import ActivityRecognitionPipeline, compute_accuracy_with_mapping
from evaluation import (
    find_best_state_mapping, compute_per_class_metrics, print_evaluation_table,
    plot_transition_matrix, plot_confusion_matrix, plot_activity_sequence
)
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


def evaluate_model(pipeline, X_test, y_test, show_plots=False):
    """Evaluate trained model."""
    print_header("Evaluation Results")
    
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
        plot_transition_matrix(
            transition_matrix,
            title="Learned Transition Probabilities",
            save_path=os.path.join(RESULTS_DIR, f"transition_matrix_{timestamp}.png")
        )
        
        plot_confusion_matrix(
            y_test, y_pred, state_mapping,
            title="Confusion Matrix",
            save_path=os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.png")
        )
        
        plot_activity_sequence(
            y_test, y_pred, state_mapping,
            title="Activity Sequence",
            save_path=os.path.join(RESULTS_DIR, f"activity_sequence_{timestamp}.png")
        )
    
    return accuracy, metrics, state_mapping


def save_model(pipeline):
    """Save trained model."""
    print_header("Saving Model")
    pipeline.save_model('hmm_activity_model')


def main():
    parser = argparse.ArgumentParser(description='Retrain HMM model with current data')
    parser.add_argument('--show-plots', action='store_true', help='Show visualizations')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" HMM Activity Recognition - Retrain")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Scan recordings
    recordings, summary = scan_recordings()
    
    total_recs = sum(s['recordings'] for s in summary.values())
    if total_recs == 0:
        print("\n❌ No recordings found! Add data to data/raw/ first.")
        return
    
    # Step 2: Extract features
    X, y = prepare_features(recordings)
    
    if len(X) < 20:
        print("\n⚠ Warning: Very few samples. Consider adding more recordings.")
    
    # Step 3: Train model
    pipeline, X_train, X_test, y_train, y_test = train_model(X, y, args.iterations)
    
    # Step 4: Evaluate
    accuracy, metrics, state_mapping = evaluate_model(
        pipeline, X_test, y_test, show_plots=args.show_plots
    )
    
    # Step 5: Save
    save_model(pipeline)
    
    # Summary
    print_header("Summary")
    print(f"  Recordings: {total_recs}")
    print(f"  Windows: {len(X)}")
    print(f"  Test Accuracy: {accuracy:.1%}")
    print(f"\n  Model saved to: {MODELS_DIR}")
    
    if accuracy < 0.7:
        print("\n  💡 Tip: Accuracy is low. Try adding more recordings.")
    elif accuracy < 0.85:
        print("\n  💡 Tip: Good! More data would improve results further.")
    else:
        print("\n  ✓ Excellent accuracy!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
