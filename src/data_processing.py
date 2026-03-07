"""
Data Processing Module for HMM Activity Recognition.

This module handles:
- Loading raw sensor data from CSV files
- Generating synthetic data for testing
- Preprocessing and normalization
- Data segmentation into windows
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, ACTIVITIES, ACTIVITY_TO_ID,
    SAMPLING_RATE_HZ, SAMPLING_RATE_MS, WINDOW_SIZE_SAMPLES, WINDOW_OVERLAP_SAMPLES
)


def extract_activity_from_folder_name(folder_name: str) -> str:
    """
    Extract activity label from folder name.
    
    Supports multiple naming conventions:
    - 'Mathias_walking-2026-03-07_17-44-45' -> walking
    - 'Walking_01-2026-03-07_19-51-59' -> walking
    - 'Jumping_03-2026-03-07_19-48-14' -> jumping
    """
    folder_lower = folder_name.lower()
    
    # Check for known activities in folder name
    activity_keywords = {
        'jumping': 'jumping',
        'walking': 'walking', 
        'standing': 'standing',
        'still': 'still',
        'no_movement': 'still',
        'no movement': 'still',
    }
    
    for keyword, activity in activity_keywords.items():
        if keyword in folder_lower:
            return activity
    
    # Fallback: try to extract from naming pattern
    name_part = folder_name.split('-')[0]
    parts = name_part.split('_')
    
    if len(parts) >= 1:
        # Check first part (e.g., "Walking" from "Walking_01")
        first_part = parts[0].lower()
        if first_part in activity_keywords:
            return activity_keywords[first_part]
        
        # Check second part (e.g., "walking" from "Mathias_walking")
        if len(parts) >= 2:
            second_part = parts[1].lower()
            if second_part in activity_keywords:
                return activity_keywords[second_part]
    
    return 'unknown'


def load_sensor_data(folder_path: str, sensor: str = 'Accelerometer') -> pd.DataFrame:
    """
    Load sensor data from a CSV file.
    
    Args:
        folder_path: Path to the recording folder
        sensor: Sensor type ('Accelerometer' or 'Gyroscope')
    
    Returns:
        DataFrame with sensor data
    """
    file_path = os.path.join(folder_path, f'{sensor}.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Rename columns for consistency (x, y, z -> acc_x or gyro_x, etc.)
        prefix = 'acc' if sensor == 'Accelerometer' else 'gyro'
        df = df.rename(columns={
            'x': f'{prefix}_x',
            'y': f'{prefix}_y',
            'z': f'{prefix}_z'
        })
        return df
    return None


def load_recording(folder_path: str) -> pd.DataFrame:
    """
    Load both accelerometer and gyroscope data from a recording folder.
    
    Args:
        folder_path: Path to the recording folder
    
    Returns:
        Combined DataFrame with all sensor data
    """
    acc_data = load_sensor_data(folder_path, 'Accelerometer')
    gyro_data = load_sensor_data(folder_path, 'Gyroscope')
    
    if acc_data is None or gyro_data is None:
        return None
    
    # Merge on time (nearest match for slightly different timestamps)
    acc_data = acc_data[['time', 'seconds_elapsed', 'acc_x', 'acc_y', 'acc_z']]
    gyro_data = gyro_data[['time', 'gyro_x', 'gyro_y', 'gyro_z']]
    
    # Merge accelerometer and gyroscope data
    merged = pd.merge_asof(
        acc_data.sort_values('time'),
        gyro_data.sort_values('time'),
        on='time',
        direction='nearest'
    )
    
    return merged


def load_all_recordings(data_dir: str = DATA_RAW_DIR) -> Dict[str, List[pd.DataFrame]]:
    """
    Load all recordings from the raw data directory.
    
    Args:
        data_dir: Path to the raw data directory
    
    Returns:
        Dictionary mapping activity names to lists of DataFrames
    """
    recordings = {activity: [] for activity in ACTIVITIES}
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return recordings
    
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            activity = extract_activity_from_folder_name(folder_name)
            if activity in ACTIVITIES:
                df = load_recording(folder_path)
                if df is not None:
                    df['activity'] = activity
                    df['activity_id'] = ACTIVITY_TO_ID[activity]
                    recordings[activity].append(df)
                    print(f"Loaded {folder_name} -> {activity} ({len(df)} samples)")
    
    return recordings


def generate_synthetic_data(
    n_samples_per_activity: int = 1000,
    sampling_rate: float = SAMPLING_RATE_HZ,
    noise_level: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic sensor data for testing.
    
    Each activity has characteristic patterns:
    - Standing: Low variation, centered values
    - Walking: Periodic oscillation in acceleration
    - Jumping: High amplitude spikes
    - Still: Near-zero values with minimal noise
    
    Args:
        n_samples_per_activity: Number of samples to generate per activity
        sampling_rate: Sampling rate in Hz
        noise_level: Standard deviation of noise to add
    
    Returns:
        Dictionary mapping activity names to DataFrames
    """
    np.random.seed(42)
    synthetic_data = {}
    
    time_axis = np.arange(n_samples_per_activity) / sampling_rate
    timestamps = (time_axis * 1e9).astype(int)  # Nanosecond timestamps
    
    for activity in ACTIVITIES:
        # Initialize arrays
        acc_x = np.zeros(n_samples_per_activity)
        acc_y = np.zeros(n_samples_per_activity)
        acc_z = np.zeros(n_samples_per_activity)
        gyro_x = np.zeros(n_samples_per_activity)
        gyro_y = np.zeros(n_samples_per_activity)
        gyro_z = np.zeros(n_samples_per_activity)
        
        if activity == 'standing':
            # Standing: gravity on z-axis, small variations
            acc_z = np.ones(n_samples_per_activity) * -9.8
            acc_x = np.random.normal(0, 0.1, n_samples_per_activity)
            acc_y = np.random.normal(0, 0.1, n_samples_per_activity)
            acc_z += np.random.normal(0, 0.1, n_samples_per_activity)
            gyro_x = np.random.normal(0, 0.02, n_samples_per_activity)
            gyro_y = np.random.normal(0, 0.02, n_samples_per_activity)
            gyro_z = np.random.normal(0, 0.02, n_samples_per_activity)
            
        elif activity == 'walking':
            # Walking: periodic patterns with ~2Hz frequency
            freq = 2.0  # Steps per second
            acc_x = 0.5 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.2, n_samples_per_activity)
            acc_y = 0.8 * np.sin(2 * np.pi * freq * time_axis + np.pi/4) + np.random.normal(0, 0.2, n_samples_per_activity)
            acc_z = -9.8 + 0.6 * np.sin(2 * np.pi * freq * 2 * time_axis) + np.random.normal(0, 0.2, n_samples_per_activity)
            gyro_x = 0.3 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.05, n_samples_per_activity)
            gyro_y = 0.2 * np.cos(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.05, n_samples_per_activity)
            gyro_z = 0.1 * np.sin(2 * np.pi * freq * time_axis + np.pi/2) + np.random.normal(0, 0.05, n_samples_per_activity)
            
        elif activity == 'jumping':
            # Jumping: high amplitude periodic spikes ~1.5Hz
            freq = 1.5  # Jumps per second
            jump_signal = np.abs(np.sin(2 * np.pi * freq * time_axis))
            acc_z = -9.8 + 15.0 * jump_signal * np.exp(-((time_axis % (1/freq)) * freq * 3))
            acc_x = 1.0 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.5, n_samples_per_activity)
            acc_y = 0.8 * np.cos(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.5, n_samples_per_activity)
            acc_z += np.random.normal(0, 0.3, n_samples_per_activity)
            gyro_x = 0.5 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.1, n_samples_per_activity)
            gyro_y = 0.4 * np.cos(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.1, n_samples_per_activity)
            gyro_z = np.random.normal(0, 0.1, n_samples_per_activity)
            
        elif activity == 'still':
            # Still: minimal values, very low noise
            acc_z = -9.8 * np.ones(n_samples_per_activity)
            acc_x = np.random.normal(0, 0.01, n_samples_per_activity)
            acc_y = np.random.normal(0, 0.01, n_samples_per_activity)
            acc_z += np.random.normal(0, 0.01, n_samples_per_activity)
            gyro_x = np.random.normal(0, 0.005, n_samples_per_activity)
            gyro_y = np.random.normal(0, 0.005, n_samples_per_activity)
            gyro_z = np.random.normal(0, 0.005, n_samples_per_activity)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': timestamps,
            'seconds_elapsed': time_axis,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'activity': activity,
            'activity_id': ACTIVITY_TO_ID[activity]
        })
        
        synthetic_data[activity] = df
        print(f"Generated synthetic data for {activity}: {len(df)} samples")
    
    return synthetic_data


def segment_into_windows(
    data: pd.DataFrame,
    window_size: int = WINDOW_SIZE_SAMPLES,
    overlap: int = WINDOW_OVERLAP_SAMPLES
) -> List[pd.DataFrame]:
    """
    Segment time series data into overlapping windows.
    
    Args:
        data: DataFrame with sensor data
        window_size: Number of samples per window
        overlap: Number of overlapping samples between windows
    
    Returns:
        List of DataFrames, each representing a window
    """
    step = window_size - overlap
    windows = []
    
    for start in range(0, len(data) - window_size + 1, step):
        window = data.iloc[start:start + window_size].copy()
        windows.append(window)
    
    return windows


def preprocess_data(
    data: pd.DataFrame,
    apply_lowpass: bool = True,
    cutoff_freq: float = 20.0
) -> pd.DataFrame:
    """
    Preprocess sensor data.
    
    Args:
        data: DataFrame with sensor data
        apply_lowpass: Whether to apply low-pass filter
        cutoff_freq: Cutoff frequency for low-pass filter (Hz)
    
    Returns:
        Preprocessed DataFrame
    """
    df = data.copy()
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Apply low-pass filter to reduce noise
    if apply_lowpass and len(df) > 20:
        nyquist = SAMPLING_RATE_HZ / 2
        normalized_cutoff = cutoff_freq / nyquist
        if normalized_cutoff < 1:
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            for col in sensor_cols:
                if col in df.columns:
                    df[col] = signal.filtfilt(b, a, df[col])
    
    return df


def normalize_data(data: pd.DataFrame, method: str = 'zscore') -> Tuple[pd.DataFrame, dict]:
    """
    Normalize sensor data.
    
    Args:
        data: DataFrame with sensor data
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        Normalized DataFrame and normalization parameters
    """
    df = data.copy()
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    params = {}
    
    for col in sensor_cols:
        if col in df.columns:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                params[col] = {'mean': mean, 'std': std}
                df[col] = (df[col] - mean) / (std + 1e-8)
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                params[col] = {'min': min_val, 'max': max_val}
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
    
    return df, params


def prepare_dataset(
    use_synthetic: bool = False,
    n_synthetic_samples: int = 1000
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Prepare dataset for training.
    
    Args:
        use_synthetic: Whether to use synthetic data
        n_synthetic_samples: Number of synthetic samples per activity
    
    Returns:
        Tuple of (list of windowed data arrays, list of activity labels)
    """
    if use_synthetic:
        data_dict = generate_synthetic_data(n_samples_per_activity=n_synthetic_samples)
    else:
        data_dict = load_all_recordings()
    
    all_windows = []
    all_labels = []
    
    for activity, recordings in data_dict.items():
        if isinstance(recordings, pd.DataFrame):
            recordings = [recordings]
        
        for recording in recordings:
            # Preprocess
            processed = preprocess_data(recording)
            
            # Segment into windows
            windows = segment_into_windows(processed)
            
            for window in windows:
                # Extract sensor columns
                sensor_data = window[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
                all_windows.append(sensor_data)
                all_labels.append(ACTIVITY_TO_ID[activity])
    
    return all_windows, all_labels


def save_processed_data(windows: List[np.ndarray], labels: List[int], filename: str = 'processed_data.npz'):
    """Save processed data to file."""
    save_path = os.path.join(DATA_PROCESSED_DIR, filename)
    np.savez(save_path, windows=np.array(windows, dtype=object), labels=np.array(labels))
    print(f"Saved processed data to {save_path}")


def load_processed_data(filename: str = 'processed_data.npz') -> Tuple[List[np.ndarray], List[int]]:
    """Load processed data from file."""
    load_path = os.path.join(DATA_PROCESSED_DIR, filename)
    data = np.load(load_path, allow_pickle=True)
    return list(data['windows']), list(data['labels'])


if __name__ == "__main__":
    # Test data loading and synthetic generation
    print("=" * 50)
    print("Testing Data Processing Module")
    print("=" * 50)
    
    # Try loading real data
    print("\n1. Loading real recordings...")
    recordings = load_all_recordings()
    total_real = sum(len(v) for v in recordings.values())
    print(f"Loaded {total_real} real recordings")
    
    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    synthetic = generate_synthetic_data(n_samples_per_activity=500)
    
    # Test windowing
    print("\n3. Testing window segmentation...")
    for activity, df in synthetic.items():
        windows = segment_into_windows(df)
        print(f"  {activity}: {len(windows)} windows")
    
    # Prepare full dataset
    print("\n4. Preparing full dataset...")
    windows, labels = prepare_dataset(use_synthetic=True, n_synthetic_samples=500)
    print(f"Total windows: {len(windows)}")
    print(f"Windows per activity: {dict(zip(*np.unique(labels, return_counts=True)))}")
