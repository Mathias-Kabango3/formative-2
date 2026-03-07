"""
Utility script to generate fake sensor recordings in the exact format 
expected by the data processing module.

This generates CSV files that match the format of real phone sensor recordings.
You can replace these with real data later.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_RAW_DIR, ACTIVITIES, SAMPLING_RATE_MS


def generate_fake_recording(
    activity: str,
    duration_seconds: float = 10.0,
    sampling_rate_ms: int = SAMPLING_RATE_MS,
    participant_name: str = "Demo",
    noise_level: float = 0.1
) -> dict:
    """
    Generate a fake sensor recording for a given activity.
    
    Args:
        activity: Activity type ('standing', 'walking', 'jumping', 'still')
        duration_seconds: Recording duration in seconds
        sampling_rate_ms: Sampling rate in milliseconds
        participant_name: Name of the participant
        noise_level: Amount of noise to add
    
    Returns:
        Dictionary with DataFrames for each sensor type
    """
    n_samples = int(duration_seconds * 1000 / sampling_rate_ms)
    
    # Generate timestamps
    base_timestamp = int(datetime.now().timestamp() * 1e9)  # Nanoseconds
    timestamps = base_timestamp + np.arange(n_samples) * sampling_rate_ms * 1e6
    seconds_elapsed = np.arange(n_samples) * sampling_rate_ms / 1000
    
    # Initialize sensor data
    acc_x = np.zeros(n_samples)
    acc_y = np.zeros(n_samples)
    acc_z = np.zeros(n_samples)
    gyro_x = np.zeros(n_samples)
    gyro_y = np.zeros(n_samples)
    gyro_z = np.zeros(n_samples)
    
    time_axis = seconds_elapsed
    
    # Generate activity-specific patterns
    if activity == 'standing':
        # Standing: gravity on z-axis, small body sway
        acc_z = np.ones(n_samples) * -9.8
        acc_x = 0.05 * np.sin(2 * np.pi * 0.3 * time_axis) + np.random.normal(0, 0.02, n_samples)
        acc_y = 0.05 * np.cos(2 * np.pi * 0.3 * time_axis) + np.random.normal(0, 0.02, n_samples)
        acc_z += np.random.normal(0, 0.05, n_samples)
        gyro_x = np.random.normal(0, 0.01, n_samples)
        gyro_y = np.random.normal(0, 0.01, n_samples)
        gyro_z = np.random.normal(0, 0.01, n_samples)
        
    elif activity == 'walking':
        # Walking: periodic patterns ~2Hz step frequency
        freq = 1.8 + np.random.uniform(-0.2, 0.2)
        acc_x = 0.4 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.15, n_samples)
        acc_y = 0.6 * np.sin(2 * np.pi * freq * time_axis + np.pi/4) + np.random.normal(0, 0.15, n_samples)
        acc_z = -9.8 + 0.5 * np.sin(2 * np.pi * freq * 2 * time_axis) + np.random.normal(0, 0.2, n_samples)
        gyro_x = 0.2 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.05, n_samples)
        gyro_y = 0.15 * np.cos(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.05, n_samples)
        gyro_z = 0.1 * np.sin(2 * np.pi * freq * time_axis + np.pi/2) + np.random.normal(0, 0.03, n_samples)
        
    elif activity == 'jumping':
        # Jumping: high amplitude spikes ~1.5Hz
        freq = 1.5 + np.random.uniform(-0.2, 0.2)
        phase = (time_axis * freq) % 1
        
        # Jump phases: crouch, push, air, land
        for i in range(n_samples):
            p = phase[i]
            if p < 0.15:  # Crouch
                acc_z[i] = -9.8 + 3 * np.sin(np.pi * p / 0.15)
            elif p < 0.3:  # Push off
                acc_z[i] = -9.8 + 12 * np.sin(np.pi * (p - 0.15) / 0.15)
            elif p < 0.6:  # Air (free fall-ish)
                acc_z[i] = -9.8 + 2 * np.sin(np.pi * (p - 0.3) / 0.3)
            else:  # Landing
                acc_z[i] = -9.8 + 8 * np.exp(-5 * (p - 0.6))
        
        acc_x = 0.8 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.3, n_samples)
        acc_y = 0.6 * np.cos(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.3, n_samples)
        acc_z += np.random.normal(0, 0.3, n_samples)
        gyro_x = 0.4 * np.sin(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.1, n_samples)
        gyro_y = 0.3 * np.cos(2 * np.pi * freq * time_axis) + np.random.normal(0, 0.1, n_samples)
        gyro_z = np.random.normal(0, 0.05, n_samples)
        
    elif activity == 'still' or activity == 'no_movement':
        # Still: phone on flat surface, minimal noise
        acc_z = -9.8 * np.ones(n_samples)
        acc_x = np.random.normal(0, 0.005, n_samples)
        acc_y = np.random.normal(0, 0.005, n_samples)
        acc_z += np.random.normal(0, 0.005, n_samples)
        gyro_x = np.random.normal(0, 0.002, n_samples)
        gyro_y = np.random.normal(0, 0.002, n_samples)
        gyro_z = np.random.normal(0, 0.002, n_samples)
    
    # Create DataFrames matching the real data format
    accelerometer_df = pd.DataFrame({
        'time': timestamps.astype(int),
        'seconds_elapsed': seconds_elapsed,
        'z': acc_z,
        'y': acc_y,
        'x': acc_x
    })
    
    gyroscope_df = pd.DataFrame({
        'time': timestamps.astype(int),
        'seconds_elapsed': seconds_elapsed,
        'z': gyro_z,
        'y': gyro_y,
        'x': gyro_x
    })
    
    # Total acceleration magnitude
    total_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    total_acceleration_df = pd.DataFrame({
        'time': timestamps.astype(int),
        'seconds_elapsed': seconds_elapsed,
        'total_acceleration': total_acc
    })
    
    # Uncalibrated versions (slightly offset)
    acc_uncal_df = accelerometer_df.copy()
    acc_uncal_df['x'] += np.random.uniform(-0.01, 0.01)
    acc_uncal_df['y'] += np.random.uniform(-0.01, 0.01)
    acc_uncal_df['z'] += np.random.uniform(-0.01, 0.01)
    
    gyro_uncal_df = gyroscope_df.copy()
    gyro_uncal_df['x'] += np.random.uniform(-0.001, 0.001)
    gyro_uncal_df['y'] += np.random.uniform(-0.001, 0.001)
    gyro_uncal_df['z'] += np.random.uniform(-0.001, 0.001)
    
    # Annotation (empty for now)
    annotation_df = pd.DataFrame(columns=['time', 'annotation'])
    
    # Metadata
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metadata_df = pd.DataFrame([{
        'version': 3,
        'device name': 'Demo Device',
        'recording epoch time': int(datetime.now().timestamp() * 1000),
        'recording time': timestamp_str,
        'recording timezone': 'Africa/Kigali',
        'platform': 'demo',
        'appVersion': '1.0.0',
        'device id': 'demo-device-001',
        'sensors': 'Accelerometer|Gyroscope|Annotation|TotalAcceleration|GyroscopeUncalibrated|AccelerometerUncalibrated',
        'sampleRateMs': f'{sampling_rate_ms}|{sampling_rate_ms}||{sampling_rate_ms}|{sampling_rate_ms}|{sampling_rate_ms}',
        'standardisation': 'false',
        'platform version': 'demo'
    }])
    
    return {
        'Accelerometer': accelerometer_df,
        'Gyroscope': gyroscope_df,
        'TotalAcceleration': total_acceleration_df,
        'AccelerometerUncalibrated': acc_uncal_df,
        'GyroscopeUncalibrated': gyro_uncal_df,
        'Annotation': annotation_df,
        'Metadata': metadata_df
    }


def save_recording(
    recording: dict,
    activity: str,
    participant_name: str,
    output_dir: str = DATA_RAW_DIR
):
    """
    Save a recording to the data/raw directory.
    
    Args:
        recording: Dictionary with sensor DataFrames
        activity: Activity type
        participant_name: Participant name
        output_dir: Output directory
    """
    # Create folder name matching the format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{participant_name}_{activity}-{timestamp.replace('_', '-').replace('-', '-')}"
    folder_path = os.path.join(output_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    
    # Save each sensor file
    for sensor_name, df in recording.items():
        file_path = os.path.join(folder_path, f"{sensor_name}.csv")
        df.to_csv(file_path, index=False)
    
    print(f"Saved recording: {folder_name}")
    return folder_path


def generate_fake_dataset(
    n_recordings_per_activity: int = 3,
    duration_seconds: float = 10.0,
    participant_name: str = "Demo"
):
    """
    Generate a complete fake dataset with multiple recordings per activity.
    
    Args:
        n_recordings_per_activity: Number of recordings per activity
        duration_seconds: Duration of each recording
        participant_name: Participant name
    """
    print("=" * 50)
    print("Generating Fake Sensor Dataset")
    print("=" * 50)
    
    activities_map = {
        'jumping': 'jumping',
        'no_movement': 'still',
        'standing': 'standing',
        'walking': 'walking'
    }
    
    total_recordings = 0
    
    for folder_activity, data_activity in activities_map.items():
        print(f"\nGenerating {n_recordings_per_activity} recordings for '{folder_activity}'...")
        
        for i in range(n_recordings_per_activity):
            # Add variation
            duration = duration_seconds + np.random.uniform(-2, 2)
            
            recording = generate_fake_recording(
                activity=data_activity,
                duration_seconds=duration,
                participant_name=participant_name
            )
            
            save_recording(
                recording,
                activity=folder_activity,
                participant_name=participant_name
            )
            
            total_recordings += 1
            
            # Small delay to get different timestamps
            import time
            time.sleep(0.1)
    
    print(f"\n{'=' * 50}")
    print(f"Generated {total_recordings} total recordings")
    print(f"Saved to: {DATA_RAW_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fake sensor recordings')
    parser.add_argument('--n-recordings', type=int, default=3, 
                        help='Number of recordings per activity')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration in seconds')
    parser.add_argument('--participant', type=str, default='Demo',
                        help='Participant name')
    
    args = parser.parse_args()
    
    generate_fake_dataset(
        n_recordings_per_activity=args.n_recordings,
        duration_seconds=args.duration,
        participant_name=args.participant
    )
