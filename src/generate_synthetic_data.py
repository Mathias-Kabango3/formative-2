#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import random

print("Creating synthetic data generator...")

def generate_timestamps(num_samples, sampling_rate=20):
    start_time = datetime.now()
    timestamps = []
    seconds_elapsed = []
    for i in range(num_samples):
        ts = start_time + timedelta(seconds=i/sampling_rate)
        timestamps.append(int(ts.timestamp() * 1e9))
        seconds_elapsed.append(i/sampling_rate)
    return timestamps, seconds_elapsed

# Generate 17 synthetic recordings
print("Starting generation...")
target = 17
generated = 0
counter = 2

while generated < target:
    for activity in ['Standing', 'Walking', 'Jumping', 'Still']:
        if generated >= target:
            break
        
        folder_name = f"Daniel_{activity}_{counter:02d}_synthetic"
        folder_path = os.path.join('data/raw', folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create simple data
        num_samples = 200
        timestamps, seconds = generate_timestamps(num_samples)
        
        # Create Accelerometer.csv
        acc_df = pd.DataFrame({
            'time': timestamps,
            'seconds_elapsed': seconds,
            'z': np.random.normal(9.8, 0.1, num_samples),
            'y': np.random.normal(0, 0.1, num_samples),
            'x': np.random.normal(0, 0.1, num_samples)
        })
        acc_df.to_csv(os.path.join(folder_path, 'Accelerometer.csv'), index=False)
        
        # Create Gyroscope.csv
        gyro_df = pd.DataFrame({
            'time': timestamps,
            'seconds_elapsed': seconds,
            'z': np.random.normal(0, 0.05, num_samples),
            'y': np.random.normal(0, 0.05, num_samples),
            'x': np.random.normal(0, 0.05, num_samples)
        })
        gyro_df.to_csv(os.path.join(folder_path, 'Gyroscope.csv'), index=False)
        
        # Create Metadata.csv
        with open(os.path.join(folder_path, 'Metadata.csv'), 'w') as f:
            f.write('key,value\ndevice_model,Synthetic\nsampling_rate,20\n')
        
        # Create empty Annotation.csv
        with open(os.path.join(folder_path, 'Annotation.csv'), 'w') as f:
            f.write('')
        
        generated += 1
        print(f"Created {generated}/{target}: {folder_name}")
    
    counter += 1

print(f"\n✅ Done! Created {generated} synthetic recordings")
