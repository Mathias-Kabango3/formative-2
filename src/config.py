"""
Configuration settings for the HMM Activity Recognition project.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create directories if they don't exist
for dir_path in [DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Activity labels and mapping
ACTIVITIES = ["standing", "walking", "jumping", "still"]
ACTIVITY_TO_ID = {activity: idx for idx, activity in enumerate(ACTIVITIES)}
ID_TO_ACTIVITY = {idx: activity for idx, activity in enumerate(ACTIVITIES)}

# Sensor configuration
SAMPLING_RATE_MS = 10  # 10ms = 100Hz
SAMPLING_RATE_HZ = 1000 / SAMPLING_RATE_MS  # 100 Hz

# Feature extraction parameters
WINDOW_SIZE_MS = 500  # 500ms window for feature extraction
WINDOW_OVERLAP_MS = 250  # 50% overlap
WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_MS / SAMPLING_RATE_MS)  # 50 samples per window
WINDOW_OVERLAP_SAMPLES = int(WINDOW_OVERLAP_MS / SAMPLING_RATE_MS)  # 25 samples overlap

# HMM parameters
N_STATES = len(ACTIVITIES)  # Number of hidden states (activities)
N_COMPONENTS_GMM = 4  # Number of Gaussian components for emission probabilities

# Training parameters
TEST_SIZE = 0.2  # 20% of data for testing
RANDOM_STATE = 42

# Data collection info
DATA_COLLECTION_INFO = {
    "phone_model": "OPPO CPH2641",
    "sampling_rate_ms": SAMPLING_RATE_MS,
    "sensors": ["Accelerometer", "Gyroscope"],
    "group_member": "Mathias"
}
