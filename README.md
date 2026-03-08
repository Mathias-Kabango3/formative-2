# HMM Activity Recognition Project

This project implements a Hidden Markov Model (HMM) based activity recognition system using accelerometer and gyroscope sensor data from smartphones.

## Activities Recognized

| Activity | Description | Duration |
|----------|-------------|----------|
| Standing | Keep phone steady at waist level | 5-10 seconds |
| Walking | Maintain consistent pace | 5-10 seconds |
| Jumping | Continuous jumps | 5-10 seconds |
| Still | Phone on flat surface, no movement | 5-10 seconds |

## Project Structure

```
formative-2/
├── data/
│   ├── raw/                    # Raw sensor recordings (CSV files)
│   └── processed/              # Processed feature data
├── models/                     # Saved trained models
├── results/                    # Evaluation results and plots
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── feature_extraction.py   # Time & frequency domain features
│   ├── hmm_model.py           # HMM implementation (Viterbi, Baum-Welch)
│   ├── training.py            # Training pipeline
│   ├── evaluation.py          # Metrics and visualization
│   ├── generate_fake_data.py  # Fake data generator for testing
│   └── main.py                # Main entry point
└── requirements.txt
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Fake Data (for testing)

```bash
cd src
python generate_fake_data.py --n-recordings 5 --participant YourName
```

### 2. Run Complete Pipeline

```bash
python main.py --all
```

### 3. Train Only

```bash
python main.py --train --synthetic
```

### 4. Evaluate Only

```bash
python main.py --evaluate
```

## Raw Data Format

Each recording should be in a folder like: `ParticipantName_activity-YYYY-MM-DD_HH-MM-SS/`

Required files:
- `Accelerometer.csv`: time, seconds_elapsed, z, y, x
- `Gyroscope.csv`: time, seconds_elapsed, z, y, x
- `Metadata.csv`: Device info and sampling rate

## HMM Components

### Hidden States (Z)
The underlying activities: standing, walking, jumping, still

### Observations (X)
Feature vectors derived from sensor data windows:
- **Time-domain**: mean, std, variance, SMA, correlation, zero-crossing rate
- **Frequency-domain**: dominant frequency, spectral energy, FFT coefficients

### Transition Probabilities (A)
Learned probability of transitioning between activities

### Emission Probabilities (B)
Gaussian distributions modeling feature patterns for each activity

### Initial State Probabilities (π)
Probability of starting in each activity

## Algorithms Implemented

### Viterbi Algorithm
Finds the most likely sequence of activities given observations.

### Baum-Welch Algorithm
Expectation-Maximization (EM) algorithm for training HMM parameters.

### Forward-Backward Algorithm
Computes posterior state probabilities.

## Features Extracted

### Time Domain (per axis)
- Mean, Standard deviation, Variance
- Min, Max, Range
- Signal Magnitude Area (SMA)
- Correlation between axes
- Zero crossing rate
- RMS, Skewness, Kurtosis

### Frequency Domain (per axis)
- Dominant frequency
- Spectral energy
- FFT coefficients (first 5)
- Spectral entropy
- Mean frequency

## Output

### Models
- `hmm_activity_model.pkl`: Trained HMM model
- `hmm_activity_model_scaler.pkl`: Feature scaler

### Results
- `transition_matrix_*.png`: Heatmap of learned transitions
- `confusion_matrix_*.png`: Classification confusion matrix
- `activity_sequence_*.png`: Predicted vs true activity sequences
- `final_report.json`: Evaluation metrics

## Evaluation Metrics

For each activity:
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Precision**: Positive Predictive Value
- **F1 Score**: Harmonic mean of precision and recall

## Data Collection Guidelines

1. **Sampling Rate**: 100 Hz (10ms intervals)
2. **Phone Position**: Waist level in pocket
3. **Duration**: 10 seconds per activity
4. **Target**: ~50 samples total across activities
5. **Format**: CSV with timestamp column


## Group Member Data

| Member | Phone Model | Sampling Rate |
|--------|-------------|---------------|
| Mathias | OPPO CPH2641 | 20 Hz (10ms) |
| Daniel |IPHONE PRO 12 | 20Hz|

## Analysis Notes

### Easiest to Distinguish
- Still vs Walking (very different patterns)
- Jumping (high amplitude spikes)

### Hardest to Distinguish
- Standing vs Still (both minimal movement)
- Activity transitions

### Improvements Suggested
1. More training data from different participants
2. Additional sensors (magnetometer)
3. Hierarchical HMM for sub-activities
4. Deep learning hybrid approaches

## License

Educational project for HMM-based activity recognition.
