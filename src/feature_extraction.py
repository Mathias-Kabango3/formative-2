"""
Feature Extraction Module for HMM Activity Recognition.

This module extracts time-domain and frequency-domain features from
sensor data windows for use as HMM observations.

Features extracted:
Time-domain:
    - Mean, variance, standard deviation
    - Signal Magnitude Area (SMA)
    - Correlation between axes
    - Min, max, range
    - Zero crossing rate
    - Peak-to-peak amplitude

Frequency-domain:
    - Dominant frequency
    - Spectral energy
    - FFT coefficients
    - Spectral entropy
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import List, Dict, Tuple, Optional
import warnings

from config import SAMPLING_RATE_HZ


def compute_mean(data: np.ndarray) -> np.ndarray:
    """Compute mean for each axis."""
    return np.mean(data, axis=0)


def compute_std(data: np.ndarray) -> np.ndarray:
    """Compute standard deviation for each axis."""
    return np.std(data, axis=0)


def compute_variance(data: np.ndarray) -> np.ndarray:
    """Compute variance for each axis."""
    return np.var(data, axis=0)


def compute_min(data: np.ndarray) -> np.ndarray:
    """Compute minimum for each axis."""
    return np.min(data, axis=0)


def compute_max(data: np.ndarray) -> np.ndarray:
    """Compute maximum for each axis."""
    return np.max(data, axis=0)


def compute_range(data: np.ndarray) -> np.ndarray:
    """Compute range (max - min) for each axis."""
    return compute_max(data) - compute_min(data)


def compute_sma(acc_data: np.ndarray) -> float:
    """
    Compute Signal Magnitude Area (SMA).
    SMA = (1/n) * sum(|ax| + |ay| + |az|)
    
    Args:
        acc_data: Accelerometer data with shape (n_samples, 3)
    
    Returns:
        SMA value
    """
    return np.mean(np.sum(np.abs(acc_data), axis=1))


def compute_magnitude(data: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of 3D signal.
    magnitude = sqrt(x^2 + y^2 + z^2)
    
    Args:
        data: Sensor data with shape (n_samples, 3)
    
    Returns:
        Magnitude array with shape (n_samples,)
    """
    return np.sqrt(np.sum(data ** 2, axis=1))


def compute_correlation(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation coefficients between axes.
    
    Args:
        data: Sensor data with shape (n_samples, 3) for x, y, z
    
    Returns:
        Array with correlation values [corr_xy, corr_xz, corr_yz]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = np.corrcoef(data.T)
        
    # Handle NaN values (when variance is zero)
    if np.isnan(corr_matrix).any():
        corr_matrix = np.nan_to_num(corr_matrix, 0.0)
    
    # Extract upper triangle (excluding diagonal)
    corr_xy = corr_matrix[0, 1]
    corr_xz = corr_matrix[0, 2]
    corr_yz = corr_matrix[1, 2]
    
    return np.array([corr_xy, corr_xz, corr_yz])


def compute_zero_crossing_rate(data: np.ndarray) -> np.ndarray:
    """
    Compute zero crossing rate for each axis.
    
    Args:
        data: Sensor data with shape (n_samples, n_axes)
    
    Returns:
        Zero crossing rate for each axis
    """
    # Subtract mean to center the signal
    centered = data - np.mean(data, axis=0)
    signs = np.sign(centered)
    zero_crossings = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=0)
    return zero_crossings / (len(data) - 1)


def compute_peak_to_peak(data: np.ndarray) -> np.ndarray:
    """
    Compute peak-to-peak amplitude for each axis.
    """
    return np.ptp(data, axis=0)


def compute_rms(data: np.ndarray) -> np.ndarray:
    """
    Compute Root Mean Square for each axis.
    """
    return np.sqrt(np.mean(data ** 2, axis=0))


def compute_skewness(data: np.ndarray) -> np.ndarray:
    """
    Compute skewness for each axis.
    """
    return stats.skew(data, axis=0)


def compute_kurtosis(data: np.ndarray) -> np.ndarray:
    """
    Compute kurtosis for each axis.
    """
    return stats.kurtosis(data, axis=0)


# --------------------- Frequency Domain Features ---------------------

def compute_fft_features(
    data: np.ndarray,
    sampling_rate: float = SAMPLING_RATE_HZ,
    n_coefficients: int = 5
) -> Dict[str, np.ndarray]:
    """
    Compute FFT-based frequency domain features.
    
    Args:
        data: Sensor data with shape (n_samples, n_axes)
        sampling_rate: Sampling rate in Hz
        n_coefficients: Number of FFT coefficients to return
    
    Returns:
        Dictionary with frequency features
    """
    n_samples, n_axes = data.shape
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(n_samples)
    
    fft_features = {
        'dominant_freq': np.zeros(n_axes),
        'spectral_energy': np.zeros(n_axes),
        'fft_coefficients': np.zeros((n_axes, n_coefficients)),
        'spectral_entropy': np.zeros(n_axes),
        'mean_frequency': np.zeros(n_axes)
    }
    
    freqs = fftfreq(n_samples, 1/sampling_rate)
    positive_freqs = freqs[:n_samples//2]
    
    for axis in range(n_axes):
        # Apply window and compute FFT
        windowed_signal = data[:, axis] * window
        fft_vals = fft(windowed_signal)
        fft_magnitude = np.abs(fft_vals[:n_samples//2])
        
        # Power spectrum
        power_spectrum = fft_magnitude ** 2
        power_spectrum_normalized = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        
        # Dominant frequency
        dominant_idx = np.argmax(fft_magnitude)
        fft_features['dominant_freq'][axis] = abs(positive_freqs[dominant_idx]) if len(positive_freqs) > 0 else 0
        
        # Spectral energy
        fft_features['spectral_energy'][axis] = np.sum(power_spectrum) / n_samples
        
        # FFT coefficients (first n_coefficients, excluding DC)
        start_idx = 1  # Skip DC component
        end_idx = min(start_idx + n_coefficients, len(fft_magnitude))
        coeffs = fft_magnitude[start_idx:end_idx]
        fft_features['fft_coefficients'][axis, :len(coeffs)] = coeffs
        
        # Spectral entropy
        entropy = -np.sum(power_spectrum_normalized * np.log2(power_spectrum_normalized + 1e-10))
        fft_features['spectral_entropy'][axis] = entropy
        
        # Mean frequency (centroid)
        mean_freq = np.sum(positive_freqs * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10)
        fft_features['mean_frequency'][axis] = mean_freq
    
    return fft_features


def compute_spectral_energy_bands(
    data: np.ndarray,
    sampling_rate: float = SAMPLING_RATE_HZ,
    bands: List[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute energy in different frequency bands.
    
    Args:
        data: Sensor data with shape (n_samples, n_axes)
        sampling_rate: Sampling rate in Hz
        bands: List of (low_freq, high_freq) tuples
    
    Returns:
        Energy in each band for each axis
    """
    if bands is None:
        # Default bands: low (0-2Hz), mid (2-5Hz), high (5-15Hz)
        bands = [(0, 2), (2, 5), (5, 15)]
    
    n_samples, n_axes = data.shape
    band_energies = np.zeros((n_axes, len(bands)))
    
    freqs = fftfreq(n_samples, 1/sampling_rate)
    
    for axis in range(n_axes):
        fft_vals = fft(data[:, axis])
        power = np.abs(fft_vals) ** 2
        
        for band_idx, (low, high) in enumerate(bands):
            mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
            band_energies[axis, band_idx] = np.sum(power[mask]) / n_samples
    
    return band_energies


# --------------------- Main Feature Extraction ---------------------

def extract_features_from_window(
    window_data: np.ndarray,
    sampling_rate: float = SAMPLING_RATE_HZ
) -> np.ndarray:
    """
    Extract all features from a single window of sensor data.
    
    Args:
        window_data: Sensor data with shape (n_samples, 6)
                    Columns: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        sampling_rate: Sampling rate in Hz
    
    Returns:
        1D feature vector
    """
    # Split accelerometer and gyroscope data
    acc_data = window_data[:, :3]  # acc_x, acc_y, acc_z
    gyro_data = window_data[:, 3:]  # gyro_x, gyro_y, gyro_z
    
    features = []
    
    # ===== Time Domain Features =====
    
    # Accelerometer features
    features.extend(compute_mean(acc_data))  # 3 features
    features.extend(compute_std(acc_data))   # 3 features
    features.extend(compute_variance(acc_data))  # 3 features
    features.extend(compute_min(acc_data))   # 3 features
    features.extend(compute_max(acc_data))   # 3 features
    features.extend(compute_range(acc_data))  # 3 features
    features.append(compute_sma(acc_data))   # 1 feature
    features.extend(compute_correlation(acc_data))  # 3 features
    features.extend(compute_zero_crossing_rate(acc_data))  # 3 features
    features.extend(compute_rms(acc_data))   # 3 features
    features.extend(compute_skewness(acc_data))  # 3 features
    features.extend(compute_kurtosis(acc_data))  # 3 features
    
    # Accelerometer magnitude features
    acc_magnitude = compute_magnitude(acc_data)
    features.append(np.mean(acc_magnitude))
    features.append(np.std(acc_magnitude))
    features.append(np.max(acc_magnitude))
    
    # Gyroscope features
    features.extend(compute_mean(gyro_data))  # 3 features
    features.extend(compute_std(gyro_data))   # 3 features
    features.extend(compute_variance(gyro_data))  # 3 features
    features.extend(compute_min(gyro_data))   # 3 features
    features.extend(compute_max(gyro_data))   # 3 features
    features.extend(compute_range(gyro_data))  # 3 features
    features.append(compute_sma(gyro_data))   # 1 feature (treating as SMA)
    features.extend(compute_correlation(gyro_data))  # 3 features
    features.extend(compute_zero_crossing_rate(gyro_data))  # 3 features
    features.extend(compute_rms(gyro_data))   # 3 features
    
    # Gyroscope magnitude features
    gyro_magnitude = compute_magnitude(gyro_data)
    features.append(np.mean(gyro_magnitude))
    features.append(np.std(gyro_magnitude))
    
    # ===== Frequency Domain Features =====
    
    # Accelerometer FFT features
    acc_fft = compute_fft_features(acc_data, sampling_rate, n_coefficients=5)
    features.extend(acc_fft['dominant_freq'])  # 3 features
    features.extend(acc_fft['spectral_energy'])  # 3 features
    features.extend(acc_fft['fft_coefficients'].flatten())  # 15 features
    features.extend(acc_fft['spectral_entropy'])  # 3 features
    features.extend(acc_fft['mean_frequency'])  # 3 features
    
    # Gyroscope FFT features
    gyro_fft = compute_fft_features(gyro_data, sampling_rate, n_coefficients=5)
    features.extend(gyro_fft['dominant_freq'])  # 3 features
    features.extend(gyro_fft['spectral_energy'])  # 3 features
    features.extend(gyro_fft['fft_coefficients'].flatten())  # 15 features
    features.extend(gyro_fft['spectral_entropy'])  # 3 features
    features.extend(gyro_fft['mean_frequency'])  # 3 features
    
    # Spectral energy bands (accelerometer)
    acc_bands = compute_spectral_energy_bands(acc_data, sampling_rate)
    features.extend(acc_bands.flatten())  # 9 features (3 axes × 3 bands)
    
    # Convert to numpy array and handle any NaN/Inf values
    feature_vector = np.array(features, dtype=np.float64)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector


def extract_features_from_windows(
    windows: List[np.ndarray],
    sampling_rate: float = SAMPLING_RATE_HZ
) -> np.ndarray:
    """
    Extract features from multiple windows.
    
    Args:
        windows: List of window data arrays, each with shape (n_samples, 6)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Feature matrix with shape (n_windows, n_features)
    """
    feature_list = []
    
    for window in windows:
        features = extract_features_from_window(window, sampling_rate)
        feature_list.append(features)
    
    return np.array(feature_list)


def get_feature_names() -> List[str]:
    """
    Get the names of all extracted features.
    
    Returns:
        List of feature names
    """
    names = []
    sensors = ['acc', 'gyro']
    axes = ['x', 'y', 'z']
    
    # Time domain features for accelerometer
    for stat in ['mean', 'std', 'var', 'min', 'max', 'range']:
        for axis in axes:
            names.append(f'acc_{stat}_{axis}')
    
    names.append('acc_sma')
    names.extend(['acc_corr_xy', 'acc_corr_xz', 'acc_corr_yz'])
    
    for axis in axes:
        names.append(f'acc_zcr_{axis}')
    
    for axis in axes:
        names.append(f'acc_rms_{axis}')
    
    for axis in axes:
        names.append(f'acc_skew_{axis}')
    
    for axis in axes:
        names.append(f'acc_kurt_{axis}')
    
    names.extend(['acc_mag_mean', 'acc_mag_std', 'acc_mag_max'])
    
    # Time domain features for gyroscope
    for stat in ['mean', 'std', 'var', 'min', 'max', 'range']:
        for axis in axes:
            names.append(f'gyro_{stat}_{axis}')
    
    names.append('gyro_sma')
    names.extend(['gyro_corr_xy', 'gyro_corr_xz', 'gyro_corr_yz'])
    
    for axis in axes:
        names.append(f'gyro_zcr_{axis}')
    
    for axis in axes:
        names.append(f'gyro_rms_{axis}')
    
    names.extend(['gyro_mag_mean', 'gyro_mag_std'])
    
    # Frequency domain features for accelerometer
    for axis in axes:
        names.append(f'acc_dom_freq_{axis}')
    for axis in axes:
        names.append(f'acc_spec_energy_{axis}')
    for axis in axes:
        for i in range(5):
            names.append(f'acc_fft_{i+1}_{axis}')
    for axis in axes:
        names.append(f'acc_spec_entropy_{axis}')
    for axis in axes:
        names.append(f'acc_mean_freq_{axis}')
    
    # Frequency domain features for gyroscope
    for axis in axes:
        names.append(f'gyro_dom_freq_{axis}')
    for axis in axes:
        names.append(f'gyro_spec_energy_{axis}')
    for axis in axes:
        for i in range(5):
            names.append(f'gyro_fft_{i+1}_{axis}')
    for axis in axes:
        names.append(f'gyro_spec_entropy_{axis}')
    for axis in axes:
        names.append(f'gyro_mean_freq_{axis}')
    
    # Spectral energy bands for accelerometer
    bands = ['low', 'mid', 'high']
    for axis in axes:
        for band in bands:
            names.append(f'acc_band_{band}_{axis}')
    
    return names


if __name__ == "__main__":
    # Test feature extraction
    print("=" * 50)
    print("Testing Feature Extraction Module")
    print("=" * 50)
    
    # Generate sample window data
    np.random.seed(42)
    n_samples = 50  # 500ms at 100Hz
    sample_window = np.random.randn(n_samples, 6) * 0.5
    
    # Add realistic patterns
    sample_window[:, 2] = -9.8 + np.random.randn(n_samples) * 0.1  # acc_z (gravity)
    
    print(f"\nInput window shape: {sample_window.shape}")
    
    # Extract features
    features = extract_features_from_window(sample_window)
    print(f"Output feature vector shape: {features.shape}")
    print(f"Number of features: {len(features)}")
    
    # Get feature names
    feature_names = get_feature_names()
    print(f"Number of feature names: {len(feature_names)}")
    
    # Print sample features
    print("\nSample features:")
    for i in range(min(10, len(features))):
        if i < len(feature_names):
            print(f"  {feature_names[i]}: {features[i]:.4f}")
    
    # Test with multiple windows
    print("\n" + "=" * 50)
    print("Testing batch feature extraction")
    windows = [np.random.randn(50, 6) for _ in range(10)]
    feature_matrix = extract_features_from_windows(windows)
    print(f"Feature matrix shape: {feature_matrix.shape}")
