"""
Amplitude processing algorithms for CSI data.

Provides normalization and outlier removal for CSI amplitude values.
"""
import numpy as np


def normalize_amplitude(csi, method='z-score'):
    """
    Normalize CSI amplitude along the time axis.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        method: Normalization method
            - 'z-score': Zero mean, unit variance per subcarrier
            - 'min-max': Scale to [0, 1] per subcarrier

    Returns:
        np.ndarray: Normalized CSI with same shape and dtype

    Reference:
        Standard statistical normalization. For CSI-specific usage:
        Ma Y, et al. "PhaseFi: Phase Fingerprinting for Indoor Localization
        with a Deep Learning Approach." IEEE GLOBECOM, 2015.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    valid_methods = ('z-score', 'min-max')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")

    result = np.empty_like(csi)
    amplitude = np.abs(csi)

    # Normalize along time axis (axis=0) per subcarrier
    if method == 'z-score':
        mean = np.mean(amplitude, axis=0, keepdims=True)
        std = np.std(amplitude, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # avoid division by zero
        norm_amp = (amplitude - mean) / std
    else:  # min-max
        amin = np.min(amplitude, axis=0, keepdims=True)
        amax = np.max(amplitude, axis=0, keepdims=True)
        range_val = amax - amin
        range_val = np.where(range_val < 1e-10, 1.0, range_val)
        norm_amp = (amplitude - amin) / range_val

    if np.iscomplexobj(csi):
        phase = np.angle(csi)
        result = norm_amp * np.exp(1j * phase)
    else:
        result = norm_amp

    return result


def remove_outliers(csi, method='iqr', factor=1.5):
    """
    Remove or clip outlier amplitudes in CSI data.

    Detects outliers per subcarrier along the time axis and clips them
    to the boundary values. This preserves data shape while mitigating
    the effect of anomalous measurements.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        method: Outlier detection method
            - 'iqr': Interquartile Range method
            - 'z-score': Standard deviation based method
        factor: Multiplier for the detection threshold (default: 1.5)
            For IQR: outliers are outside Q1 - factor*IQR to Q3 + factor*IQR
            For z-score: outliers are beyond mean ± factor*std

    Returns:
        np.ndarray: CSI with outliers clipped, same shape and dtype

    Reference:
        Tukey JW. "Exploratory Data Analysis." Addison-Wesley, 1977.
        (IQR method definition)
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    valid_methods = ('iqr', 'z-score')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")

    result = csi.copy()
    amplitude = np.abs(csi)

    if method == 'iqr':
        q1 = np.percentile(amplitude, 25, axis=0, keepdims=True)
        q3 = np.percentile(amplitude, 75, axis=0, keepdims=True)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
    else:  # z-score
        mean = np.mean(amplitude, axis=0, keepdims=True)
        std = np.std(amplitude, axis=0, keepdims=True)
        lower = mean - factor * std
        upper = mean + factor * std

    # Clip amplitudes to [lower, upper]
    clipped_amp = np.clip(amplitude, lower, upper)

    if np.iscomplexobj(csi):
        phase = np.angle(csi)
        result = clipped_amp * np.exp(1j * phase)
    else:
        result = clipped_amp

    return result
