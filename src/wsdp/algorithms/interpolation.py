"""
Frequency grid interpolation for CSI data.

Interpolates CSI measurements across subcarriers to a target resolution.
Useful for aligning data from different hardware platforms or increasing
frequency resolution for fine-grained analysis.

Reference:
    Bianchi V, et al. "Indoor Localization by Interpolation of 
    Radio Maps." Sensors, 2020.
"""
import numpy as np
from scipy.interpolate import interp1d


def interpolate_grid(csi, target_K=30, method='cubic'):
    """
    Interpolate CSI data to a target number of subcarriers.

    Resamples each antenna's CSI profile from the original K subcarriers
    to target_K subcarriers using the specified interpolation method.

    Args:
        csi: CSI array of shape (T, F, A) — complex or real
        target_K: Target number of subcarriers (default: 30)
        method: Interpolation method
            - 'linear': Piecewise linear interpolation
            - 'cubic': Cubic spline interpolation
            - 'nearest': Nearest-neighbor interpolation

    Returns:
        np.ndarray: Interpolated CSI with shape (T, target_K, A)

    Reference:
        Standard interpolation methods applied to frequency domain.
        For cubic spline: de Boor C. "A Practical Guide to Splines."
        Springer, 1978.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")
    if target_K < 2:
        raise ValueError(f"target_K must be >= 2, got {target_K}")

    valid_methods = ('linear', 'cubic', 'nearest')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")

    T, F, A = csi.shape

    if F == target_K:
        return csi.copy()

    # Original subcarrier indices mapped to [0, 1]
    orig_indices = np.linspace(0, 1, F)
    target_indices = np.linspace(0, 1, target_K)

    result = np.zeros((T, target_K, A), dtype=csi.dtype)

    for t in range(T):
        for a in range(A):
            signal = csi[t, :, a]
            if np.iscomplexobj(signal):
                interp_real = interp1d(orig_indices, np.real(signal),
                                       kind=method, bounds_error=False, fill_value='extrapolate')
                interp_imag = interp1d(orig_indices, np.imag(signal),
                                       kind=method, bounds_error=False, fill_value='extrapolate')
                result[t, :, a] = interp_real(target_indices) + 1j * interp_imag(target_indices)
            else:
                interp_func = interp1d(orig_indices, signal,
                                       kind=method, bounds_error=False, fill_value='extrapolate')
                result[t, :, a] = interp_func(target_indices)

    return result
