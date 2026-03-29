"""
Frequency grid interpolation for CSI data.

Interpolates CSI measurements across subcarriers to a target resolution.
Useful for aligning data from different hardware platforms or increasing
frequency resolution for fine-grained analysis.

Uses actual OFDM subcarrier frequency positions (not uniform 0..F-1) for
accurate frequency-domain interpolation. This is critical because OFDM
subcarriers are not uniformly spaced in the reported CSI — e.g., Intel
IWL5300 reports 30 subcarriers with a gap around the DC tone.

Reference:
    Bianchi V, et al. "Indoor Localization by Interpolation of
    Radio Maps." Sensors, 2020.
    IEEE 802.11n-2009, Table 7-25f (subcarrier grouping).
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import decimate as _scipy_decimate, resample_poly
from typing import Optional

from .subcarrier_mapping import get_subcarrier_indices


def interpolate_grid(csi, target_K=30, method='cubic',
                     subcarrier_indices: Optional[np.ndarray] = None,
                     dataset: Optional[str] = None):
    """
    Interpolate CSI data to a target number of subcarriers.

    Resamples each antenna's CSI profile from the original K subcarriers
    to target_K uniformly-spaced subcarriers. The interpolation uses the
    actual OFDM subcarrier frequency positions as the x-axis, producing
    correctly-spaced output.

    Args:
        csi: CSI array of shape (T, F, A) — complex or real.
        target_K: Target number of subcarriers (default: 30).
        method: Interpolation method ('linear', 'cubic', 'nearest').
        subcarrier_indices: 1D array of actual OFDM subcarrier indices,
            length F. Determines the x-positions for interpolation.
        dataset: Dataset name for automatic subcarrier index lookup.

    Returns:
        np.ndarray: Interpolated CSI with shape (T, target_K, A).

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

    # Resolve actual subcarrier positions for the x-axis
    if subcarrier_indices is not None:
        orig_freq = np.asarray(subcarrier_indices, dtype=np.float64)
    elif dataset is not None:
        orig_freq = get_subcarrier_indices(dataset=dataset, num_subcarriers=F)
    else:
        orig_freq = get_subcarrier_indices(num_subcarriers=F)

    if len(orig_freq) != F:
        raise ValueError(
            f"subcarrier_indices length ({len(orig_freq)}) != F ({F})"
        )

    # Target: uniformly spaced within the original frequency range
    target_freq = np.linspace(orig_freq[0], orig_freq[-1], target_K)

    result = np.zeros((T, target_K, A), dtype=csi.dtype)

    for t in range(T):
        for a in range(A):
            signal = csi[t, :, a]
            if np.iscomplexobj(signal):
                interp_real = interp1d(orig_freq, np.real(signal),
                                       kind=method, bounds_error=False,
                                       fill_value='extrapolate')
                interp_imag = interp1d(orig_freq, np.imag(signal),
                                       kind=method, bounds_error=False,
                                       fill_value='extrapolate')
                result[t, :, a] = (interp_real(target_freq)
                                   + 1j * interp_imag(target_freq))
            else:
                interp_func = interp1d(orig_freq, signal,
                                       kind=method, bounds_error=False,
                                       fill_value='extrapolate')
                result[t, :, a] = interp_func(target_freq)

    return result


def decimate_antialias(csi, target_K, axis=1):
    """
    Anti-aliased decimation of CSI along a specified axis.

    Applies a low-pass anti-aliasing filter before downsampling to prevent
    spectral aliasing artifacts. Uses ``scipy.signal.decimate`` with an
    FIR filter when the decimation factor is an integer; falls back to
    ``scipy.signal.resample_poly`` for non-integer factors.

    Typical use: reduce the number of subcarriers (axis=1) for
    dimensionality reduction while preserving the frequency-domain
    envelope shape.

    Args:
        csi: CSI array of shape (T, F, A) — complex or real.
            The size along ``axis`` must be > target_K.
        target_K: Target size along the decimation axis.
        axis: Axis to decimate (default: 1, subcarrier axis).

    Returns:
        np.ndarray: Decimated CSI. For default axis=1, shape is
            (T, target_K, A).

    Reference:
        Oppenheim AV, Schafer RW. "Discrete-Time Signal Processing."
        Prentice Hall, 3rd ed., 2009. (Multirate signal processing)
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")

    current_size = csi.shape[axis]
    if target_K < 1:
        raise ValueError(f"target_K must be >= 1, got {target_K}")
    if target_K >= current_size:
        raise ValueError(
            f"target_K ({target_K}) must be < current size along "
            f"axis {axis} ({current_size})"
        )

    def _decimate_signal(x, ax):
        """Decimate a real ndarray along axis ax."""
        if current_size % target_K == 0:
            q = current_size // target_K
            return _scipy_decimate(x, q, ftype='fir', axis=ax)
        else:
            # resample_poly(x, up, down) resamples by factor up/down
            from math import gcd
            g = gcd(target_K, current_size)
            up = target_K // g
            down = current_size // g
            return resample_poly(x, up, down, axis=ax)

    if np.iscomplexobj(csi):
        real_part = _decimate_signal(np.real(csi), axis)
        imag_part = _decimate_signal(np.imag(csi), axis)
        return real_part + 1j * imag_part
    else:
        return _decimate_signal(csi, axis)
