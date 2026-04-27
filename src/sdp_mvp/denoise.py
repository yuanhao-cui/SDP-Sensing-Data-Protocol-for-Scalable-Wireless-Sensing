"""Vectorized denoising primitives for CSI tensors.

All public functions accept CSI tensors shaped as `(T, F, A)` where:
- T: time samples
- F: subcarriers
- A: antenna pairs / streams

2-D `(T, F)` tensors are accepted and returned as 2-D tensors.
"""

from __future__ import annotations

import numpy as np


def _as_3d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x)
    squeezed = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeezed = True
    if arr.ndim != 3:
        raise ValueError(f"expected CSI shape (T,F,A) or (T,F), got {arr.shape}")
    return arr, squeezed


def _restore_shape(x: np.ndarray, squeezed: bool) -> np.ndarray:
    return x[:, :, 0] if squeezed else x


def _apply_real_or_complex(x: np.ndarray, fn) -> np.ndarray:
    if np.iscomplexobj(x):
        return fn(x.real) + 1j * fn(x.imag)
    return fn(x)


def hampel_filter(csi: np.ndarray, window: int = 3, n_sigmas: float = 3.0) -> np.ndarray:
    """Remove impulse noise with a vectorized Hampel filter along time.

    The implementation uses rolling median and MAD over the time axis and
    replaces outliers with the local median. Complex tensors are filtered by
    applying the same rule to real and imaginary parts independently.
    """

    arr, squeezed = _as_3d(csi)
    if window < 1:
        return np.array(csi, copy=True)
    if n_sigmas <= 0:
        raise ValueError("n_sigmas must be positive")

    width = 2 * window + 1
    eps = np.finfo(np.float64).eps

    def _hampel_real(x: np.ndarray) -> np.ndarray:
        work = x.astype(np.float64, copy=False)
        padded = np.pad(work, ((window, window), (0, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
        med = np.median(windows, axis=-1)
        mad = np.median(np.abs(windows - med[..., None]), axis=-1)
        threshold = n_sigmas * np.maximum(1.4826 * mad, eps)
        return np.where(np.abs(work - med) > threshold, med, work).astype(x.dtype, copy=False)

    filtered = _apply_real_or_complex(arr, _hampel_real)
    return _restore_shape(filtered.astype(arr.dtype, copy=False), squeezed)


def moving_average_denoise(csi: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth CSI with an edge-padded moving average along time.

    This is intentionally simple and dependency-free. It is useful after a
    bandpass/highpass step when small residual jitter remains.
    """

    arr, squeezed = _as_3d(csi)
    if window <= 1:
        return np.array(csi, copy=True)

    left = window // 2
    right = window - left - 1

    def _smooth_real(x: np.ndarray) -> np.ndarray:
        padded = np.pad(x, ((left, right), (0, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
        return windows.mean(axis=-1).astype(x.dtype, copy=False)

    smoothed = _apply_real_or_complex(arr, _smooth_real)
    return _restore_shape(smoothed.astype(arr.dtype, copy=False), squeezed)


def fft_bandpass(
    csi: np.ndarray,
    fs: float,
    low_hz: float | None = 0.3,
    high_hz: float | None = 12.0,
    keep_dc: bool = False,
) -> np.ndarray:
    """Apply an FFT-domain bandpass filter along time.

    Compared with per-subcarrier IIR loops, this vectorizes over all
    subcarriers and antenna streams. For CSI sensing, it also naturally removes
    static clutter when `low_hz > 0`.
    """

    arr, squeezed = _as_3d(csi)
    if fs <= 0:
        raise ValueError("fs must be positive")

    t_len = arr.shape[0]
    if t_len < 2:
        return np.array(csi, copy=True)

    nyquist = fs / 2.0
    low = 0.0 if low_hz is None else max(0.0, float(low_hz))
    high = nyquist if high_hz is None else min(float(high_hz), nyquist)
    if low > high:
        raise ValueError(f"low_hz ({low}) must be <= high_hz ({high})")

    freqs = np.fft.fftfreq(t_len, d=1.0 / fs)
    abs_freqs = np.abs(freqs)
    mask = (abs_freqs >= low) & (abs_freqs <= high)
    if keep_dc:
        mask |= abs_freqs < 1e-12

    spectrum = np.fft.fft(arr, axis=0)
    filtered = np.fft.ifft(spectrum * mask[:, None, None], axis=0)
    if not np.iscomplexobj(arr):
        filtered = filtered.real
    return _restore_shape(filtered.astype(arr.dtype, copy=False), squeezed)
