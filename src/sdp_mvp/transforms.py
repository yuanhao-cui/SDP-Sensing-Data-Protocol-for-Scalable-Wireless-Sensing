"""CSI signal transformation and feature construction utilities."""

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


def remove_static(csi: np.ndarray, method: str = "mean") -> np.ndarray:
    """Remove static multipath/background along time.

    `mean` is fast and usually sufficient. `median` is more robust to short
    bursts but slightly slower.
    """

    arr, squeezed = _as_3d(csi)
    if method == "mean":
        baseline = arr.mean(axis=0, keepdims=True)
    elif method == "median":
        baseline = np.median(arr, axis=0, keepdims=True)
    else:
        raise ValueError("method must be 'mean' or 'median'")
    return _restore_shape(arr - baseline, squeezed)


def phase_sanitize_linear(
    csi: np.ndarray,
    subcarrier_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized linear phase sanitization across subcarriers.

    It removes per-frame/per-antenna linear phase trends caused by timing and
    carrier offsets while preserving amplitude. This replaces slow loops over
    `(time, antenna)` with closed-form least squares.
    """

    arr, squeezed = _as_3d(csi)
    if not np.iscomplexobj(arr):
        return np.array(csi, copy=True)

    _, f_dim, _ = arr.shape
    if subcarrier_indices is None:
        k = np.arange(f_dim, dtype=np.float64)
    else:
        k = np.asarray(subcarrier_indices, dtype=np.float64)
        if k.shape != (f_dim,):
            raise ValueError(f"subcarrier_indices must have shape ({f_dim},)")

    phase = np.unwrap(np.angle(arr), axis=1)
    k_centered = k - k.mean()
    denom = np.sum(k_centered * k_centered)
    if denom <= 0:
        return np.array(csi, copy=True)

    phase_mean = phase.mean(axis=1, keepdims=True)
    slope = np.sum((phase - phase_mean) * k_centered[None, :, None], axis=1) / denom
    intercept = phase_mean[:, 0, :] - slope * k.mean()
    trend = slope[:, None, :] * k[None, :, None] + intercept[:, None, :]
    corrected_phase = phase - trend
    corrected = np.abs(arr) * np.exp(1j * corrected_phase)
    return _restore_shape(corrected.astype(arr.dtype, copy=False), squeezed)


def conjugate_multiply(csi: np.ndarray, ref_antenna: int = 0, eps: float = 1e-10) -> np.ndarray:
    """Cancel common phase noise by multiplying with a reference conjugate.

    Returns `(T, F, A-1)` and excludes the reference antenna stream.
    """

    arr, _ = _as_3d(csi)
    if arr.shape[2] < 2:
        return arr.copy()
    if not 0 <= ref_antenna < arr.shape[2]:
        raise ValueError("ref_antenna out of range")

    ref = arr[:, :, ref_antenna]
    others = np.delete(arr, ref_antenna, axis=2)
    power = np.maximum(np.abs(ref) ** 2, eps)
    return others * np.conj(ref)[:, :, None] / power[:, :, None]


def csi_ratio(csi: np.ndarray, pairs: list[tuple[int, int]] | None = None, eps: float = 1e-10) -> np.ndarray:
    """Compute antenna-pair CSI ratios for phase-error-resistant features."""

    arr, _ = _as_3d(csi)
    a_dim = arr.shape[2]
    if a_dim < 2:
        return arr.copy()
    if pairs is None:
        pairs = [(i, i + 1) for i in range(a_dim - 1)]

    out = []
    for num, den in pairs:
        if not (0 <= num < a_dim and 0 <= den < a_dim):
            raise ValueError(f"antenna pair out of range: {(num, den)}")
        denom = np.where(np.abs(arr[:, :, den]) < eps, eps, arr[:, :, den])
        out.append(arr[:, :, num] / denom)
    return np.stack(out, axis=2)


def delay_transform(csi: np.ndarray, n_delay: int | None = None, window: bool = True) -> np.ndarray:
    """Transform frequency-domain CSI into a delay-like domain by IFFT."""

    arr, squeezed = _as_3d(csi)
    f_dim = arr.shape[1]
    if n_delay is None:
        n_delay = f_dim
    work = arr
    if window and f_dim > 1:
        win = np.hanning(f_dim).astype(arr.real.dtype, copy=False)
        work = work * win[None, :, None]
    delay = np.fft.ifft(work, n=n_delay, axis=1)
    return _restore_shape(delay.astype(np.complex64, copy=False), squeezed)


def doppler_spectrum(
    csi: np.ndarray,
    fs: float,
    n_fft: int = 64,
    hop: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a vectorized Doppler/STFT magnitude over the time axis.

    Returns `(spectrum, doppler_freqs)` where spectrum shape is
    `(num_windows, n_fft, F, A)`. Frequencies are FFT-shifted.
    """

    arr, squeezed = _as_3d(csi)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if n_fft < 2 or hop < 1:
        raise ValueError("n_fft must be >= 2 and hop must be >= 1")

    t_len = arr.shape[0]
    if t_len < n_fft:
        pad = n_fft - t_len
        work = np.pad(arr, ((0, pad), (0, 0), (0, 0)), mode="edge")
    else:
        work = arr

    starts = np.arange(0, work.shape[0] - n_fft + 1, hop)
    if starts.size == 0:
        starts = np.array([0])

    frames = np.stack([work[s : s + n_fft] for s in starts], axis=0)
    win = np.hanning(n_fft).astype(frames.real.dtype, copy=False)
    frames = frames * win[None, :, None, None]
    spec = np.fft.fftshift(np.fft.fft(frames, n=n_fft, axis=1), axes=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))
    mag = np.abs(spec).astype(np.float32, copy=False)
    if squeezed:
        mag = mag[:, :, :, 0]
    return mag, freqs


def _zscore_channels(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = x.std(axis=(1, 2, 3), keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def make_feature_tensor(
    csi: np.ndarray,
    channels: tuple[str, ...] = (
        "amp",
        "amp_delta",
        "phase_sin",
        "phase_cos",
        "phase_delta_sin",
        "phase_delta_cos",
    ),
    normalize: bool = True,
) -> np.ndarray:
    """Build a real-valued model-ready tensor `[C, T, F, A]`.

    The default channels preserve amplitude dynamics and wrapped phase without
    directly exposing discontinuous raw phase angles.
    """

    arr, _ = _as_3d(csi)
    amp = np.log1p(np.abs(arr))
    amp_delta = np.diff(amp, axis=0, prepend=amp[:1])
    phase = np.angle(arr) if np.iscomplexobj(arr) else np.zeros_like(amp)

    prev = np.roll(arr, shift=1, axis=0)
    phase_delta = np.angle(arr * np.conj(prev)) if np.iscomplexobj(arr) else np.zeros_like(amp)
    phase_delta[0] = 0.0

    lookup = {
        "amp": amp,
        "amp_delta": amp_delta,
        "phase_sin": np.sin(phase),
        "phase_cos": np.cos(phase),
        "phase_delta_sin": np.sin(phase_delta),
        "phase_delta_cos": np.cos(phase_delta),
        "real": arr.real,
        "imag": arr.imag if np.iscomplexobj(arr) else np.zeros_like(amp),
    }

    missing = [name for name in channels if name not in lookup]
    if missing:
        raise ValueError(f"unknown feature channels: {missing}")

    features = np.stack([lookup[name] for name in channels], axis=0).astype(np.float32, copy=False)
    if normalize:
        features = _zscore_channels(features).astype(np.float32, copy=False)
    return features
