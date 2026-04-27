"""Minimal optimized CSI processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .denoise import fft_bandpass, hampel_filter, moving_average_denoise
from .transforms import (
    conjugate_multiply,
    delay_transform,
    doppler_spectrum,
    make_feature_tensor,
    phase_sanitize_linear,
    remove_static,
)


@dataclass
class SignalProcessingConfig:
    """Configuration for the MVP CSI processing flow."""

    fs: float = 100.0
    band: tuple[float | None, float | None] = (0.3, 12.0)
    hampel_window: int = 3
    hampel_sigmas: float = 3.0
    smooth_window: int = 3
    static_method: str = "mean"
    subcarrier_indices: np.ndarray | None = None
    use_conjugate_multiply: bool = True
    ref_antenna: int = 0
    feature_channels: tuple[str, ...] = (
        "amp",
        "amp_delta",
        "phase_sin",
        "phase_cos",
        "phase_delta_sin",
        "phase_delta_cos",
    )
    emit_delay: bool = False
    delay_bins: int | None = None
    emit_doppler: bool = False
    doppler_n_fft: int = 64
    doppler_hop: int = 16


def _ensure_3d(csi: np.ndarray) -> np.ndarray:
    arr = np.asarray(csi)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected CSI shape (T,F,A) or (T,F), got {arr.shape}")
    return arr


def process_csi_sample(csi: np.ndarray, config: SignalProcessingConfig | None = None) -> dict[str, np.ndarray]:
    """Run the MVP optimized CSI processing flow for one sample.

    Input shape: `(T, F, A)` complex CSI.
    Output keys:
    - `cleaned`: complex dynamic CSI after denoising and transforms
    - `features`: real model-ready tensor `[C, T, F, A']`
    - optional `delay` and `doppler` representations
    """

    cfg = config or SignalProcessingConfig()
    x = _ensure_3d(csi)

    # 1) Remove impulsive outliers without per-subcarrier Python loops.
    x = hampel_filter(x, window=cfg.hampel_window, n_sigmas=cfg.hampel_sigmas)

    # 2) Sanitize phase before deriving phase-aware channels.
    x = phase_sanitize_linear(x, subcarrier_indices=cfg.subcarrier_indices)

    # 3) Remove static multipath and isolate human motion frequencies.
    x = remove_static(x, method=cfg.static_method)
    low_hz, high_hz = cfg.band
    x = fft_bandpass(x, fs=cfg.fs, low_hz=low_hz, high_hz=high_hz, keep_dc=False)

    # 4) Small residual smoothing after bandpass.
    if cfg.smooth_window > 1:
        x = moving_average_denoise(x, window=cfg.smooth_window)

    # 5) Optional common phase cancellation across antenna streams.
    if cfg.use_conjugate_multiply and x.shape[2] >= 2:
        x = conjugate_multiply(x, ref_antenna=cfg.ref_antenna)

    features = make_feature_tensor(x, channels=cfg.feature_channels, normalize=True)
    result: dict[str, np.ndarray] = {
        "cleaned": x.astype(np.complex64 if np.iscomplexobj(x) else np.float32, copy=False),
        "features": features,
    }

    if cfg.emit_delay:
        result["delay"] = delay_transform(x, n_delay=cfg.delay_bins)
    if cfg.emit_doppler:
        spec, freqs = doppler_spectrum(x, fs=cfg.fs, n_fft=cfg.doppler_n_fft, hop=cfg.doppler_hop)
        result["doppler"] = spec
        result["doppler_freqs"] = freqs.astype(np.float32, copy=False)

    return result
