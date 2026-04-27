"""Synthetic checks for the MVP denoising and signal processing algorithms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdp_mvp import (  # noqa: E402
    SignalProcessingConfig,
    fft_bandpass,
    hampel_filter,
    phase_sanitize_linear,
    process_csi_sample,
)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b) ** 2))


def validate_hampel() -> dict[str, float]:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 128, endpoint=False)
    clean = np.sin(2 * np.pi * 3 * t)[:, None, None]
    noisy = clean + 0.03 * rng.standard_normal(clean.shape)
    noisy[::17] += 4.0
    filtered = hampel_filter(noisy, window=3, n_sigmas=3.0)
    return {
        "mse_before": mse(noisy, clean),
        "mse_after": mse(filtered, clean),
    }


def validate_bandpass() -> dict[str, float]:
    fs = 100.0
    t = np.arange(256) / fs
    clean = np.sin(2 * np.pi * 2.0 * t)[:, None, None]
    noisy = clean + 0.7 * np.sin(2 * np.pi * 30.0 * t)[:, None, None]
    filtered = fft_bandpass(noisy, fs=fs, low_hz=0.5, high_hz=5.0)
    return {
        "mse_before": mse(noisy, clean),
        "mse_after": mse(filtered, clean),
    }


def validate_phase_sanitization() -> dict[str, float]:
    rng = np.random.default_rng(11)
    t_dim, f_dim, a_dim = 64, 30, 3
    k = np.linspace(-28, 28, f_dim)
    base_phase = rng.uniform(-0.5, 0.5, size=(t_dim, 1, a_dim))
    linear_error = 0.18 * k[None, :, None] + base_phase
    true_csi = np.exp(1j * rng.normal(scale=0.03, size=(t_dim, f_dim, a_dim)))
    observed = true_csi * np.exp(1j * linear_error)
    corrected = phase_sanitize_linear(observed, subcarrier_indices=k)

    def mean_abs_slope(x: np.ndarray) -> float:
        phase = np.unwrap(np.angle(x), axis=1)
        k_centered = k - k.mean()
        denom = np.sum(k_centered * k_centered)
        slope = np.sum((phase - phase.mean(axis=1, keepdims=True)) * k_centered[None, :, None], axis=1) / denom
        return float(np.mean(np.abs(slope)))

    return {
        "slope_before": mean_abs_slope(observed),
        "slope_after": mean_abs_slope(corrected),
    }


def validate_pipeline() -> dict[str, object]:
    rng = np.random.default_rng(23)
    fs = 100.0
    t_dim, f_dim, a_dim = 256, 30, 4
    t = np.arange(t_dim) / fs
    k = np.linspace(-28, 28, f_dim)

    static = 2.0 * np.exp(1j * 0.05 * k[None, :, None])
    motion = 0.25 * np.sin(2 * np.pi * 2.0 * t)[:, None, None]
    antenna_phase = np.linspace(0.0, 1.0, a_dim)[None, None, :]
    csi = (static + motion * np.exp(1j * antenna_phase)).astype(np.complex128)
    csi += 0.05 * (rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape))
    csi[40, :, :] += 3.0

    cfg = SignalProcessingConfig(
        fs=fs,
        band=(0.5, 6.0),
        subcarrier_indices=k,
        use_conjugate_multiply=True,
        emit_delay=True,
        emit_doppler=True,
        doppler_n_fft=64,
        doppler_hop=32,
    )
    out = process_csi_sample(csi, cfg)
    features = out["features"]
    return {
        "cleaned_shape": tuple(out["cleaned"].shape),
        "features_shape": tuple(features.shape),
        "delay_shape": tuple(out["delay"].shape),
        "doppler_shape": tuple(out["doppler"].shape),
        "features_finite": bool(np.isfinite(features).all()),
    }


def main() -> None:
    results = {
        "hampel": validate_hampel(),
        "bandpass": validate_bandpass(),
        "phase": validate_phase_sanitization(),
        "pipeline": validate_pipeline(),
    }

    assert results["hampel"]["mse_after"] < results["hampel"]["mse_before"]
    assert results["bandpass"]["mse_after"] < results["bandpass"]["mse_before"]
    assert results["phase"]["slope_after"] < 0.1 * results["phase"]["slope_before"]
    assert results["pipeline"]["features_finite"] is True

    for section, values in results.items():
        print(f"[{section}]")
        for key, value in values.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
