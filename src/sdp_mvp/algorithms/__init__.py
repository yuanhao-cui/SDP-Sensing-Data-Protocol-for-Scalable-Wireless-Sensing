"""Algorithm modules and registry exports."""

from __future__ import annotations

from sdp_mvp.denoise import fft_bandpass, hampel_filter, moving_average_denoise
from sdp_mvp.transforms import (
    conjugate_multiply,
    csi_ratio,
    delay_transform,
    doppler_spectrum,
    make_feature_tensor,
    phase_sanitize_linear,
    remove_static,
)

from .registry import (
    AlgorithmStep,
    execute_algorithm_steps,
    get_algorithm,
    list_algorithms,
    register_algorithm,
    unregister_algorithm,
)


def _register_builtin_algorithms() -> None:
    builtins = {
        ("denoise", "hampel"): hampel_filter,
        ("denoise", "moving_average"): moving_average_denoise,
        ("denoise", "fft_bandpass"): fft_bandpass,
        ("denoise", "bandpass"): fft_bandpass,
        ("phase", "linear_sanitize"): phase_sanitize_linear,
        ("calibrate", "linear"): phase_sanitize_linear,
        ("transform", "remove_static"): remove_static,
        ("transform", "conjugate_multiply"): conjugate_multiply,
        ("transform", "csi_ratio"): csi_ratio,
        ("transform", "delay"): delay_transform,
        ("transform", "doppler"): doppler_spectrum,
        ("feature", "tensor"): make_feature_tensor,
    }
    for (category, name), func in builtins.items():
        register_algorithm(category, name, func, replace=True)


_register_builtin_algorithms()

__all__ = [
    "AlgorithmStep",
    "execute_algorithm_steps",
    "get_algorithm",
    "list_algorithms",
    "register_algorithm",
    "unregister_algorithm",
    "hampel_filter",
    "moving_average_denoise",
    "fft_bandpass",
    "remove_static",
    "phase_sanitize_linear",
    "conjugate_multiply",
    "csi_ratio",
    "delay_transform",
    "doppler_spectrum",
    "make_feature_tensor",
]
