"""MVP CSI denoising and signal processing package."""

from .denoise import fft_bandpass, hampel_filter, moving_average_denoise
from .pipeline import SignalProcessingConfig, process_csi_sample
from .readers import (
    BaseReader,
    BfeeReader,
    ElderReader,
    XrfReader,
    ZTEReader,
    get_all_reader_metadata,
    get_reader_class,
    list_datasets,
    load_data,
)
from .structure import BfeeFrame, BaseFrame, CSIData
from .transforms import (
    conjugate_multiply,
    csi_ratio,
    delay_transform,
    doppler_spectrum,
    make_feature_tensor,
    phase_sanitize_linear,
    remove_static,
)

__all__ = [
    "SignalProcessingConfig",
    "process_csi_sample",
    "hampel_filter",
    "fft_bandpass",
    "moving_average_denoise",
    "BaseFrame",
    "BfeeFrame",
    "CSIData",
    "BaseReader",
    "BfeeReader",
    "ElderReader",
    "XrfReader",
    "ZTEReader",
    "get_reader_class",
    "list_datasets",
    "get_all_reader_metadata",
    "load_data",
    "remove_static",
    "phase_sanitize_linear",
    "conjugate_multiply",
    "csi_ratio",
    "delay_transform",
    "doppler_spectrum",
    "make_feature_tensor",
]
