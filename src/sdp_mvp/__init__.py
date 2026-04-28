"""MVP CSI denoising and modular signal processing package."""

from .algorithms import (
    AlgorithmStep,
    execute_algorithm_steps,
    get_algorithm,
    list_algorithms,
    register_algorithm,
    unregister_algorithm,
)
from .denoise import fft_bandpass, hampel_filter, moving_average_denoise
from .models import get_model, list_models, register_model, run_model, unregister_model
from .pipeline import PipelineConfig, SignalProcessingConfig, build_processor, process_csi_sample, run_pipeline
from .processors import ModelSpec, ModularProcessor, ensure_3d
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
    register_reader,
    unregister_reader,
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
    "PipelineConfig",
    "process_csi_sample",
    "run_pipeline",
    "build_processor",
    "ModularProcessor",
    "ModelSpec",
    "ensure_3d",
    "AlgorithmStep",
    "register_algorithm",
    "unregister_algorithm",
    "get_algorithm",
    "list_algorithms",
    "execute_algorithm_steps",
    "register_model",
    "unregister_model",
    "get_model",
    "list_models",
    "run_model",
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
    "register_reader",
    "unregister_reader",
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
