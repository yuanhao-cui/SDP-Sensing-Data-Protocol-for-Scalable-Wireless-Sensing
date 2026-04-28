"""Modular CSI processing pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import AlgorithmStep
from .processors import ModelSpec, ModularProcessor, ensure_3d
from .readers import BaseReader, load_data
from .structure import CSIData


@dataclass
class SignalProcessingConfig:
    """Configuration for the default MVP CSI processing flow."""

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


@dataclass
class PipelineConfig:
    """End-to-end configurable pipeline options.

    ``steps`` controls signal-processing modules. When omitted, the optimized
    MVP defaults are derived from ``signal``. ``model`` is optional and may be a
    registered model name, a callable/model object, or a ``ModelSpec``.
    """

    dataset: str | None = None
    reader: str | type[BaseReader] | BaseReader | None = None
    reader_workers: int | None = 16
    signal: SignalProcessingConfig = field(default_factory=SignalProcessingConfig)
    steps: Sequence[AlgorithmStep | Mapping[str, Any] | Sequence[Any]] | None = None
    model: ModelSpec | str | Any | None = None


def _default_steps(config: SignalProcessingConfig) -> list[AlgorithmStep]:
    """Build the default optimized flow as pluggable algorithm steps."""

    low_hz, high_hz = config.band
    steps: list[AlgorithmStep] = [
        AlgorithmStep(
            "denoise",
            "hampel",
            {"window": config.hampel_window, "n_sigmas": config.hampel_sigmas},
        ),
        AlgorithmStep(
            "phase",
            "linear_sanitize",
            {"subcarrier_indices": config.subcarrier_indices},
        ),
        AlgorithmStep("transform", "remove_static", {"method": config.static_method}),
        AlgorithmStep(
            "denoise",
            "fft_bandpass",
            {"fs": config.fs, "low_hz": low_hz, "high_hz": high_hz, "keep_dc": False},
        ),
    ]

    if config.smooth_window > 1:
        steps.append(AlgorithmStep("denoise", "moving_average", {"window": config.smooth_window}))
    if config.use_conjugate_multiply:
        steps.append(
            AlgorithmStep(
                "transform",
                "conjugate_multiply",
                {"ref_antenna": config.ref_antenna},
                enabled=True,
            )
        )

    steps.append(
        AlgorithmStep(
            "feature",
            "tensor",
            {"channels": config.feature_channels, "normalize": True},
            input_key="csi",
            output_key="features",
        )
    )
    if config.emit_delay:
        steps.append(
            AlgorithmStep(
                "transform",
                "delay",
                {"n_delay": config.delay_bins},
                input_key="csi",
                output_key="delay",
            )
        )
    if config.emit_doppler:
        steps.append(
            AlgorithmStep(
                "transform",
                "doppler",
                {"fs": config.fs, "n_fft": config.doppler_n_fft, "hop": config.doppler_hop},
                input_key="csi",
                output_key=("doppler", "doppler_freqs"),
            )
        )
    return steps


def build_processor(
    config: SignalProcessingConfig | PipelineConfig | None = None,
    *,
    steps: Sequence[AlgorithmStep | Mapping[str, Any] | Sequence[Any]] | None = None,
    model: ModelSpec | str | Any | None = None,
) -> ModularProcessor:
    """Create a processor from default config or explicit algorithm steps."""

    if isinstance(config, PipelineConfig):
        pipeline_config = config
        selected_steps = steps if steps is not None else pipeline_config.steps
        selected_model = model if model is not None else pipeline_config.model
        signal_config = pipeline_config.signal
    else:
        selected_steps = steps
        selected_model = model
        signal_config = config or SignalProcessingConfig()

    return ModularProcessor(selected_steps or _default_steps(signal_config), model=selected_model)


def process_csi_sample(
    csi: np.ndarray,
    config: SignalProcessingConfig | PipelineConfig | None = None,
    *,
    steps: Sequence[AlgorithmStep | Mapping[str, Any] | Sequence[Any]] | None = None,
    model: ModelSpec | str | Any | None = None,
) -> dict[str, np.ndarray]:
    """Run one CSI sample through a modular processing pipeline.

    Defaults preserve the MVP optimized flow. Pass ``steps`` to replace the
    algorithm chain, and pass ``model`` to append a pluggable model stage.
    """

    processor = build_processor(config, steps=steps, model=model)
    state = processor.process(ensure_3d(csi))
    if "cleaned" not in state:
        x = state.get("csi")
        state["cleaned"] = x.astype(np.complex64 if np.iscomplexobj(x) else np.float32, copy=False)
    if "doppler_freqs" in state:
        state["doppler_freqs"] = state["doppler_freqs"].astype(np.float32, copy=False)
    return state


def _read_input(
    data: str | Path | CSIData | np.ndarray,
    config: PipelineConfig,
) -> list[tuple[str | None, np.ndarray]]:
    if isinstance(data, np.ndarray):
        return [(None, data)]
    if isinstance(data, CSIData):
        return [(data.file_name, data.to_numpy())]
    if isinstance(data, (str, Path)):
        reader = config.reader or config.dataset
        if reader is None:
            raise ValueError("dataset or reader is required when input is a path")
        records = load_data(str(data), reader, max_workers=config.reader_workers)
        return [(record.file_name, record.to_numpy()) for record in records]
    raise TypeError(f"unsupported pipeline input: {type(data)!r}")


def run_pipeline(
    data: str | Path | CSIData | np.ndarray,
    config: PipelineConfig | None = None,
    *,
    dataset: str | None = None,
    steps: Sequence[AlgorithmStep | Mapping[str, Any] | Sequence[Any]] | None = None,
    model: ModelSpec | str | Any | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run the main pipeline on a tensor, one ``CSIData``, file, or folder.

    Returns a single state dict for tensor/``CSIData`` input and a list of state
    dicts for path input. Each state includes ``features`` by default and can
    include ``model_output`` when a model is configured.
    """

    cfg = config or PipelineConfig(dataset=dataset)
    if dataset is not None:
        cfg.dataset = dataset
    if steps is not None:
        cfg.steps = steps
    if model is not None:
        cfg.model = model

    records = _read_input(data, cfg)
    processor = build_processor(cfg)
    outputs: list[dict[str, Any]] = []
    for file_name, csi in records:
        state = processor.process(csi, initial_state={"file_name": file_name} if file_name else None)
        x = state.get("csi")
        state.setdefault("cleaned", x.astype(np.complex64 if np.iscomplexobj(x) else np.float32, copy=False))
        outputs.append(state)
    return outputs[0] if isinstance(data, (np.ndarray, CSIData)) else outputs
