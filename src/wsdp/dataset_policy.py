"""Dataset-specific preprocessing policy helpers."""

import numpy as np


AMPLITUDE_PRIMARY_DATASETS = {"xrf55"}
PHASE_AMPLITUDE_DATASETS = {"widar", "gait"}


def is_amplitude_primary_dataset(dataset: str) -> bool:
    """Return True for datasets whose downstream features use amplitude values."""
    return dataset in AMPLITUDE_PRIMARY_DATASETS


def uses_phase_amplitude(dataset: str) -> bool:
    """Return True when model inputs should contain both amplitude and phase."""
    return str(dataset).strip().lower() in PHASE_AMPLITUDE_DATASETS


def pipeline_uses_zscore(pipeline_steps) -> bool:
    """Return True when a pipeline config contains z-score normalization."""
    return (
        isinstance(pipeline_steps, dict)
        and pipeline_steps.get("normalize", {}).get("method") == "z-score"
    )


def real_if_negligible_imaginary(csi, dataset: str, atol: float = 1e-10):
    """Drop a near-zero imaginary part for amplitude-primary datasets only."""
    if not is_amplitude_primary_dataset(dataset) or not np.iscomplexobj(csi):
        return csi

    imag_part = np.imag(csi)
    if imag_part.size == 0 or np.max(np.abs(imag_part)) < atol:
        return np.real(csi)
    return csi


def convert_to_model_input(csi, dataset: str, pipeline_steps=None, output_dtype=None):
    """Convert a sample to the real representation consumed by the model.

    ``output_dtype`` is used by compact shared storage. External callers leave
    it unset to preserve normalization-before-tensor-conversion numerics.
    """
    data = np.asarray(csi)
    phase_dataset = uses_phase_amplitude(dataset)
    phase_zscore = phase_dataset and pipeline_uses_zscore(pipeline_steps)

    if phase_zscore:
        if np.iscomplexobj(data):
            raise RuntimeError(
                "Widar/Gait z-score input must be converted to real "
                "amplitude-phase channels by ConfigurableProcessor"
            )
        result = data
    elif phase_dataset and np.iscomplexobj(data):
        antenna_count = data.shape[-1]
        if output_dtype is None:
            # Preserve the historical amplitude/phase computation order.
            result = np.concatenate([np.abs(data), np.angle(data)], axis=-1)
        else:
            result = np.empty(
                (*data.shape[:-1], antenna_count * 2), dtype=output_dtype
            )
            np.abs(data, out=result[..., :antenna_count])
            np.arctan2(
                data.imag, data.real, out=result[..., antenna_count:]
            )
    elif is_amplitude_primary_dataset(dataset) and not np.iscomplexobj(data):
        # XRF55 train-split normalization can contain signed real values.
        result = data
    else:
        result = np.abs(data)

    if output_dtype is not None:
        return result.astype(output_dtype, copy=False)
    return result
