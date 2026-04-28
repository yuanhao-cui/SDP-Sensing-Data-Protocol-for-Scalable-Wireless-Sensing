"""Composable CSI processors built from registered algorithm steps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .algorithms import AlgorithmStep, execute_algorithm_steps
from .models import get_model, run_model


def ensure_3d(csi: np.ndarray) -> np.ndarray:
    """Return CSI as ``(T,F,A)`` or raise for incompatible shapes."""

    arr = np.asarray(csi)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected CSI shape (T,F,A) or (T,F), got {arr.shape}")
    return arr


@dataclass
class ModelSpec:
    """Optional model stage attached after signal processing."""

    model: str | Any
    params: Mapping[str, Any] = field(default_factory=dict)
    input_key: str = "features"
    output_key: str = "model_output"


class ModularProcessor:
    """Run registered algorithm steps and an optional pluggable model."""

    def __init__(
        self,
        steps: Sequence[AlgorithmStep | Mapping[str, Any] | Sequence[Any]],
        *,
        model: ModelSpec | str | Any | None = None,
    ) -> None:
        self.steps = [AlgorithmStep.from_config(step) for step in steps]
        self.model_spec = self._normalize_model_spec(model)

    @staticmethod
    def _normalize_model_spec(model: ModelSpec | str | Any | None) -> ModelSpec | None:
        if model is None:
            return None
        if isinstance(model, ModelSpec):
            return model
        if isinstance(model, str):
            return ModelSpec(model=model)
        return ModelSpec(model=model)

    def process(self, csi: np.ndarray, *, initial_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        state = execute_algorithm_steps(ensure_3d(csi), self.steps, initial_state=initial_state)
        if self.model_spec is not None:
            spec = self.model_spec
            if spec.input_key not in state:
                raise KeyError(f"model input_key not found: {spec.input_key}")
            model = get_model(spec.model, **dict(spec.params)) if isinstance(spec.model, str) else spec.model
            state[spec.output_key] = run_model(model, state[spec.input_key])
        return state
