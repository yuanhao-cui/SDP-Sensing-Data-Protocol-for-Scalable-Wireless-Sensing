"""Pluggable algorithm registry for CSI processing steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Algorithm = Callable[..., Any]

_ALGORITHM_REGISTRY: dict[str, dict[str, Algorithm]] = {}


def register_algorithm(category: str, name: str, func: Algorithm, *, replace: bool = False) -> None:
    """Register one algorithm implementation under ``category/name``."""

    if not callable(func):
        raise TypeError("func must be callable")
    bucket = _ALGORITHM_REGISTRY.setdefault(category, {})
    if name in bucket and not replace:
        raise ValueError(f"algorithm already registered: {category}/{name}")
    bucket[name] = func


def unregister_algorithm(category: str, name: str) -> bool:
    """Remove one registered algorithm, returning whether it existed."""

    bucket = _ALGORITHM_REGISTRY.get(category)
    if not bucket or name not in bucket:
        return False
    del bucket[name]
    if not bucket:
        del _ALGORITHM_REGISTRY[category]
    return True


def get_algorithm(category: str, name: str) -> Algorithm:
    """Return one registered algorithm implementation."""

    try:
        return _ALGORITHM_REGISTRY[category][name]
    except KeyError as exc:
        available = list_algorithms(category) if category in _ALGORITHM_REGISTRY else {}
        raise KeyError(f"unknown algorithm {category}/{name}; available: {list(available)}") from exc


def list_algorithms(category: str | None = None) -> dict[str, Any]:
    """List registered algorithms, optionally for one category."""

    if category is not None:
        return dict(_ALGORITHM_REGISTRY.get(category, {}))
    return {cat: sorted(methods) for cat, methods in sorted(_ALGORITHM_REGISTRY.items())}


@dataclass(frozen=True)
class AlgorithmStep:
    """One configurable algorithm step in a modular pipeline."""

    category: str
    method: str
    params: Mapping[str, Any] = field(default_factory=dict)
    input_key: str = "csi"
    output_key: str | Sequence[str] = "csi"
    enabled: bool = True

    @classmethod
    def from_config(cls, config: "AlgorithmStep | Mapping[str, Any] | Sequence[Any]") -> "AlgorithmStep":
        """Create a step from a dataclass, mapping, or tuple-like config."""

        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            params = dict(config.get("params", {}))
            for key, value in config.items():
                if key not in {"category", "method", "params", "input_key", "output_key", "enabled"}:
                    params[key] = value
            return cls(
                category=str(config["category"]),
                method=str(config["method"]),
                params=params,
                input_key=str(config.get("input_key", "csi")),
                output_key=config.get("output_key", "csi"),
                enabled=bool(config.get("enabled", True)),
            )
        if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            if len(config) < 2:
                raise ValueError("step tuple must contain at least category and method")
            params = config[2] if len(config) > 2 else {}
            return cls(str(config[0]), str(config[1]), dict(params))
        raise TypeError(f"unsupported step config: {type(config)!r}")


def _assign_output(state: dict[str, Any], output_key: str | Sequence[str], value: Any) -> None:
    if isinstance(output_key, str):
        state[output_key] = value
        return
    keys = list(output_key)
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"step returned a single value but output_key expects {keys}")
    if len(keys) != len(value):
        raise ValueError(f"output_key count {len(keys)} does not match result count {len(value)}")
    for key, item in zip(keys, value):
        state[str(key)] = item


def execute_algorithm_steps(
    csi: np.ndarray,
    steps: Sequence[AlgorithmStep | Mapping[str, Any] | Sequence[Any]],
    *,
    initial_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute configurable algorithm steps and return the full state dict."""

    state: dict[str, Any] = dict(initial_state or {})
    state.setdefault("csi", csi)
    for raw_step in steps:
        step = AlgorithmStep.from_config(raw_step)
        if not step.enabled:
            continue
        if step.input_key not in state:
            raise KeyError(f"step {step.category}/{step.method} missing input_key: {step.input_key}")
        func = get_algorithm(step.category, step.method)
        result = func(state[step.input_key], **dict(step.params))
        _assign_output(state, step.output_key, result)
    return state
