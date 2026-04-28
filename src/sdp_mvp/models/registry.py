"""Lightweight pluggable model registry.

The MVP package stays framework-agnostic: registered models can be plain
callables, objects with ``predict``/``transform``, or factories returning those.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ModelFactory = Callable[..., Any]
_MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory, *, replace: bool = False) -> None:
    """Register a model factory by name."""

    key = name.lower()
    if not callable(factory):
        raise TypeError("factory must be callable")
    if key in _MODEL_REGISTRY and not replace:
        raise ValueError(f"model already registered: {name}")
    _MODEL_REGISTRY[key] = factory


def unregister_model(name: str) -> bool:
    """Remove a model registration."""

    return _MODEL_REGISTRY.pop(name.lower(), None) is not None


def get_model(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered model."""

    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"unknown model {name}; available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[key](**kwargs)


def list_models() -> list[str]:
    """List registered model names."""

    return sorted(_MODEL_REGISTRY)


def run_model(model: Any, data: Any) -> Any:
    """Run a model using the common callable/predict/transform conventions."""

    if hasattr(model, "predict"):
        return model.predict(data)
    if hasattr(model, "transform"):
        return model.transform(data)
    if callable(model):
        return model(data)
    raise TypeError("model must be callable or expose predict()/transform()")


class IdentityModel:
    """Default pass-through model useful for tests and feature extraction."""

    def predict(self, data: Any) -> Any:
        return data


def _register_builtin_models() -> None:
    register_model("identity", lambda **_: IdentityModel(), replace=True)


_register_builtin_models()
