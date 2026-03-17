"""Pluggable model registry for WSDP."""

from typing import Dict, Type, Optional
import torch.nn as nn

# Global registry: {lowercase_name: (category, model_class)}
MODEL_REGISTRY: Dict[str, tuple] = {}


def register_model(category: str, name: str, model_class: Type[nn.Module]):
    """Register a model class in the global registry.

    Args:
        category: Model category (baseline, mainstream, sota).
        name: Human-readable model name.
        model_class: nn.Module subclass.
    """
    key = name.lower()
    if key in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' already registered.")
    MODEL_REGISTRY[key] = (category.lower(), model_class)


def get_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: Model name (case-insensitive).
        **kwargs: Passed to model constructor (must include num_classes, input_shape).

    Returns:
        Instantiated nn.Module.

    Raises:
        KeyError: If model name not found.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    _, model_class = MODEL_REGISTRY[key]
    return model_class(**kwargs)


def list_models(category: Optional[str] = None) -> Dict[str, str]:
    """List all registered models, optionally filtered by category.

    Args:
        category: Filter by category name (baseline, mainstream, sota).

    Returns:
        Dict mapping model names to their categories.
    """
    result = {}
    for name, (cat, _) in MODEL_REGISTRY.items():
        if category is None or cat == category.lower():
            result[name] = cat
    return result
