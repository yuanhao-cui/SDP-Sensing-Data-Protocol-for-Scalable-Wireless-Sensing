"""WSDP Model Library - Pluggable model registry and unified API."""

from .registry import MODEL_REGISTRY, register_model, get_model, list_models
from .csi_model import CSIModel
from .baselines import MLPModel, CNN1DModel, CNN2DModel, LSTMModel
from .mainstream import ResNet1D, ResNet2D, BiLSTMAttention, EfficientNetCSI
from .sota import VisionTransformerCSI, MambaCSI, GraphNeuralCSI

__all__ = [
    # Registry
    "MODEL_REGISTRY", "register_model", "get_model", "list_models", "create_model",
    # Baseline models
    "MLPModel", "CNN1DModel", "CNN2DModel", "LSTMModel",
    # Mainstream models
    "ResNet1D", "ResNet2D", "BiLSTMAttention", "EfficientNetCSI",
    # SOTA models
    "VisionTransformerCSI", "MambaCSI", "GraphNeuralCSI",
    # Original model (backward compatible)
    "CSIModel",
]


def create_model(name: str, num_classes: int, input_shape: tuple, **kwargs):
    """Create a model by name with unified interface.

    Args:
        name: Model name from registry (case-insensitive).
        num_classes: Number of output classes.
        input_shape: (T, F, A) tuple — time steps, frequency bins, antennas.
        **kwargs: Extra model-specific hyperparameters.

    Returns:
        nn.Module instance.
    """
    return get_model(name, num_classes=num_classes, input_shape=input_shape, **kwargs)
