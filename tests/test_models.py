"""Comprehensive tests for all WSDP models.

Tests cover: initialization, forward pass, output shape, complex/real input,
different input sizes, and backward compatibility.
"""

import pytest
import torch
import torch.nn as nn

from wsdp.models import (
    create_model, list_models, get_model,
    MLPModel, CNN1DModel, CNN2DModel, LSTMModel,
    ResNet1D, ResNet2D, BiLSTMAttention, EfficientNetCSI,
    VisionTransformerCSI, MambaCSI, GraphNeuralCSI,
    CSIModel,
)


# Default test shapes
DEFAULT_SHAPE = (20, 30, 3)  # T=20, F=30, A=3
SMALL_SHAPE = (10, 16, 2)
LARGE_SHAPE = (50, 64, 4)
NUM_CLASSES = 10
BATCH_SIZE = 4

# All model names
ALL_MODELS = [
    "MLPModel", "CNN1DModel", "CNN2DModel", "LSTMModel",
    "ResNet1D", "ResNet2D", "BiLSTMAttention", "EfficientNetCSI",
    "VisionTransformerCSI", "MambaCSI", "GraphNeuralCSI", "CSIModel",
]

BASELINE_MODELS = ["MLPModel", "CNN1DModel", "CNN2DModel", "LSTMModel"]
MAINSTREAM_MODELS = ["ResNet1D", "ResNet2D", "BiLSTMAttention", "EfficientNetCSI"]
SOTA_MODELS = ["VisionTransformerCSI", "MambaCSI", "GraphNeuralCSI", "CSIModel"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_complex_input(shape, batch=BATCH_SIZE):
    """Create random complex CSI tensor."""
    T, F, A = shape
    real = torch.randn(batch, T, F, A)
    imag = torch.randn(batch, T, F, A)
    return torch.complex(real, imag)


def _make_real_input(shape, batch=BATCH_SIZE):
    """Create random real CSI tensor."""
    T, F, A = shape
    return torch.randn(batch, T, F, A)


# ===========================================================================
# Section 1: Registry Tests
# ===========================================================================
class TestRegistry:
    """Test the model registry system."""

    def test_list_models_returns_all(self):
        models = list_models()
        for name in ALL_MODELS:
            assert name.lower() in models, f"Missing model: {name}"

    def test_list_models_by_category(self):
        baselines = list_models("baseline")
        assert "mlpmodel" in baselines
        assert "cnn1dmodel" in baselines
        mainstream = list_models("mainstream")
        assert "resnet1d" in mainstream
        sota = list_models("sota")
        assert "csimodel" in sota

    def test_get_model_case_insensitive(self):
        model = get_model("mlpmodel", num_classes=5, input_shape=DEFAULT_SHAPE)
        assert isinstance(model, MLPModel)
        model2 = get_model("MLPModel", num_classes=5, input_shape=DEFAULT_SHAPE)
        assert isinstance(model2, MLPModel)

    def test_get_model_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("NonExistentModel", num_classes=5, input_shape=DEFAULT_SHAPE)

    def test_create_model_api(self):
        model = create_model("cnn1dmodel", num_classes=NUM_CLASSES, input_shape=DEFAULT_SHAPE)
        assert isinstance(model, CNN1DModel)

    def test_register_duplicate_raises(self):
        from wsdp.models.registry import register_model
        with pytest.raises(ValueError, match="already registered"):
            register_model("test", "MLPModel", nn.Module)


# ===========================================================================
# Section 2: Initialization Tests
# ===========================================================================
class TestInitialization:
    """Test that all models can be initialized without errors."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_init_default(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=DEFAULT_SHAPE)
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_init_small_shape(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=SMALL_SHAPE)
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_init_large_shape(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=LARGE_SHAPE)
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_init_different_classes(self, model_name):
        model = get_model(model_name, num_classes=5, input_shape=DEFAULT_SHAPE)
        assert isinstance(model, nn.Module)
        model = get_model(model_name, num_classes=100, input_shape=DEFAULT_SHAPE)
        assert isinstance(model, nn.Module)


# ===========================================================================
# Section 3: Forward Pass & Output Shape Tests
# ===========================================================================
class TestForwardPass:
    """Test forward pass with complex and real inputs."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_forward_complex(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=DEFAULT_SHAPE)
        model.eval()
        x = _make_complex_input(DEFAULT_SHAPE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), \
            f"{model_name}: expected ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_forward_real(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=DEFAULT_SHAPE)
        model.eval()
        x = _make_real_input(DEFAULT_SHAPE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), \
            f"{model_name}: expected ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_forward_small_shape(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=SMALL_SHAPE)
        model.eval()
        x = _make_complex_input(SMALL_SHAPE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_forward_large_shape(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=LARGE_SHAPE)
        model.eval()
        x = _make_complex_input(LARGE_SHAPE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_forward_single_sample(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=DEFAULT_SHAPE)
        model.eval()
        x = _make_complex_input(DEFAULT_SHAPE, batch=1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES)


# ===========================================================================
# Section 4: Backward Compatibility Tests
# ===========================================================================
class TestBackwardCompatibility:
    """Test CSIModel backward compatibility (original API still works)."""

    def test_csi_model_original_api(self):
        """Original CSIModel(num_classes=10) without input_shape should still work."""
        model = CSIModel(num_classes=10, base_channels=32, latent_dim=128)
        x = torch.randn(4, 20, 30, 3)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)

    def test_csi_model_with_input_shape(self):
        """New API with input_shape parameter."""
        model = CSIModel(num_classes=10, input_shape=DEFAULT_SHAPE)
        x = torch.randn(4, 20, 30, 3)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)

    def test_csi_model_registered(self):
        """CSIModel should be in registry under sota category."""
        models = list_models("sota")
        assert "csimodel" in models


# ===========================================================================
# Section 5: Gradient Flow Tests
# ===========================================================================
class TestGradientFlow:
    """Test that gradients flow properly during backprop."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_gradient_flow(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=DEFAULT_SHAPE)
        model.train()
        x = _make_complex_input(DEFAULT_SHAPE)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, f"{model_name}: no gradients flowing"


# ===========================================================================
# Section 6: Model-specific Tests
# ===========================================================================
class TestModelSpecific:
    """Model-specific behavior tests."""

    def test_mlp_with_hidden_dims(self):
        model = MLPModel(num_classes=5, input_shape=DEFAULT_SHAPE, hidden_dims=[256, 128, 64])
        x = _make_complex_input(DEFAULT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_resnet1d_base_channels(self):
        model = ResNet1D(num_classes=5, input_shape=DEFAULT_SHAPE, base_channels=32)
        x = _make_complex_input(DEFAULT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_bilstm_hidden_size(self):
        model = BiLSTMAttention(num_classes=5, input_shape=DEFAULT_SHAPE, hidden_size=64)
        x = _make_complex_input(DEFAULT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_vit_patch_size(self):
        model = VisionTransformerCSI(num_classes=5, input_shape=DEFAULT_SHAPE,
                                      patch_size_f=5, patch_size_a=1)
        x = _make_complex_input(DEFAULT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_mamba_num_layers(self):
        model = MambaCSI(num_classes=5, input_shape=DEFAULT_SHAPE, num_layers=2)
        x = _make_complex_input(DEFAULT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_efficientnet_width_depth(self):
        model = EfficientNetCSI(num_classes=5, input_shape=DEFAULT_SHAPE,
                                 width_mult=0.5, depth_mult=0.5)
        x = _make_complex_input(DEFAULT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 5)


# ===========================================================================
# Section 7: Parameter Count Sanity
# ===========================================================================
class TestParameterCount:
    """Verify models have reasonable parameter counts."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_parameter_count_positive(self, model_name):
        model = get_model(model_name, num_classes=NUM_CLASSES, input_shape=DEFAULT_SHAPE)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, f"{model_name} has 0 parameters"
        assert n_params < 100_000_000, f"{model_name} has too many params: {n_params:,}"
