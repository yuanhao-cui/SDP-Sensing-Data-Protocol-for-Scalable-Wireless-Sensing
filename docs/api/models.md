# API Reference - Models

## `create_model()`

Create any of the 19 built-in models using the unified factory function.

```python
from wsdp.models import create_model

model = create_model(model_name, num_classes, input_shape, **kwargs)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Name of the model (see table below) |
| `num_classes` | `int` | Number of output classes |
| `input_shape` | `tuple` | `(T, F, A)` - time steps, frequency bins, antenna count |
| `**kwargs` | | Model-specific hyperparameters |

**Returns:** `torch.nn.Module`

## All Available Models

### Baseline Models

| Name | Class | Description |
|------|-------|-------------|
| `MLPModel` | `MLPModel` | Fully-connected network |
| `CNN1DModel` | `CNN1DModel` | 1D convolution over time axis |
| `CNN2DModel` | `CNN2DModel` | 2D convolution on F×A per time step |
| `LSTMModel` | `LSTMModel` | LSTM over spatially-encoded features |

### Mainstream Models

| Name | Class | Description |
|------|-------|-------------|
| `ResNet1D` | `ResNet1D` | 1D residual network |
| `ResNet2D` | `ResNet2D` | 2D residual network |
| `BiLSTMAttention` | `BiLSTMAttention` | Bidirectional LSTM + attention |
| `EfficientNetCSI` | `EfficientNetCSI` | Efficient CNN with configurable width/depth |

### SOTA Models

| Name | Class | Description |
|------|-------|-------------|
| `VisionTransformerCSI` | `VisionTransformerCSI` | ViT for CSI patches |
| `MambaCSI` | `MambaCSI` | State space model |
| `GraphNeuralCSI` | `GraphNeuralCSI` | GNN on antenna/subcarrier topology |
| `CSIModel` | `CSIModel` | CNN + Transformer |
| `THAT` | `THAT` | Two-stream conv-augmented Transformer |
| `CSITime` | `CSITime` | Inception-Time variant for CSI |
| `PA_CSI` | `PA_CSI` | Phase-Amplitude dual-channel attention |

### Lightweight Models

| Name | Class | Description |
|------|-------|-------------|
| `WiFlexFormer` | `WiFlexFormer` | Efficient WiFi Transformer |
| `AttentionGRU` | `AttentionGRU` | Single GRU + temporal attention |

### Cross-Domain Models

| Name | Class | Description |
|------|-------|-------------|
| `EI` | `EI` | Gradient reversal domain adaptation |
| `FewSense` | `FewSense` | Prototypical few-shot learning |

## Examples

### Baseline Model

```python
model = create_model("MLPModel", num_classes=6, input_shape=(200, 30, 3))
```

### Mainstream Model

```python
model = create_model("ResNet1D", num_classes=6, input_shape=(200, 30, 3),
                      base_channels=64, num_blocks=4)
```

### SOTA Model

```python
# Vision Transformer
model = create_model("VisionTransformerCSI", num_classes=6, input_shape=(200, 30, 3),
                      embed_dim=256, num_layers=6, num_heads=8)

# Two-stream Transformer
model = create_model("THAT", num_classes=6, input_shape=(200, 30, 3))

# Phase-Amplitude attention
model = create_model("PA_CSI", num_classes=6, input_shape=(200, 30, 3))

# Inception-Time variant
model = create_model("CSITime", num_classes=6, input_shape=(200, 30, 3))
```

### Lightweight Model

```python
# For edge deployment (~62K params)
model = create_model("WiFlexFormer", num_classes=6, input_shape=(200, 30, 3))

# Ultra-lightweight (~52K params)
model = create_model("AttentionGRU", num_classes=6, input_shape=(200, 30, 3))
```

### Cross-Domain Model

```python
# Domain adaptation with gradient reversal
model = create_model("EI", num_classes=6, input_shape=(200, 30, 3),
                      num_domains=3)

# Few-shot prototypical network
model = create_model("FewSense", num_classes=6, input_shape=(200, 30, 3),
                      n_support=5)
```

## Model Utilities

```python
from wsdp.models import list_models, get_model, register_model

# List all available model names
all_models = list_models()                    # all 19 models
baselines = list_models("baseline")           # 4 models
mainstream = list_models("mainstream")        # 4 models
sota = list_models("sota")                    # 7 models
lightweight = list_models("lightweight")      # 2 models
cross_domain = list_models("cross_domain")    # 2 models

# get_model is an alias for create_model
model = get_model("MambaCSI", num_classes=6, input_shape=(200, 30, 3))

# Register a custom model
register_model("custom", "MyModel", MyModelClass)
```

See the [Model Selection Guide](../models.md) for recommendations on choosing the right model.
