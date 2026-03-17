# WSDP Model Selection Guide

## Overview

WSDP provides 12 built-in models organized into three tiers, from lightweight baselines to state-of-the-art architectures. All models share a unified interface and are accessible through the pluggable registry.

## Quick Start

```python
from wsdp.models import create_model

# Simplest usage
model = create_model("ResNet1D", num_classes=10, input_shape=(20, 30, 3))

# With custom parameters
model = create_model("VisionTransformerCSI", num_classes=10, input_shape=(20, 30, 3),
                      embed_dim=256, num_layers=6)
```

## Model Categories

### Baseline Models
Simple architectures for establishing performance baselines and sanity checking.

| Model | Description | Best For | Params (default) |
|-------|-------------|----------|------------------|
| **MLPModel** | Fully-connected network, flattens input | Quick baseline, debugging | ~1-5M |
| **CNN1DModel** | 1D convolution over time axis | Temporal patterns | ~0.5-2M |
| **CNN2DModel** | 2D convolution on F×A per time step | Spatial-spectral patterns | ~0.1-0.5M |
| **LSTMModel** | LSTM over spatially-encoded features | Sequential dependencies | ~0.5-1M |

### Mainstream Models
Well-established architectures with proven track records.

| Model | Description | Best For | Params (default) |
|-------|-------------|----------|------------------|
| **ResNet1D** | 1D residual network with 3 blocks | Deep temporal features | ~1-5M |
| **ResNet2D** | 2D residual network | Spatial feature extraction | ~0.5-2M |
| **BiLSTMAttention** | Bidirectional LSTM + multi-head attention | Complex temporal dynamics | ~1-3M |
| **EfficientNetCSI** | Efficient CNN with configurable width/depth | Resource-constrained deployment | ~0.2-5M |

### SOTA Models
State-of-the-art architectures for maximum accuracy.

| Model | Description | Best For | Params (default) |
|-------|-------------|----------|------------------|
| **VisionTransformerCSI** | ViT treating F×A patches across time | Large-scale pretraining | ~2-10M |
| **MambaCSI** | State space model (Mamba) for temporal modeling | Long sequences, linear complexity | ~1-5M |
| **GraphNeuralCSI** | GNN on antenna/subcarrier topology | Physical structure modeling | ~0.5-2M |
| **CSIModel** | CNN + Transformer (original WSDP model) | General-purpose | ~1-2M |

## Choosing a Model

### By Dataset Size

| Dataset Size | Recommended Models |
|-------------|-------------------|
| Small (<1K samples) | MLPModel, CNN2DModel, LSTMModel |
| Medium (1K-10K) | ResNet1D, BiLSTMAttention, CSIModel |
| Large (>10K) | VisionTransformerCSI, MambaCSI, EfficientNetCSI |

### By Computational Budget

| Budget | Recommended Models |
|--------|-------------------|
| Low (CPU / small GPU) | MLPModel, CNN1DModel, CNN2DModel, LSTMModel |
| Medium (single GPU) | ResNet1D, ResNet2D, BiLSTMAttention, GraphNeuralCSI |
| High (multi-GPU) | VisionTransformerCSI, MambaCSI, EfficientNetCSI |

### By Task Characteristics

| Task Type | Recommended Models |
|-----------|-------------------|
| Gesture recognition | VisionTransformerCSI, CSIModel, ResNet2D |
| Gait analysis | BiLSTMAttention, MambaCSI, LSTMModel |
| Activity detection | ResNet1D, EfficientNetCSI, CNN1DModel |
| Fall detection | CNN2DModel, ResNet1D, MLPModel |

## Input Format

All models expect CSI tensors in the format `(B, T, F, A)`:
- **B**: Batch size
- **T**: Time steps (e.g., 20-100)
- **F**: Frequency bins (e.g., 30 for canonical grid)
- **A**: Antenna count (e.g., 3)

Both **complex** (`torch.complex64/128`) and **real** (`torch.float32`) inputs are supported. Complex inputs are automatically converted to real by stacking real and imaginary parts.

## Model Registration

All models are registered in a central registry:

```python
from wsdp.models import list_models, get_model

# List all models
all_models = list_models()

# Filter by category
baselines = list_models("baseline")
sota_models = list_models("sota")

# Get a model by name
model = get_model("MambaCSI", num_classes=10, input_shape=(20, 30, 3))
```

## Custom Model Registration

Register your own models to use with the WSDP pipeline:

```python
from wsdp.models import register_model
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, input_shape, **kwargs):
        super().__init__()
        T, F, A = input_shape
        self.fc = nn.Linear(T * F * A * 2, num_classes)
    
    def forward(self, x):
        # x: (B, T, F, A) complex or real
        return self.fc(x.reshape(x.shape[0], -1))

# Register it
register_model("custom", "MyModel", MyModel)

# Now usable via the standard API
from wsdp.models import create_model
model = create_model("MyModel", num_classes=10, input_shape=(20, 30, 3))
```

## Performance Tips

1. **Start with baselines**: Always establish a baseline with MLPModel or CNN1DModel before trying complex architectures.

2. **Match model to data**: 
   - Short sequences → CNN-based models
   - Long sequences → LSTM/Mamba models
   - Rich spatial structure → ViT or GNN models

3. **Hyperparameter tuning**: Most models expose key hyperparameters:
   - `base_channels`: Controls model width
   - `num_layers`/`num_blocks`: Controls depth
   - `hidden_size`/`embed_dim`: Controls representation capacity

4. **EfficientNetCSI**: Use `width_mult` and `depth_mult` < 1.0 for smaller models, > 1.0 for larger.

5. **VisionTransformerCSI**: Larger `patch_size` = fewer patches = faster but less detailed.
