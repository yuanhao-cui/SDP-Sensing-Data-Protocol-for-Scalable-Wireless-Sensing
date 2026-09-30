# WSDP Model Selection Guide

## Overview

WSDP provides 19 built-in models, from lightweight baselines to state-of-the-art and cross-domain architectures. All models share a unified interface and are accessible through the pluggable registry.

The registry organizes models into **three categories**:

- **baseline** — simple architectures for sanity checking and performance floors
- **mainstream** — well-established architectures with proven track records
- **sota** — advanced architectures, including lightweight and cross-domain designs

> **Note**: "Lightweight" and "cross-domain" are functional labels, not registry
> categories. `list_models()` only accepts `baseline`, `mainstream`, or `sota`
> as the category filter — see [Model Registration](#model-registration).

## Quick Start

```python
from wsdp.models import create_model

# Simplest usage
model = create_model("ResNet1D", num_classes=10, input_shape=(20, 30, 3))

# With custom parameters
model = create_model("VisionTransformerCSI", num_classes=10, input_shape=(20, 30, 3),
                     embed_dim=256, num_layers=6)
```

## Model Catalog

Parameter counts below were measured with `num_classes=10, input_shape=(20, 30, 3)`. They scale with the input shape and class count, so treat them as relative size indicators rather than exact figures.

### Baseline Models

Simple architectures for establishing performance baselines and sanity checking.

| Model | Description | Best For | Params |
|-------|-------------|----------|--------|
| **MLPModel** | Fully-connected network on spatially-encoded features | Quick baseline, debugging | ~665K |
| **CNN1DModel** | 1D convolution over the time axis | Temporal patterns | ~237K |
| **CNN2DModel** | 2D convolution on F×A per time step | Spatial-spectral patterns | ~100K |
| **LSTMModel** | LSTM over spatially-encoded features | Sequential dependencies | ~729K |

### Mainstream Models

Well-established architectures with proven track records.

| Model | Description | Best For | Params |
|-------|-------------|----------|--------|
| **ResNet1D** | 1D residual network with 3 blocks | Deep temporal features | ~520K |
| **ResNet2D** | 2D residual network | Spatial feature extraction | ~308K |
| **BiLSTMAttention** | Bidirectional LSTM + multi-head attention | Complex temporal dynamics | ~980K |
| **EfficientNetCSI** | Efficient CNN with configurable width/depth | Resource-constrained deployment | ~897K |

### SOTA Models

State-of-the-art architectures for maximum accuracy, plus the original WSDP model.

| Model | Description | Best For | Params |
|-------|-------------|----------|--------|
| **VisionTransformerCSI** | ViT treating F×A patches across time | Large-scale pretraining | ~820K |
| **MambaCSI** | State space model for temporal modeling | Long sequences, linear complexity | ~1.2M |
| **GraphNeuralCSI** | GNN on antenna/subcarrier topology | Physical structure modeling | ~35K |
| **CSIModel** | CNN + Transformer (original WSDP model) | General-purpose | ~270K |
| **THAT** | Two-stream conv-augmented Transformer | Gesture/activity recognition | ~286K |
| **CSITime** | Inception-Time variant for CSI | General activity recognition | ~81K |
| **PA_CSI** | Phase-Amplitude dual-channel attention | Phase-sensitive tasks | ~292K |

### Lightweight Models *(registered as `sota`)*

Compact architectures designed for edge and resource-constrained deployment.

| Model | Description | Best For | Params |
|-------|-------------|----------|--------|
| **WiFlexFormer** | Efficient WiFi Transformer | Edge deployment | ~58K |
| **AttentionGRU** | Single GRU + temporal attention | Ultra-lightweight | ~52K |

### Cross-Domain Models *(registered as `sota`)*

Architectures with built-in domain adaptation for cross-environment generalization.

| Model | Description | Best For | Params |
|-------|-------------|----------|--------|
| **EI** | Gradient reversal domain adaptation | Cross-environment generalization | ~226K |
| **FewSense** | Prototypical few-shot learning | Few-shot cross-domain | ~459K |

## Choosing a Model

### By Dataset Size

| Dataset Size | Recommended Models |
|-------------|-------------------|
| Small (<1K samples) | MLPModel, CNN2DModel, LSTMModel, AttentionGRU, FewSense |
| Medium (1K-10K) | ResNet1D, BiLSTMAttention, CSIModel, THAT, CSITime, PA_CSI |
| Large (>10K) | VisionTransformerCSI, MambaCSI, EfficientNetCSI, EI |

### By Computational Budget

| Budget | Recommended Models |
|--------|-------------------|
| Ultra-low (MCU / edge) | AttentionGRU, WiFlexFormer |
| Low (CPU / small GPU) | MLPModel, CNN1DModel, CNN2DModel, LSTMModel, CSITime |
| Medium (single GPU) | ResNet1D, ResNet2D, BiLSTMAttention, GraphNeuralCSI, THAT, PA_CSI |
| High (multi-GPU) | VisionTransformerCSI, MambaCSI, EfficientNetCSI, EI, FewSense |

### By Task Characteristics

| Task Type | Recommended Models |
|-----------|-------------------|
| Gesture recognition | VisionTransformerCSI, CSIModel, ResNet2D, THAT |
| Gait analysis | BiLSTMAttention, MambaCSI, LSTMModel, CSITime |
| Activity detection | ResNet1D, EfficientNetCSI, CNN1DModel, CSITime, THAT |
| Fall detection | CNN2DModel, ResNet1D, MLPModel, AttentionGRU |
| Phase-sensitive tasks | PA_CSI, GraphNeuralCSI |
| Edge / real-time | WiFlexFormer, AttentionGRU |
| Cross-environment | EI, FewSense |
| Few-shot learning | FewSense |

## Input Format

All models expect CSI tensors in the format `(B, T, F, A)`:

- **B**: Batch size
- **T**: Time steps (e.g., 20-100)
- **F**: Frequency bins (e.g., 30 for the canonical grid)
- **A**: Antenna count (e.g., 3)

Both **complex** (`torch.complex64/128`) and **real** (`torch.float32`) inputs are supported, but the conversion differs by model family:

- **Mainstream, SOTA, lightweight, and cross-domain models** stack the real and imaginary parts along the antenna dimension (`A → 2A`), preserving phase information. Real inputs are zero-padded to the same width.
- **Baseline models** (`MLPModel`, `CNN1DModel`, `CNN2DModel`, `LSTMModel`) reduce complex input to its magnitude `|x|`, discarding phase.

## Model Registration

All models are registered in a central registry:

```python
from wsdp.models import list_models, get_model

# List all models: dict mapping name -> category
all_models = list_models()

# Filter by registry category: "baseline", "mainstream", or "sota"
baselines = list_models("baseline")     # MLPModel, CNN1DModel, CNN2DModel, LSTMModel
sota_models = list_models("sota")       # includes lightweight & cross-domain models

# Get a model by name (case-insensitive)
model = get_model("MambaCSI", num_classes=10, input_shape=(20, 30, 3))
```

## Custom Model Registration

Register your own models to use with the WSDP pipeline:

```python
from wsdp.models import register_model, create_model
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, input_shape, **kwargs):
        super().__init__()
        T, F, A = input_shape
        self.fc = nn.Linear(T * F * A, num_classes)

    def forward(self, x):
        # x: (B, T, F, A) real tensor
        return self.fc(x.reshape(x.shape[0], -1))

# Register under a category, then use via the standard API
register_model("custom", "MyModel", MyModel)
model = create_model("MyModel", num_classes=10, input_shape=(20, 30, 3))
```

## Performance Tips

1. **Start with baselines**: Always establish a baseline with MLPModel or CNN1DModel before trying complex architectures.

2. **Match model to data**:
   - Short sequences → CNN-based models
   - Long sequences → LSTM/Mamba models
   - Rich spatial structure → ViT or GNN models
   - Edge deployment → WiFlexFormer or AttentionGRU
   - Cross-environment → EI or FewSense

3. **Hyperparameter tuning**: Most models expose key hyperparameters:
   - `base_channels`: Controls model width (CNN/ResNet families)
   - `num_layers` / `num_blocks`: Controls depth
   - `hidden_size` / `embed_dim` / `d_model`: Controls representation capacity

4. **EfficientNetCSI**: Use `width_mult` and `depth_mult` < 1.0 for smaller models, > 1.0 for larger ones.

5. **VisionTransformerCSI**: Patch size is configured per dimension via `patch_size_f` (frequency) and `patch_size_a` (antenna). Larger patches mean fewer tokens — faster but less detailed.

6. **THAT**: Two-stream design processes temporal and spectral features in parallel for strong gesture recognition.

7. **WiFlexFormer / AttentionGRU**: Both under 60K parameters, ideal for on-device inference with minimal accuracy trade-off.

8. **EI**: Uses gradient reversal for domain adaptation — training requires domain labels in addition to activity labels (pass `return_domain=True` to get domain predictions).

9. **FewSense**: Best when only a handful of labeled samples are available in the target domain.
