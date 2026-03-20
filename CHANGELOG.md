# Changelog

All notable changes to WSDP are documented here.

## [Unreleased] — 2026-03-21

### 🔧 Bug Fixes

#### Model Architecture Fix (Critical)
**Problem**: Baseline models (MLPModel, CNN1DModel, CNN2DModel, LSTMModel) had a
dimension explosion bug inherited from an earlier refactoring. Models directly
flattened the full `(T, F, A)` tensor into the first Linear layer:

```python
# BEFORE (buggy):
input_dim = T * F * A * 2  # e.g., 199*30*9*2 = 107,460
x.reshape(B, -1)           # → Linear(107460, 512) = 55M params!
```

This caused severe overfitting on small datasets and was completely untrainable
for most practical CSI shapes.

**Fix**: All baseline models now use a **Spatial Encoder** (Conv2d-based) that
compresses `(F, A)` down to 1024 dimensions per time step before the temporal
processor — exactly matching Huyuochi's original CSIModel architecture:

```python
# AFTER (fixed):
# 1. Spatial encode: (B*T, 1, F, A) → SpatialEncoder → (B*T, 1024)
# 2. Temporal: (B, T, 1024) → CNN/LSTM/Transformer → (B, latent)
# 3. Classify: Linear → (B, num_classes)

# Parameter count comparison:
MLPModel:   55M → 664k  (98.8% reduction)
CNN1DModel: untrainable → 235k
```

**Files changed**: `src/wsdp/models/baselines.py`

**Tests**: 60 forward-pass tests + 268 total tests passing ✅

#### SpatialEncoder Adaptive Padding Fix
**Problem**: Conv2d kernel_size=3 with padding=1 requires input spatial dimensions
≥ (3, 2). For very small antenna arrays (F < 3 or A < 2), the convolution would
fail with "kernel size can't be greater than actual input size".

**Fix**: Added replication padding before the first convolution when
`F < 3 or A < 3`, ensuring the kernel always operates on ≥ (3, 3) spatial
dimensions without altering valid data semantics.

#### Processor squeeze() Guard Fix
**Problem**: `base_processor._process_single_csi()` used `.squeeze()` without
checking, which could silently drop antenna dimensions for single-antenna data.

**Fix**: Replaced bare `.squeeze()` with explicit shape checking that preserves
`(T, F, 1)` for single-antenna data and explicitly guards against degenerate
1D cases.

### 📖 Documentation

- Added architecture overview to `baselines.py` docstring (canonical input shape,
  spatial encoder diagram, adaptive padding explanation)
- Added this CHANGELOG.md

### 🔬 Tests

- **268 tests passing** (all unit, integration, inference, CLI tests)
- **Critical fixes verified** by regression-testing all model forward passes with
  both real and complex input across small/default/large input shapes

---

## [Previous Versions]

See `git log` for full history.
