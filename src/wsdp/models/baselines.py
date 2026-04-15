"""
Phase 1: Baseline models for CSI classification.

Architecture Overview
====================
All baseline models follow a unified 3-stage pipeline:

    1. Spatial Encoder  (per time step)
       (B*T, 1, F, A) → SpatialEncoder → (B*T, spatial_dim=1024)

    2. Temporal Processor
       (B, T, spatial_dim) → [CNN / LSTM / Transformer] → (B, T, latent)

    3. Classifier
       (B, latent) → Linear → (B, num_classes)

Canonical Input Shape
====================
All models accept x of shape (B, T, F, A) where:
    B  = batch size
    T  = number of time steps (varies per file, padded to same length)
    F  = number of subcarriers / frequency bins (e.g., 30 for WiFi, 512 for mmWave)
    A  = number of antennas (Rx × Ant, e.g., 3×3=9 for WiFi)

Input Type
==========
Models accept both real and complex input:
    - Real (amplitude):    x.shape = (B, T, F, A)
    - Complex (CSI):       x.shape = (B, T, F, A) where values are complex64
    Internally, complex input is converted to amplitude via |x| before processing.

Spatial Encoder
==============
Mirrors Huyuochi's CSIModel spatial encoder:
    Conv2d(1, 32, 3, padding=1)
      → BatchNorm2d(32) → GELU()
      → DepthwiseSeparableConv(32, 64, kernel=3, padding=1)
      → PointwiseConv(64, 64)
      → AdaptiveAvgPool2d((4, 4))
      → Flatten
    Output: (B*T, 1024) per time step

Adaptive Padding
===============
For very small spatial dimensions (F < 3 or A < 3), the SpatialEncoder
applies replication padding before the first convolution to ensure
kernel compatibility without changing valid data.

Unified Interface
================
    model = MLPModel(num_classes=10, input_shape=(T, F, A))
    output = model(x)  # x: (B, T, F, A), output: (B, num_classes)

Registration
============
All models are registered in the model registry under category "baseline":
    from wsdp.models import get_model
    model = get_model("MLPModel", num_classes=10, input_shape=(T, F, A))
"""

import torch
import torch.nn as nn

from .registry import register_model


def _to_spatial(x: torch.Tensor) -> torch.Tensor:
    """
    Convert CSI tensor (B, T, F, A) to spatial encoder input (B*T, 1, F, A).
    
    Always converts to real amplitude first. Handles both real and complex inputs:
    - Complex: take magnitude |complex| → (B, T, F, A)
    - Real: use as-is → (B, T, F, A)
    Then reshape to (B*T, 1, F, A) for the spatial encoder.
    """
    # Convert complex to real magnitude
    if torch.is_complex(x):
        x = torch.abs(x)  # (B, T, F, A) — complex → amplitude
    
    B, T, F_dim, A = x.shape
    # (B, T, F, A) → permute → (B, T, A, F) → reshape → (B*T, 1, F, A)
    return x.permute(0, 1, 3, 2).reshape(B * T, 1, F_dim, A)


class SpatialEncoder(nn.Module):
    """
    Spatial encoder: (B*T, 1, F, A) → (B*T, spatial_dim)
    Mirrors Huyuochi's spatial_encoder in CSIModel.
    Conv2d(1, base_ch, 3) → DepthwiseConv2d(base_ch, base_ch*2) → AvgPool → Flatten
    Output: (B*T, base_channels*2*16) = (B*T, 1024) for base_ch=32
    """
    def __init__(self, F_dim, A, base_channels=32):
        super().__init__()
        # Extra padding: ensure input always >= (3, 3) before first conv
        pad_f = max(0, 3 - F_dim)  # pad F dimension if < 3
        pad_a = max(0, 3 - A)      # pad A dimension if < 3
        self._extra_pad = (0, pad_a, 0, pad_f)  # (left, right, top, bottom) for F.pad
        self.conv1 = nn.Conv2d(1, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.dwconv = nn.Conv2d(base_channels, base_channels * 2, 3, groups=base_channels, padding=1)
        self.pointwise = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.spatial_dim = base_channels * 2 * 16  # 32*2*16 = 1024

    def forward(self, x):
        # Apply extra padding to ensure F≥3, A≥3 before conv
        if any(s < k for s, k in zip(x.shape[2:], (3, 3))):
            x = torch.nn.functional.pad(x, self._extra_pad, mode='replicate')
        x = torch.nn.functional.gelu(self.bn1(self.conv1(x)))
        x = self.pointwise(self.dwconv(x))
        x = self.pool(x)
        return x.flatten(1)  # (B*T, spatial_dim)


# ---------------------------------------------------------------------------
# 1. MLPModel — Spatial encode → pool over time → MLP
# ---------------------------------------------------------------------------
class MLPModel(nn.Module):
    """MLP baseline: spatial encode per time step → temporal pool → MLP → logits."""

    def __init__(self, num_classes: int, input_shape: tuple, hidden_dims: list = None,
                 dropout: float = 0.3, base_channels: int = 32):
        super().__init__()
        T, F_dim, A = input_shape
        self.spatial_encoder = SpatialEncoder(F_dim, A, base_channels)
        spatial_dim = self.spatial_encoder.spatial_dim  # 1024

        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers = []
        in_dim = spatial_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = _to_spatial(x)  # (B*T, 1, F, A)
        spatial_feat = self.spatial_encoder(x_spatial)  # (B*T, spatial_dim)
        B = x.shape[0]
        T = x.shape[1]
        spatial_feat = spatial_feat.view(B, T, -1)  # (B, T, spatial_dim)
        pooled = spatial_feat.mean(dim=1)  # (B, spatial_dim)
        return self.net(pooled)


# ---------------------------------------------------------------------------
# 2. CNN1DModel — Spatial encode → 1D Conv over time
# ---------------------------------------------------------------------------
class CNN1DModel(nn.Module):
    """1D CNN: spatial encode per time step → Conv1d over time → pool → logits."""

    def __init__(self, num_classes: int, input_shape: tuple,
                 base_channels: int = 32, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        T, F_dim, A = input_shape
        self.spatial_encoder = SpatialEncoder(F_dim, A, base_channels)
        spatial_dim = self.spatial_encoder.spatial_dim  # 1024

        layers = []
        ch = spatial_dim
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers += [
                nn.Conv1d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.temporal_conv = nn.Sequential(*layers)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = _to_spatial(x)  # (B*T, 1, F, A)
        spatial_feat = self.spatial_encoder(x_spatial)  # (B*T, spatial_dim)
        B = x.shape[0]
        T = x.shape[1]
        spatial_feat = spatial_feat.view(B, T, -1)  # (B, T, spatial_dim)
        x = self.temporal_conv(spatial_feat.permute(0, 2, 1))  # (B, ch, T)
        x = x.squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# 3. CNN2DModel — Spatial encode → Conv2D over (time × spatial)
# ---------------------------------------------------------------------------
class CNN2DModel(nn.Module):
    """2D CNN: spatial encode → Conv2D over (time × spatial) → pool → logits."""

    def __init__(self, num_classes: int, input_shape: tuple,
                 base_channels: int = 32, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        T, F_dim, A = input_shape
        self.spatial_encoder = SpatialEncoder(F_dim, A, base_channels)
        # spatial_dim = self.spatial_encoder.spatial_dim  # 1024

        layers = []
        ch = 1
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.spatial_temporal = nn.Sequential(*layers)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = _to_spatial(x)  # (B*T, 1, F, A)
        spatial_feat = self.spatial_encoder(x_spatial)  # (B*T, spatial_dim)
        B = x.shape[0]
        T = x.shape[1]
        spatial_feat = spatial_feat.view(B, T, -1)  # (B, T, spatial_dim)
        x = spatial_feat.unsqueeze(1)  # (B, 1, T, spatial_dim)
        x = self.spatial_temporal(x)  # (B, ch, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, ch)
        return self.head(x)


# ---------------------------------------------------------------------------
# 4. LSTMModel — Spatial encode → LSTM over time
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """LSTM: spatial encode per time step → LSTM over time → last output → logits."""

    def __init__(self, num_classes: int, input_shape: tuple,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 base_channels: int = 32):
        super().__init__()
        T, F_dim, A = input_shape
        self.spatial_encoder = SpatialEncoder(F_dim, A, base_channels)
        spatial_dim = self.spatial_encoder.spatial_dim  # 1024

        self.lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = _to_spatial(x)  # (B*T, 1, F, A)
        spatial_feat = self.spatial_encoder(x_spatial)  # (B*T, spatial_dim)
        B = x.shape[0]
        T = x.shape[1]
        spatial_feat = spatial_feat.view(B, T, -1)  # (B, T, spatial_dim)
        out, _ = self.lstm(spatial_feat)  # (B, T, hidden_size)
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------------------
# Register all baseline models
# ---------------------------------------------------------------------------
register_model("baseline", "MLPModel", MLPModel)
register_model("baseline", "CNN1DModel", CNN1DModel)
register_model("baseline", "CNN2DModel", CNN2DModel)
register_model("baseline", "LSTMModel", LSTMModel)
