"""Phase 1: Baseline models for CSI classification.

All models follow the unified interface:
    __init__(num_classes, input_shape, **kwargs)  where input_shape = (T, F, A)
    forward(x) where x shape = (B, T, F, A), output shape = (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


def _handle_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert complex CSI tensor to real by stacking real/imag as last dim.
    
    For real input, zero-pad the last dimension to match complex format (A → 2A).
    Output always has last dim = 2 * original_antenna_dim.
    """
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=-1)
    # Real input: treat as if imag=0, pad last dim to 2A
    B, T, F_dim, A = x.shape
    zeros = torch.zeros(B, T, F_dim, A, device=x.device, dtype=x.dtype)
    return torch.cat([x, zeros], dim=-1)


# ---------------------------------------------------------------------------
# 1. MLPModel — fully-connected baseline
# ---------------------------------------------------------------------------
class MLPModel(nn.Module):
    """MLP baseline: flatten (B,T,F,A) → MLP → logits."""

    def __init__(self, num_classes: int, input_shape: tuple, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        T, F_dim, A = input_shape
        # Account for complex → 2A real channels
        self.input_dim = T * F_dim * A * 2  # assume complex by default
        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)
        B = x.shape[0]
        x = x.reshape(B, -1)
        # Pad if needed (real input has fewer features)
        if x.shape[1] < self.input_dim:
            x = F.pad(x, (0, self.input_dim - x.shape[1]))
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. CNN1DModel — 1-D convolution over time axis
# ---------------------------------------------------------------------------
class CNN1DModel(nn.Module):
    """1D CNN: treat each (F,A) feature vector as a channel, convolve over time."""

    def __init__(self, num_classes: int, input_shape: tuple, base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        T, F_dim, A = input_shape
        in_channels = F_dim * A * 2  # complex → 2A per F
        layers = []
        ch = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers += [
                nn.Conv1d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))  # Global pooling over time
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, F_dim * A2, T)  # (B, F*2A, T)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# 3. CNN2DModel — 2-D convolution treating (F,A) as spatial dims
# ---------------------------------------------------------------------------
class CNN2DModel(nn.Module):
    """2D CNN: treat each time step independently with 2D conv, then pool over time."""

    def __init__(self, num_classes: int, input_shape: tuple, base_channels: int = 32, num_layers: int = 3):
        super().__init__()
        T, F_dim, A = input_shape
        self.num_time_steps = T
        in_ch = 2  # real + imag
        layers = []
        ch = in_ch
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            layers += [
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        # Use adaptive pooling to handle any input size
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape
        A_orig = A2 // 2
        x_real = x[..., :A_orig]  # (B, T, F, A)
        x_imag = x[..., A_orig:]  # (B, T, F, A)
        x = torch.stack([x_real, x_imag], dim=2)  # (B, T, 2, F, A)
        x = x.reshape(B * T, 2, F_dim, A_orig)
        feat = self.features(x).squeeze(-1).squeeze(-1)  # (B*T, ch)
        feat = feat.view(B, T, -1).mean(dim=1)  # (B, ch)
        return self.head(feat)


# ---------------------------------------------------------------------------
# 4. LSTMModel — LSTM over time axis
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """LSTM: encode spatial features per time step, then LSTM over time."""

    def __init__(self, num_classes: int, input_shape: tuple, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        T, F_dim, A = input_shape
        self.spatial_proj = nn.Sequential(
            nn.Linear(F_dim * A * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1)
        x = self.spatial_proj(x)  # (B, T, hidden)
        out, _ = self.lstm(x)  # (B, T, hidden)
        return self.head(out[:, -1, :])  # last time step


# ---------------------------------------------------------------------------
# Register all baseline models
# ---------------------------------------------------------------------------
register_model("baseline", "MLPModel", MLPModel)
register_model("baseline", "CNN1DModel", CNN1DModel)
register_model("baseline", "CNN2DModel", CNN2DModel)
register_model("baseline", "LSTMModel", LSTMModel)
