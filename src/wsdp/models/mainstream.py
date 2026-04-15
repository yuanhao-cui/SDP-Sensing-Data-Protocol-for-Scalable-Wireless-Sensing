"""Phase 2: Mainstream models for CSI classification.

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
    B, T, F_dim, A = x.shape
    zeros = torch.zeros(B, T, F_dim, A, device=x.device, dtype=x.dtype)
    return torch.cat([x, zeros], dim=-1)


# ---------------------------------------------------------------------------
# 5. ResNet1D — 1D residual network
# ---------------------------------------------------------------------------
class _ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """1D ResNet with 3 residual blocks over the time axis."""

    def __init__(self, num_classes: int, input_shape: tuple, base_channels: int = 64):
        super().__init__()
        T, F, A = input_shape
        in_channels = F * A * 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.layer1 = _ResBlock1D(base_channels, base_channels)
        self.layer2 = _ResBlock1D(base_channels, base_channels * 2, stride=2)
        self.layer3 = _ResBlock1D(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F, A2 = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, F * A2, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# 6. ResNet2D — 2D residual network
# ---------------------------------------------------------------------------
class _ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + self.shortcut(x))


class ResNet2D(nn.Module):
    """2D ResNet treating each time step as an independent 2D sample."""

    def __init__(self, num_classes: int, input_shape: tuple, base_channels: int = 32):
        super().__init__()
        T, F, A = input_shape
        self.T = T
        in_ch = 2  # real + imag channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.layer1 = _ResBlock2D(base_channels, base_channels)
        self.layer2 = _ResBlock2D(base_channels, base_channels * 2, stride=2)
        self.layer3 = _ResBlock2D(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F, A2 = x.shape
        A_orig = A2 // 2
        x_real = x[..., :A_orig]
        x_imag = x[..., A_orig:]
        x = torch.stack([x_real, x_imag], dim=2)  # (B, T, 2, F, A)
        x = x.reshape(B * T, 2, F, A_orig)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B*T, ch)
        x = x.view(B, T, -1).mean(dim=1)  # temporal average
        return self.head(x)


# ---------------------------------------------------------------------------
# 7. BiLSTMAttention — Bidirectional LSTM with multi-head attention
# ---------------------------------------------------------------------------
class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM + multi-head self-attention over time."""

    def __init__(self, num_classes: int, input_shape: tuple, hidden_size: int = 128,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        T, F, A = input_shape
        self.spatial_proj = nn.Sequential(
            nn.Linear(F * A * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1)
        x = self.spatial_proj(x)  # (B, T, hidden)
        x, _ = self.bilstm(x)  # (B, T, hidden*2)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)  # residual
        x = x.mean(dim=1)  # global average over time
        return self.head(x)


# ---------------------------------------------------------------------------
# 8. EfficientNetCSI — configurable efficient CNN
# ---------------------------------------------------------------------------
class _MBConv(nn.Module):
    """Mobile inverted bottleneck conv block."""
    def __init__(self, in_ch, out_ch, expand_ratio=4, kernel_size=3, stride=1):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU()]
        layers += [
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                       padding=kernel_size // 2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_residual else out


class EfficientNetCSI(nn.Module):
    """EfficientNet-inspired CSI classifier with configurable width/depth."""

    def __init__(self, num_classes: int, input_shape: tuple, width_mult: float = 1.0,
                 depth_mult: float = 1.0, base_channels: int = 16):
        super().__init__()
        T, F, A = input_shape
        self.T = T
        ch = max(8, int(base_channels * width_mult))

        # Process each time step as 2D image
        self.stem = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
        )
        # Build MBConv stages
        def _round(v):
            return max(1, int(v + 0.5))
        configs = [
            # (out_ch, num_blocks, stride, expand, kernel)
            (int(24 * width_mult), _round(2 * depth_mult), 1, 6, 3),
            (int(40 * width_mult), _round(3 * depth_mult), 2, 6, 5),
            (int(80 * width_mult), _round(3 * depth_mult), 2, 6, 3),
            (int(112 * width_mult), _round(4 * depth_mult), 1, 6, 5),
        ]
        stages = []
        for out_ch, n_blocks, stride, expand, ks in configs:
            for i in range(n_blocks):
                s = stride if i == 0 else 1
                stages.append(_MBConv(ch, out_ch, expand, ks, s))
                ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F, A2 = x.shape
        A_orig = A2 // 2
        x_real = x[..., :A_orig]
        x_imag = x[..., A_orig:]
        x = torch.stack([x_real, x_imag], dim=2)  # (B, T, 2, F, A)
        x = x.reshape(B * T, 2, F, A_orig)
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_pool(x).squeeze(-1).squeeze(-1)  # (B*T, ch)
        x = x.view(B, T, -1).mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Register all mainstream models
# ---------------------------------------------------------------------------
register_model("mainstream", "ResNet1D", ResNet1D)
register_model("mainstream", "ResNet2D", ResNet2D)
register_model("mainstream", "BiLSTMAttention", BiLSTMAttention)
register_model("mainstream", "EfficientNetCSI", EfficientNetCSI)
