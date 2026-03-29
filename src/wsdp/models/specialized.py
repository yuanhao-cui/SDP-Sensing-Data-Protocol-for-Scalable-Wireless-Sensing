"""Specialized WiFi CSI recognition models.

Contains architectures designed specifically for CSI-based human activity recognition:
    - THAT: Two-stream Convolution Augmented Transformer (Li et al., 2021)
    - CSITime: Inception-Time variant for CSI (Yadav et al., 2023)
    - PA_CSI: Phase-Amplitude dual-channel attention (Sensors 2025)

All models follow the unified interface:
    __init__(num_classes, input_shape, **kwargs)  where input_shape = (T, F, A)
    forward(x) where x shape = (B, T, F, A), output shape = (B, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .sota import _handle_complex


# ---------------------------------------------------------------------------
# THAT - Two-stream Convolution Augmented Transformer
# Paper: Li et al., "Two-stream Convolution Augmented Transformer for
#        Human Activity Recognition", AAAI 2021
# ---------------------------------------------------------------------------

class _GaussianRangeEncoding(nn.Module):
    """Gaussian Range Encoding for positional information."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Learnable Gaussian parameters
        self.mu = nn.Parameter(torch.linspace(0, max_len - 1, d_model))
        self.sigma = nn.Parameter(torch.ones(d_model) * (max_len / d_model))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns positional encoding of shape (1, seq_len, d_model)."""
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        # Gaussian: exp(-0.5 * ((pos - mu) / sigma)^2)
        encoding = torch.exp(-0.5 * ((positions - self.mu) / self.sigma.clamp(min=1e-5)) ** 2)
        return encoding.unsqueeze(0)


class _THATStream(nn.Module):
    """One stream of the THAT model: conv layers followed by transformer."""

    def __init__(self, in_channels: int, embed_dim: int, num_heads: int, depth: int,
                 max_len: int, dropout: float = 0.1):
        super().__init__()
        # Conv augmentation layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        # Positional encoding
        self.pos_enc = _GaussianRangeEncoding(max_len, embed_dim)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, L, embed_dim)"""
        # Conv expects (B, C, L)
        h = x.transpose(1, 2)
        h = self.conv1(h)
        h = self.conv2(h)
        h = h.transpose(1, 2)  # (B, L, embed_dim)
        h = h + self.pos_enc(h.shape[1], h.device)
        h = self.transformer(h)
        return h


class THAT(nn.Module):
    """Two-stream Convolution Augmented Transformer for Human Activity Recognition.

    Reference: Li et al., "Two-stream Convolution Augmented Transformer for
    Human Activity Recognition", AAAI 2021.

    Dual-stream architecture:
      - Channel stream: operates on (F x 2A) spatial features per time step
      - Temporal stream: operates on temporal features per (F, A) location
    Cross-attention fuses the two streams before classification.
    """

    def __init__(self, num_classes: int, input_shape: tuple, embed_dim: int = 64,
                 num_heads: int = 4, depth: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        T, F_dim, A_dim = input_shape
        spatial_dim = F_dim * A_dim * 2  # after _handle_complex: F * 2A

        # Channel stream: input is (B, T, F*2A) -> process spatial features across time
        self.channel_stream = _THATStream(
            in_channels=spatial_dim, embed_dim=embed_dim, num_heads=num_heads,
            depth=depth, max_len=T, dropout=dropout,
        )

        # Temporal stream: input is (B, F*2A, T) -> process temporal features across space
        self.temporal_stream = _THATStream(
            in_channels=T, embed_dim=embed_dim, num_heads=num_heads,
            depth=depth, max_len=spatial_dim, dropout=dropout,
        )

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape

        # Channel stream: flatten spatial dims
        x_ch = x.reshape(B, T, F_dim * A2)  # (B, T, F*2A)
        ch_out = self.channel_stream(x_ch)  # (B, T, embed_dim)

        # Temporal stream: flatten spatial, transpose to (B, spatial, T)
        x_temp = x.reshape(B, T, F_dim * A2).transpose(1, 2)  # (B, F*2A, T)
        temp_out = self.temporal_stream(x_temp)  # (B, F*2A, embed_dim)

        # Cross-attention: channel queries, temporal keys/values
        fused, _ = self.cross_attn(ch_out, temp_out, temp_out)
        fused = self.fusion_norm(ch_out + fused)

        # Global average pooling over time
        out = fused.mean(dim=1)  # (B, embed_dim)
        return self.head(out)


# ---------------------------------------------------------------------------
# CSITime - Inception-Time variant for CSI temporal classification
# Paper: Yadav et al., "CSITime: Privacy-preserving human activity
#        recognition using WiFi channel state information", 2023
# ---------------------------------------------------------------------------

class _InceptionModule(nn.Module):
    """Single Inception module with parallel conv branches and max-pool branch."""

    def __init__(self, in_channels: int, filters: int = 32):
        super().__init__()
        # Bottleneck
        self.bottleneck = nn.Conv1d(in_channels, filters, kernel_size=1, bias=False)

        # Parallel branches on bottlenecked input
        self.conv1 = nn.Conv1d(filters, filters, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(filters, filters, kernel_size=5, padding=2, bias=False)

        # Max-pool branch (operates on original input)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(filters * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, 4*filters, T)"""
        x_bt = self.bottleneck(x)
        out1 = self.conv1(x_bt)
        out3 = self.conv3(x_bt)
        out5 = self.conv5(x_bt)
        out_pool = self.conv_pool(self.maxpool(x))
        out = torch.cat([out1, out3, out5, out_pool], dim=1)
        out = F.relu(self.bn(out))
        return out


class _InceptionBlock(nn.Module):
    """Inception block with residual connection."""

    def __init__(self, in_channels: int, filters: int = 32):
        super().__init__()
        out_channels = filters * 4
        self.inception = _InceptionModule(in_channels, filters)
        # Residual: match channels if needed
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.inception(x) + self.residual(x))


class CSITime(nn.Module):
    """CSITime: Inception-Time variant for WiFi CSI activity recognition.

    Reference: Yadav et al., "CSITime: Privacy-preserving human activity
    recognition using WiFi channel state information", Neural Computing and
    Applications, 2023.

    Uses 3 stacked Inception blocks with residual connections, processing
    flattened (F*2A) features as channels over the temporal dimension.
    """

    def __init__(self, num_classes: int, input_shape: tuple, filters: int = 32,
                 num_blocks: int = 3, **kwargs):
        super().__init__()
        T, F_dim, A_dim = input_shape
        in_channels = F_dim * A_dim * 2  # F * 2A

        blocks = []
        ch = in_channels
        for _ in range(num_blocks):
            blocks.append(_InceptionBlock(ch, filters))
            ch = filters * 4
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape
        # Flatten spatial dims and transpose to (B, F*2A, T)
        x = x.reshape(B, T, -1).transpose(1, 2)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# PA_CSI - Phase-Amplitude dual-channel attention network
# Paper: Phase-Amplitude CSI attention network, Sensors 2025
# ---------------------------------------------------------------------------

class _MultiScaleConv(nn.Module):
    """Multi-scale 1D convolution with kernels 3, 5, 7."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"
        branch_ch = out_channels // 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, out_channels, T)"""
        return torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)


class _GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature fusion."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = self.gate(h)
        return self.norm(x + g * h)


class PA_CSI(nn.Module):
    """Phase-Amplitude dual-channel attention network for CSI recognition.

    Reference: Phase-Amplitude CSI Attention Network, Sensors, 2025.

    Extracts amplitude and phase features through separate multi-scale
    convolution branches, fuses them with a Gated Residual Network (GRN),
    and applies temporal attention for classification.
    """

    def __init__(self, num_classes: int, input_shape: tuple, hidden_dim: int = 96,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        T, F_dim, A_dim = input_shape
        # After _handle_complex: first A columns = real (amplitude proxy),
        # second A columns = imag (phase proxy)
        spatial_dim = F_dim * A_dim  # features per branch

        # Amplitude branch: multi-scale conv
        self.amp_conv = _MultiScaleConv(spatial_dim, hidden_dim)
        # Phase branch: multi-scale conv
        self.phase_conv = _MultiScaleConv(spatial_dim, hidden_dim)

        # GRN fusion
        self.grn = _GatedResidualNetwork(hidden_dim * 2, dropout=dropout)

        # Temporal attention
        self.attn_query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_key = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Classifier
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape
        A = A2 // 2

        # Split into amplitude (real) and phase (imag) parts
        amp = x[..., :A].reshape(B, T, -1)    # (B, T, F*A)
        phase = x[..., A:].reshape(B, T, -1)  # (B, T, F*A)

        # Multi-scale conv (expects B, C, T)
        amp_feat = self.amp_conv(amp.transpose(1, 2))    # (B, hidden, T)
        phase_feat = self.phase_conv(phase.transpose(1, 2))  # (B, hidden, T)

        # Concatenate and transpose back
        fused = torch.cat([amp_feat, phase_feat], dim=1).transpose(1, 2)  # (B, T, 2*hidden)

        # GRN fusion
        fused = self.grn(fused)  # (B, T, 2*hidden)

        # Temporal attention
        q = self.attn_query(fused)
        k = self.attn_key(fused)
        d_k = q.shape[-1]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, fused)  # (B, T, 2*hidden)

        # Pool over time
        out = context.mean(dim=1)  # (B, 2*hidden)
        return self.head(out)


# ---------------------------------------------------------------------------
# Register all specialized models
# ---------------------------------------------------------------------------
register_model("sota", "THAT", THAT)
register_model("sota", "CSITime", CSITime)
register_model("sota", "PA_CSI", PA_CSI)
