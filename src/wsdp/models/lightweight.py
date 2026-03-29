"""Lightweight WiFi CSI recognition models.

Contains efficient architectures suitable for resource-constrained deployment:
    - WiFlexFormer: Efficient WiFi Transformer (arXiv 2411.04224, 2024)
    - AttentionGRU: Lightweight Attention-GRU (Sensors, March 2025)

All models follow the unified interface:
    __init__(num_classes, input_shape, **kwargs)  where input_shape = (T, F, A)
    forward(x) where x shape = (B, T, F, A), output shape = (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .sota import _handle_complex


# ---------------------------------------------------------------------------
# WiFlexFormer - Efficient WiFi Transformer
# Paper: arXiv 2411.04224, 2024
# ---------------------------------------------------------------------------

class _DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))


class _EfficientTransformerLayer(nn.Module):
    """Transformer layer with efficient low-rank attention."""

    def __init__(self, d_model: int, key_dim: int = 32, num_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5

        # Low-rank projections for efficiency
        self.q_proj = nn.Linear(d_model, num_heads * key_dim)
        self.k_proj = nn.Linear(d_model, num_heads * key_dim)
        self.v_proj = nn.Linear(d_model, num_heads * key_dim)
        self.out_proj = nn.Linear(num_heads * key_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)

        # Multi-head attention with low-rank keys
        q = self.q_proj(h).reshape(B, L, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, L, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, L, self.num_heads, self.key_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.key_dim)
        x = x + self.out_proj(out)

        x = x + self.ffn(self.norm2(x))
        return x


class WiFlexFormer(nn.Module):
    """WiFlexFormer: Efficient WiFi Sensing Transformer.

    Reference: arXiv 2411.04224, 2024.

    Lightweight architecture (~0.1M params) using depthwise separable
    convolution stem and efficient transformer layers with reduced key
    dimensions for WiFi CSI classification.
    """

    def __init__(self, num_classes: int, input_shape: tuple, embed_dim: int = 48,
                 key_dim: int = 32, num_heads: int = 2, num_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        T, F_dim, A_dim = input_shape
        in_channels = F_dim * A_dim * 2

        # Depthwise separable conv stem
        self.stem = nn.Sequential(
            _DepthwiseSeparableConv(in_channels, embed_dim, kernel_size=5, padding=2),
            _DepthwiseSeparableConv(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, T, embed_dim) * 0.02)

        # Efficient transformer layers
        self.transformer_layers = nn.ModuleList([
            _EfficientTransformerLayer(embed_dim, key_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape
        # Flatten spatial: (B, T, F*2A) -> (B, F*2A, T) for conv stem
        x = x.reshape(B, T, -1).transpose(1, 2)
        x = self.stem(x)  # (B, embed_dim, T)
        x = x.transpose(1, 2)  # (B, T, embed_dim)

        # Add positional embedding (handle variable-length T)
        x = x + self.pos_embed[:, :T, :]

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, embed_dim, T)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# AttentionGRU - Lightweight Attention-based GRU
# Paper: Sensors, March 2025
# ---------------------------------------------------------------------------

class AttentionGRU(nn.Module):
    """Lightweight Attention-GRU for WiFi CSI recognition.

    Reference: Attention-based GRU for WiFi Sensing, Sensors, March 2025.

    Single-layer GRU with temporal attention mechanism. Very lightweight
    (~0.06M params target) suitable for edge deployment.

    Attention: score_t = tanh(W_h * h_t + b), alpha_t = softmax(v^T * score_t),
    context = sum(alpha_t * h_t).
    """

    def __init__(self, num_classes: int, input_shape: tuple, hidden_size: int = 64,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        T, F_dim, A_dim = input_shape
        input_size = F_dim * A_dim * 2

        # Single-layer GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Temporal attention parameters
        self.attn_W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape
        x = x.reshape(B, T, -1)  # (B, T, F*2A)

        # GRU
        h_all, _ = self.gru(x)  # (B, T, hidden_size)

        # Temporal attention: score = tanh(W*h + b), alpha = softmax(v*score)
        scores = torch.tanh(self.attn_W(h_all))  # (B, T, hidden_size)
        attn_weights = torch.softmax(self.attn_v(scores), dim=1)  # (B, T, 1)
        context = (attn_weights * h_all).sum(dim=1)  # (B, hidden_size)

        context = self.dropout(context)
        return self.classifier(context)


# ---------------------------------------------------------------------------
# Register all lightweight models
# ---------------------------------------------------------------------------
register_model("sota", "WiFlexFormer", WiFlexFormer)
register_model("sota", "AttentionGRU", AttentionGRU)
