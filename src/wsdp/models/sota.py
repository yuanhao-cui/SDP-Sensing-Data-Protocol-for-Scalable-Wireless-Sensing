"""Phase 3: SOTA models for CSI classification.

All models follow the unified interface:
    __init__(num_classes, input_shape, **kwargs)  where input_shape = (T, F, A)
    forward(x) where x shape = (B, T, F, A), output shape = (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
# 9. VisionTransformerCSI — ViT adapted for CSI
# ---------------------------------------------------------------------------
class _PatchEmbed(nn.Module):
    """Embed (F, A) patches across time steps."""
    def __init__(self, patch_size_f, patch_size_a, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=(patch_size_f, patch_size_a),
                               stride=(patch_size_f, patch_size_a))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerCSI(nn.Module):
    """Vision Transformer for CSI: treats F×A as spatial patches across time steps.

    Each time step produces (F/patch_f) × (A/patch_a) patches, total T * patches per time.
    """

    def __init__(self, num_classes: int, input_shape: tuple, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 4, patch_size_f: int = 4,
                 patch_size_a: int = 4, dropout: float = 0.1):
        super().__init__()
        T, F_dim, A_dim = input_shape
        # Pad F and A to be divisible by patch sizes
        pad_f = (patch_size_f - F_dim % patch_size_f) % patch_size_f
        pad_a = (patch_size_a - A_dim % patch_size_a) % patch_size_a
        self._pad_f = pad_f
        self._pad_a = pad_a
        F_padded = F_dim + pad_f
        A_padded = A_dim + pad_a
        num_patches_per_t = (F_padded // patch_size_f) * (A_padded // patch_size_a)
        num_patches = T * num_patches_per_t

        self.patch_embed = _PatchEmbed(patch_size_f, patch_size_a, 2, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.blocks = nn.Sequential(*[
            _TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F_dim, A2 = x.shape
        A_orig = A2 // 2
        x_real = x[..., :A_orig]
        x_imag = x[..., A_orig:]
        x = torch.stack([x_real, x_imag], dim=2)  # (B, T, 2, F, A)

        # Pad spatial dims
        if self._pad_f > 0 or self._pad_a > 0:
            x = torch.nn.functional.pad(x, (0, self._pad_a, 0, self._pad_f))

        # Reshape: treat each time step as independent 2D image → patch embed
        _, _, C, F_p, A_p = x.shape
        x = x.reshape(B * T, C, F_p, A_p)
        patches = self.patch_embed(x)  # (B*T, num_patches_per_t, embed_dim)
        _, N_pt, D = patches.shape
        patches = patches.reshape(B, T * N_pt, D)  # (B, total_patches, D)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)  # (B, 1+total_patches, D)
        pos_cls = self.pos_embed_cls.expand(B, -1, -1)
        pos_patches = self.pos_embed[:, :patches.shape[1], :].expand(B, -1, -1)
        x = x + torch.cat([pos_cls, pos_patches], dim=1)

        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS token output


# ---------------------------------------------------------------------------
# 10. MambaCSI — State Space Model (simplified selective scan)
# ---------------------------------------------------------------------------
class _SelectiveSSM(nn.Module):
    """Simplified selective state space model block."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1,
                                 groups=self.d_inner, bias=False)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_part, z = xz.chunk(2, dim=-1)
        # Causal conv
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        # SSM params
        A = -torch.exp(self.A_log)  # (d_state,)
        x_dbl = self.x_proj(x_conv)  # (B, L, 2*d_state)
        B_ssm, C_ssm = x_dbl.chunk(2, dim=-1)
        delta = F.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)

        # Discretize and scan (simplified - sequential)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Simple selective scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(L):
            h = deltaA[:, i] * h + deltaB[:, i]
            y = (h * C_ssm[:, i].unsqueeze(1)).sum(-1)  # (B, d_inner)
            outputs.append(y + self.D * x_conv[:, i])
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)


class _MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = _SelectiveSSM(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        x = x + self.ssm(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MambaCSI(nn.Module):
    """Mamba (State Space Model) for CSI temporal modeling.

    Processes spatially-encoded CSI features with selective state space blocks.
    """

    def __init__(self, num_classes: int, input_shape: tuple, d_model: int = 128,
                 d_state: int = 16, num_layers: int = 4):
        super().__init__()
        T, F, A = input_shape
        self.spatial_encoder = nn.Sequential(
            nn.Linear(F * A * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.ssm_blocks = nn.Sequential(*[
            _MambaBlock(d_model, d_state) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1)
        x = self.spatial_encoder(x)  # (B, T, d_model)
        x = self.ssm_blocks(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))  # global average


# ---------------------------------------------------------------------------
# 11. GraphNeuralCSI — Graph Neural Network for antenna/subcarrier topology
# ---------------------------------------------------------------------------
class _GCNLayer(nn.Module):
    """Simple graph convolution layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        """
        x: (B, N, in_features)
        adj: (N, N) adjacency matrix (normalized)
        """
        support = self.linear(x)  # (B, N, out)
        out = torch.bmm(adj.unsqueeze(0).expand(x.shape[0], -1, -1), support)  # (B, N, out)
        out = out + self.bias
        B, N, C = out.shape
        out = self.bn(out.reshape(B * N, C)).reshape(B, N, C)
        return F.gelu(out)


class GraphNeuralCSI(nn.Module):
    """Graph Neural Network for CSI.

    Constructs a graph where nodes represent (frequency, antenna) pairs,
    with edges based on physical proximity. Processes each time step
    independently then aggregates.
    """

    def __init__(self, num_classes: int, input_shape: tuple, hidden_dim: int = 64,
                 num_gcn_layers: int = 3, num_heads: int = 4):
        super().__init__()
        T, F_dim, A_dim = input_shape
        self.num_time_steps = T
        self.num_freq = F_dim
        self.num_antennas = A_dim
        num_nodes = F_dim * A_dim

        # Build adjacency matrix based on subcarrier/antenna proximity
        adj = self._build_adjacency(F_dim, A_dim)
        self.register_buffer('adj', adj)

        # Node feature: real + imag per time step
        self.input_proj = nn.Linear(2, hidden_dim)

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            _GCNLayer(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)
        ])

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_adjacency(self, F, A):
        """Build normalized adjacency based on subcarrier-antenna grid."""
        N = F * A
        adj = torch.zeros(N, N)
        for f1 in range(F):
            for a1 in range(A):
                i = f1 * A + a1
                for f2 in range(F):
                    for a2 in range(A):
                        j = f2 * A + a2
                        # Connect if adjacent in frequency or antenna dimension
                        if abs(f1 - f2) <= 1 and abs(a1 - a2) <= 1:
                            adj[i, j] = 1.0
        # Normalize: D^{-1/2} A D^{-1/2}
        adj = adj + torch.eye(N)  # add self-loops
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / degree
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T, F, A2 = x.shape
        A_orig = A2 // 2
        x_real = x[..., :A_orig]
        x_imag = x[..., A_orig:]

        # For each time step, create node features (F*A nodes, each with 2 features)
        temporal_features = []
        for t in range(T):
            # (B, F, A) real/imag → (B, F*A, 2)
            nodes = torch.stack([x_real[:, t], x_imag[:, t]], dim=-1)
            nodes = nodes.reshape(B, F * A_orig, 2)
            nodes = self.input_proj(nodes)  # (B, N, hidden)

            # GCN message passing
            for gcn in self.gcn_layers:
                nodes = gcn(nodes, self.adj)

            # Global graph pooling
            graph_feat = nodes.mean(dim=1)  # (B, hidden)
            temporal_features.append(graph_feat)

        # Stack temporal features and apply attention
        temporal = torch.stack(temporal_features, dim=1)  # (B, T, hidden)
        attn_out, _ = self.temporal_attn(temporal, temporal, temporal)
        temporal = self.temporal_norm(temporal + attn_out)
        out = temporal.mean(dim=1)  # global average
        return self.head(out)


# ---------------------------------------------------------------------------
# Register all SOTA models
# ---------------------------------------------------------------------------
register_model("sota", "VisionTransformerCSI", VisionTransformerCSI)
register_model("sota", "MambaCSI", MambaCSI)
register_model("sota", "GraphNeuralCSI", GraphNeuralCSI)
