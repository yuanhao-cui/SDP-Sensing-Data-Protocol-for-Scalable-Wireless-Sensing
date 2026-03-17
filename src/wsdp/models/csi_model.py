"""CSIModel: CNN + Transformer for CSI classification (original WSDP model)."""

import torch
import torch.nn as nn
from .registry import register_model


class CSIModel(nn.Module):
    """Original CNN + Transformer model for CSI classification.

    Spatial encoder: 2D CNN over (F, A) per time step.
    Temporal processor: Transformer encoder over time dimension.
    """

    def __init__(self, num_classes: int = 10, input_shape: tuple = None,
                 base_channels: int = 32, latent_dim: int = 128):
        super().__init__()

        # Spatial Encoder: cope with dimension F and A
        # input shape (B*T, 1, F, A)
        self.num_classes = num_classes
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels*2, 3, groups=base_channels),
            nn.Conv2d(base_channels*2, base_channels*2, 1),
            nn.AdaptiveAvgPool2d((4, 4)), # -> (B*T, base_channels*2, 4, 4)
            nn.Flatten() # -> (B*T, base_channels*2 * 16)
        )

        # Temporal Processor: deal with dimension T
        # input shape: [B, T, latent_dim]
        self.temporal_processor = nn.Sequential(
            nn.Linear(base_channels*2*16, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=latent_dim*2,
                dropout=0.2,
                batch_first=True
            )
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.output_layer = nn.Linear(latent_dim, self.num_classes)

    def forward(self, x):
        # Handle complex input by taking magnitude
        if torch.is_complex(x):
            x = torch.abs(x)
        B, T, F, A = x.shape

        # Pad spatial dims if too small for 3x3 conv
        pad_f = max(0, 3 - F)
        pad_a = max(0, 3 - A)
        if pad_f > 0 or pad_a > 0:
            x = torch.nn.functional.pad(x, (0, pad_a, 0, pad_f))
            F = F + pad_f
            A = A + pad_a

        # shape: [B, T, F, A] -> [B*T, 1, F, A] '1' is the channel
        spatial_input = x.view(B*T, 1, F, A)

        # spatial_feat shape: [B*T, base_channels*2 * 16]
        spatial_feat = self.spatial_encoder(spatial_input)
        
        # shape: [B, T, base_channels*2 * 16]
        spatial_feat = spatial_feat.view(B, T, -1)

        # temporal_feat shape: [B, T, latent_dim]
        temporal_feat = self.temporal_processor(spatial_feat)
        
        # transpose(1, 2) -> shape: [B, latent_dim, T]
        # adaptive_pool -> shape: [B, latent_dim, 1]
        pooled = self.adaptive_pool(temporal_feat.transpose(1, 2))

        # flatten -> shape: [B, latent_dim]
        features = self.flatten(pooled)
        
        return self.output_layer(features)


# Register the original model
register_model("sota", "CSIModel", CSIModel)
