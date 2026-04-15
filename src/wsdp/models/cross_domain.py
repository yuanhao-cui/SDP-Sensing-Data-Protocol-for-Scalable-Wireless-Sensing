"""Cross-domain WiFi CSI recognition models.

Contains architectures for domain adaptation and few-shot learning:
    - EI: Environment-Independent model (Jiang et al., 2020)
    - FewSense: Few-shot cross-domain model (TMC 2022)

All models follow the unified interface:
    __init__(num_classes, input_shape, **kwargs)  where input_shape = (T, F, A)
    forward(x) where x shape = (B, T, F, A), output shape = (B, num_classes)
"""

import torch
import torch.nn as nn
from torch.autograd import Function

from .registry import register_model
from .sota import _handle_complex


# ---------------------------------------------------------------------------
# EI - Environment-Independent model with gradient reversal
# Paper: Jiang et al., "Towards Environment Independent Device Free
#        Human Activity Recognition", MobiCom 2020
# ---------------------------------------------------------------------------

class _GradientReversalFunction(Function):
    """Gradient reversal layer for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class _GradientReversal(nn.Module):
    """Gradient reversal layer wrapper."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.alpha)


class EI(nn.Module):
    """Environment-Independent model for cross-domain CSI recognition.

    Reference: Jiang et al., "Towards Environment Independent Device Free
    Human Activity Recognition", ACM MobiCom, 2020.

    Architecture:
      - Encoder: 3-layer 1D conv (64->128->256) on flattened (F*2A) features
      - Domain discriminator: gradient reversal + 2-layer MLP (adversarial training)
      - Activity classifier: 2-layer MLP

    During training, the gradient reversal layer forces the encoder to learn
    domain-invariant features. At inference, only encoder + classifier are used.
    The domain_alpha parameter controls the strength of the reversal.
    """

    def __init__(self, num_classes: int, input_shape: tuple, domain_alpha: float = 1.0,
                 num_domains: int = 2, **kwargs):
        super().__init__()
        T, F_dim, A_dim = input_shape
        in_features = F_dim * A_dim * 2

        # Encoder: 3-layer 1D conv over time
        self.encoder = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Activity classifier: 2-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Domain discriminator: gradient reversal + 2-layer MLP
        self.gradient_reversal = _GradientReversal(alpha=domain_alpha)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract domain-invariant features."""
        x = _handle_complex(x)  # (B, T, F, 2A)
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1).transpose(1, 2)  # (B, F*2A, T)
        features = self.encoder(x).flatten(1)  # (B, 256)
        return features

    def forward(self, x: torch.Tensor, return_domain: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, T, F, A).
            return_domain: If True, also return domain predictions (for training).

        Returns:
            Activity predictions (B, num_classes). If return_domain=True, returns
            tuple of (activity_preds, domain_preds).
        """
        features = self.encode(x)  # (B, 256)
        class_output = self.classifier(features)

        if return_domain:
            reversed_features = self.gradient_reversal(features)
            domain_output = self.domain_discriminator(reversed_features)
            return class_output, domain_output

        return class_output


# ---------------------------------------------------------------------------
# FewSense - Few-shot cross-domain sensing
# Paper: "FewSense: Towards a Scalable and Cross-Domain Wi-Fi Sensing
#        System Using Few-Shot Learning", IEEE TMC, 2022
# ---------------------------------------------------------------------------

class FewSense(nn.Module):
    """FewSense: Few-shot cross-domain WiFi sensing model.

    Reference: "FewSense: Towards a Scalable and Cross-Domain Wi-Fi Sensing
    System Using Few-Shot Learning", IEEE Trans. Mobile Computing, 2022.

    Architecture:
      - Feature extractor: 4-layer conv (32->64->128->256)
      - Prototypical network: compute class prototypes from support set
      - Standard classifier: fallback when no support set is available

    At inference with a labeled support set, classifies by nearest prototype
    (Euclidean distance in embedding space). Without support data, falls back
    to a standard linear classifier head.
    """

    def __init__(self, num_classes: int, input_shape: tuple, embed_dim: int = 256,
                 dropout: float = 0.2, **kwargs):
        super().__init__()
        T, F_dim, A_dim = input_shape
        in_channels = A_dim * 2

        # Feature extractor: 4-layer conv
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Dropout2d(dropout),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Dropout2d(dropout),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Dropout2d(dropout),
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Standard classifier fallback (when no support set available)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from input CSI data.

        Args:
            x: Input tensor (B, T, F, A).

        Returns:
            Embedding tensor (B, embed_dim).
        """
        x = _handle_complex(x)  # (B, T, F, 2A)
        # Rearrange to (B, 2A, T, F)
        x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x).flatten(1)  # (B, 256)
        return self.embedding(features)  # (B, embed_dim)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None,
                support_y: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with optional prototypical classification.

        Args:
            x: Query input (B, T, F, A).
            support_x: Support set input (N_support, T, F, A). Optional.
            support_y: Support set labels (N_support,). Optional.

        Returns:
            Class predictions (B, num_classes). When support set is provided,
            returns negative distances to prototypes (suitable for cross-entropy
            loss). Otherwise, returns standard classifier logits.
        """
        query_embed = self.extract_features(x)  # (B, embed_dim)

        if support_x is not None and support_y is not None:
            # Prototypical network mode
            support_embed = self.extract_features(support_x)  # (N, embed_dim)

            # Compute class prototypes
            unique_classes = torch.unique(support_y)
            prototypes = []
            for c in unique_classes:
                mask = support_y == c
                prototypes.append(support_embed[mask].mean(dim=0))
            prototypes = torch.stack(prototypes)  # (C, embed_dim)

            # Classify by negative Euclidean distance to prototypes
            # (B, 1, D) - (1, C, D) -> (B, C, D) -> (B, C)
            dists = torch.cdist(query_embed.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
            return -dists  # negative distance as logits

        # Standard classifier mode
        return self.classifier(query_embed)


# ---------------------------------------------------------------------------
# Register all cross-domain models
# ---------------------------------------------------------------------------
register_model("sota", "EI", EI)
register_model("sota", "FewSense", FewSense)
