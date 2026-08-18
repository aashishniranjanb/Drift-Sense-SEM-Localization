"""
Lightweight Multi-Scale Siamese Structural Re-Ranker for Drift-Sense++ HCR

Architecture:
  - Dual-scale input: 64×64 (local structure) + 128×128 (neighborhood context)
  - Small depthwise-separable CNN encoder (~350k params)
  - 128-dimensional structural embedding
  - Cosine similarity for candidate comparison
  - Trained with triplet loss on hard-negative periodic semiconductor replicas

The network answers: "Does this candidate patch represent the same physical site
as the reference?" — NOT "where is the site?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """MobileNet-style depthwise separable convolution block."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)), inplace=True)
        x = F.relu(self.bn2(self.pointwise(x)), inplace=True)
        return x


class LocalEncoder(nn.Module):
    """Encodes a 64×64 grayscale patch into a 64-dim embedding."""
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),  # 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64, stride=2),    # 16×16
            DepthwiseSeparableConv(64, 128, stride=2),   # 8×8
            DepthwiseSeparableConv(128, 128, stride=2),  # 4×4
            nn.AdaptiveAvgPool2d(1),                     # 1×1
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ContextEncoder(nn.Module):
    """Encodes a 128×128 grayscale patch into a 64-dim embedding."""
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),  # 64×64
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(24, 48, stride=2),    # 32×32
            DepthwiseSeparableConv(48, 96, stride=2),    # 16×16
            DepthwiseSeparableConv(96, 128, stride=2),   # 8×8
            DepthwiseSeparableConv(128, 128, stride=2),  # 4×4
            nn.AdaptiveAvgPool2d(1),                     # 1×1
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiScaleSiameseEncoder(nn.Module):
    """
    Multi-scale Siamese encoder combining local (64×64) and context (128×128) features.

    Input: (batch, 1, 64, 64) + (batch, 1, 128, 128)
    Output: (batch, 128) L2-normalized embedding
    """
    def __init__(self, local_dim: int = 64, context_dim: int = 64):
        super().__init__()
        self.local_encoder = LocalEncoder(embed_dim=local_dim)
        self.context_encoder = ContextEncoder(embed_dim=context_dim)
        self.embed_dim = local_dim + context_dim

        # Fusion projection
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, x_local: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        z_local = self.local_encoder(x_local)
        z_context = self.context_encoder(x_context)
        z = torch.cat([z_local, z_context], dim=1)
        z = self.projection(z)
        z = F.normalize(z, p=2, dim=1)
        return z

    def encode_single_scale(self, x_64: torch.Tensor) -> torch.Tensor:
        """Fallback: encode only the 64×64 patch (when context not available)."""
        z_local = self.local_encoder(x_64)
        z_context = torch.zeros(x_64.size(0), 64, device=x_64.device)
        z = torch.cat([z_local, z_context], dim=1)
        z = self.projection(z)
        z = F.normalize(z, p=2, dim=1)
        return z


class TripletLoss(nn.Module):
    """Triplet loss with hard-negative margin."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_pos = 1.0 - F.cosine_similarity(anchor, positive)  # distance in [0, 2]
        d_neg = 1.0 - F.cosine_similarity(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = MultiScaleSiameseEncoder()
    print(f"Model parameters: {count_parameters(model):,}")

    x64 = torch.randn(4, 1, 64, 64)
    x128 = torch.randn(4, 1, 128, 128)
    z = model(x64, x128)
    print(f"Embedding shape: {z.shape}")
    print(f"Embedding norm: {torch.norm(z, dim=1)}")

    # Test triplet loss
    anchor = model(x64, x128)
    pos = model(torch.randn(4, 1, 64, 64), torch.randn(4, 1, 128, 128))
    neg = model(torch.randn(4, 1, 64, 64), torch.randn(4, 1, 128, 128))

    criterion = TripletLoss(margin=0.5)
    loss = criterion(anchor, pos, neg)
    print(f"Triplet loss: {loss.item():.4f}")
