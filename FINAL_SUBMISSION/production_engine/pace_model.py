"""
Drift-Sense++ PACE: Process-Aware Contextual Embedding Network
Features:
  - Local Structure: 64x64 fine patch
  - Neighborhood Context: 128x128 context patch
  - Process-Variation Overlap: 4 directional transition patches (Top, Bottom, Left, Right 32x32)
  - Regularized Group Ranking Head: Dropout(0.2) + Smooth Softmax Temperature (tau=0.25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ProcessAwareContextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # Local 64x64 branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 64, stride=2),   # 16x16
            ConvBlock(64, 96, stride=2),   # 8x8
            ConvBlock(96, 96, stride=2),   # 4x4
            nn.AdaptiveAvgPool2d(1),
        )

        # Context 128x128 branch
        self.context_branch = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False), # 64x64
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            ConvBlock(24, 48, stride=2),   # 32x32
            ConvBlock(48, 96, stride=2),   # 16x16
            ConvBlock(96, 96, stride=2),   # 8x8
            ConvBlock(96, 96, stride=2),   # 4x4
            nn.AdaptiveAvgPool2d(1),
        )

        # Overlap transition branch (4 channels x 32x32)
        self.overlap_branch = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1, bias=False), # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 64, stride=2),   # 8x8
            ConvBlock(64, 64, stride=2),   # 4x4
            nn.AdaptiveAvgPool2d(1),
        )

        # Fusion head
        self.fc_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
        )

        # Ranking scoring head
        self.ranking_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward_encoder(self, x_local: torch.Tensor, x_context: torch.Tensor, x_overlaps: torch.Tensor) -> torch.Tensor:
        feat_loc = self.local_branch(x_local).view(x_local.size(0), -1)
        feat_ctx = self.context_branch(x_context).view(x_context.size(0), -1)
        feat_ovl = self.overlap_branch(x_overlaps).view(x_overlaps.size(0), -1)

        fused = torch.cat([feat_loc, feat_ctx, feat_ovl], dim=1)
        z = self.fc_fusion(fused)
        z = F.normalize(z, p=2, dim=1)
        return z

    def forward(self, z_ref: torch.Tensor, z_cands: torch.Tensor, ncc_scores: torch.Tensor) -> torch.Tensor:
        cos_sims = F.cosine_similarity(z_ref.expand_as(z_cands), z_cands) # (K,)
        feat = torch.stack([cos_sims, ncc_scores], dim=1) # (K, 2)
        scores = self.ranking_head(feat).squeeze(1).unsqueeze(0) # (1, K)
        return scores


class GroupListRankingLoss(nn.Module):
    def __init__(self, temperature: float = 0.25):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, candidate_scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        logits = candidate_scores / self.temperature
        return self.cross_entropy(logits, target_indices)
