"""
Unified Aggregator: aggregates patch/tile/clip-level scores into a video score.
Works identically for both train (GMS patches) and inference (FuPiC tiles).
"""

import torch
import torch.nn as nn
from typing import Optional


class Aggregator(nn.Module):
    """
    Unified aggregator: maps per-item scores/features to a single video score.

    Modes:
        - 'mean': simple average
        - 'saliency_weighted': weighted by saliency from semantic features
        - 'learned': learnable attention pooling
    """

    def __init__(self, mode: str = 'mean', feature_dim: int = 256):
        super().__init__()
        self.mode = mode
        self.feature_dim = feature_dim

        if mode == 'learned':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.Tanh(),
                nn.Linear(feature_dim // 4, 1),
            )

    def forward(
        self,
        per_item_scores: torch.Tensor,
        per_item_feats: Optional[torch.Tensor] = None,
        saliency_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate per-item scores to a single video score.

        Args:
            per_item_scores: [B, N] scores for N patches/tiles/clips
            per_item_feats: [B, N, D] features (for learned/saliency modes)
            saliency_weights: [B, N] pre-computed weights

        Returns:
            video_score: [B]
        """
        if per_item_scores.dim() == 1:
            return per_item_scores

        if self.mode == 'mean':
            return per_item_scores.mean(dim=1)

        elif self.mode == 'saliency_weighted':
            if saliency_weights is not None:
                w = torch.softmax(saliency_weights, dim=1)
                return (per_item_scores * w).sum(dim=1)
            else:
                return per_item_scores.mean(dim=1)

        elif self.mode == 'learned':
            if per_item_feats is not None:
                attn_weights = self.attention(per_item_feats).squeeze(-1)  # [B, N]
                attn_weights = torch.softmax(attn_weights, dim=1)
                return (per_item_scores * attn_weights).sum(dim=1)
            else:
                return per_item_scores.mean(dim=1)

        return per_item_scores.mean(dim=1)
