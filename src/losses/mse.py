"""
MSE Loss with optional target normalization.
Serves as the primary regression loss.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Standard MSE loss with optional per-batch z-score normalization of targets.

    If normalize=True, targets are z-normalized within the batch before computing
    MSE against similarly normalized predictions. This stabilizes multi-dataset
    training where label ranges differ.
    """

    def __init__(self, normalize: bool = False, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B] predicted scores
            target: [B] ground-truth MOS

        Returns:
            scalar loss
        """
        if self.normalize and pred.numel() > 1:
            t_mean = target.mean()
            t_std = target.std().clamp(min=self.eps)
            target_n = (target - t_mean) / t_std

            p_mean = pred.mean()
            p_std = pred.std().clamp(min=self.eps)
            pred_n = (pred - p_mean) / p_std

            return torch.mean((pred_n - target_n) ** 2)

        return torch.mean((pred - target) ** 2)
