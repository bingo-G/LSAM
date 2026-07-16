"""
Temporal Shift Module (TSM) - zero-parameter temporal modeling.
"""

import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    """
    TSM: Temporal Shift Module (zero extra parameters).
    Shifts a fraction of channels along the temporal dimension.
    """

    def __init__(self, fraction: float = 1 / 8):
        super().__init__()
        self.fraction = fraction

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            x: [B*T, D] or [B, T, D]
            T: number of temporal frames

        Returns:
            shifted x, same shape
        """
        if T <= 1:
            return x

        if x.dim() == 2:
            BT, D = x.shape
            B = BT // T
            x = x.view(B, T, D)
            was_2d = True
        else:
            was_2d = False
            D = x.shape[-1]

        fold = int(D * self.fraction)
        out = x.clone()
        out[:, 1:, :fold] = x[:, :-1, :fold]  # shift left
        out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]  # shift right
        # Remaining channels unchanged

        if was_2d:
            out = out.view(-1, D)

        return out
