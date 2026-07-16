"""
Pairwise Rank Loss.

Enforces correct ordering of predicted scores relative to ground truth labels.
Margin-based: if MOS(a) > MOS(b), then pred(a) should be > pred(b) by at least margin.

Supports **buffered mode** for gradient accumulation with small batch sizes.
Only computes when the buffer is full (= grad_accum samples), returning 0
at intermediate steps to avoid noisy gradients from tiny sample counts.
"""

import torch
import torch.nn as nn
from collections import deque


class PairwiseRankLoss(nn.Module):
    """
    Pairwise ranking loss that penalizes incorrectly ordered pairs.

    For all pairs (i, j) where target_i > target_j:
        loss += max(0, margin - (pred_i - pred_j))

    Args:
        margin: minimum required score gap for correctly ordered pairs.
        buffer_size: number of accumulation steps to collect before computing.
            0 = no buffering (original behaviour, needs batch_size >= 2).
    """

    def __init__(self, margin: float = 0.0, buffer_size: int = 0):
        super().__init__()
        self.margin = margin
        self.buffer_size = buffer_size
        self._pred_buf: deque = deque(maxlen=buffer_size if buffer_size > 0 else 1)
        self._tgt_buf: deque = deque(maxlen=buffer_size if buffer_size > 0 else 1)

    # ------------------------------------------------------------------
    def reset_buffer(self):
        """Clear accumulation buffer. Call after optimizer.step()."""
        self._pred_buf.clear()
        self._tgt_buf.clear()

    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B] predicted scores
            target: [B] ground-truth MOS

        Returns:
            scalar loss (0 until buffer is full)
        """
        # ---- buffered mode: collect, compute only when full ----
        if self.buffer_size > 0:
            self._pred_buf.append(pred.detach())
            self._tgt_buf.append(target.detach())

            if len(self._pred_buf) < self.buffer_size:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

            # Buffer full — assemble, replace last with live pred
            all_preds = list(self._pred_buf)
            all_preds[-1] = pred  # live tensor for gradient flow
            all_pred = torch.cat(all_preds)
            all_tgt = torch.cat(list(self._tgt_buf))

            loss = self._compute(all_pred, all_tgt, pred.device)
            return loss * self.buffer_size

        # ---- unbuffered mode ----
        return self._compute(pred, target, pred.device)

    # ------------------------------------------------------------------
    def _compute(self, pred: torch.Tensor, target: torch.Tensor,
                 device: torch.device) -> torch.Tensor:
        B = pred.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        target_diff = target.unsqueeze(1) - target.unsqueeze(0)   # [B, B]
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)         # [B, B]

        mask = (target_diff > 0).float()
        hinge = torch.clamp(self.margin - pred_diff, min=0.0)
        loss = (hinge * mask).sum()
        num_pairs = mask.sum().clamp(min=1.0)
        return loss / num_pairs

    def compute_unbuffered(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute rank loss directly on the provided tensors (no internal buffer)."""
        return self._compute(pred, target, pred.device)
