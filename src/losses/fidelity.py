"""
Pairwise Fidelity Loss based on the Bhattacharyya coefficient.

Measures soft pairwise ranking agreement between predictions and targets.
Smoother and more informative than hinge-based rank loss — it provides
graded penalties for every pair, not just those violating a margin.

Supports **buffered mode** for gradient accumulation with small batch sizes.
Only computes when the buffer is full (= grad_accum samples).
"""

import torch
import torch.nn as nn
from collections import deque


class PairwiseFidelityLoss(nn.Module):
    r"""
    Pairwise fidelity loss using the Bhattacharyya coefficient.

    For each pair :math:`(i, j)` with :math:`i < j`, compute soft ordering
    probabilities via a temperature-scaled sigmoid:

    .. math::
        p_{ij} = \sigma(\alpha \cdot (\hat{y}_i - \hat{y}_j))
        \quad
        q_{ij} = \sigma(\alpha \cdot (y_i - y_j))

    The fidelity (Bhattacharyya coefficient) for each pair:

    .. math::
        F_{ij} = \sqrt{p_{ij} \, q_{ij}}
                + \sqrt{(1 - p_{ij})(1 - q_{ij})}

    Loss:

    .. math::
        \mathcal{L} = 1 - \frac{1}{|P|} \sum_{i < j} F_{ij}

    Args:
        alpha: temperature for the sigmoid (higher → sharper step function).
            10.0 works well when MOS scores are in [0, 1]–[0, 5] range.
        eps: numerical stability epsilon inside sqrt.
        buffer_size: number of accumulation steps to collect before computing.
            0 = no buffering.
    """

    def __init__(
        self,
        alpha: float = 10.0,
        eps: float = 1e-8,
        buffer_size: int = 0,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.buffer_size = buffer_size
        self._pred_buf: deque = deque(maxlen=buffer_size if buffer_size > 0 else 1)
        self._tgt_buf: deque = deque(maxlen=buffer_size if buffer_size > 0 else 1)

    # ------------------------------------------------------------------
    def reset_buffer(self):
        """Clear accumulation buffer.  Call after optimizer.step()."""
        self._pred_buf.clear()
        self._tgt_buf.clear()

    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B] predicted scores  (live, has grad)
            target: [B] ground-truth MOS

        Returns:
            scalar loss in [0, 1] * buffer_size   when buffer full
            0.0                                    otherwise
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

        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)     # [B, B]
        tgt_diff = target.unsqueeze(1) - target.unsqueeze(0)  # [B, B]

        p = torch.sigmoid(self.alpha * pred_diff)
        q = torch.sigmoid(self.alpha * tgt_diff)

        bc = (torch.sqrt((p * q).clamp(min=self.eps))
              + torch.sqrt(((1.0 - p) * (1.0 - q)).clamp(min=self.eps)))

        mask = torch.triu(torch.ones(B, B, device=device, dtype=bc.dtype),
                          diagonal=1)
        num_pairs = mask.sum().clamp(min=1.0)
        fidelity = (bc * mask).sum() / num_pairs
        return 1.0 - fidelity

    def compute_unbuffered(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute fidelity loss directly on the provided tensors (no internal buffer)."""
        return self._compute(pred, target, pred.device)
