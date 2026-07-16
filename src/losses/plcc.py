"""
PLCC Loss: differentiable Pearson Linear Correlation Coefficient loss.

Minimizing (1 - PLCC) encourages linear agreement between predictions and targets.

Supports **buffered mode** for gradient accumulation with small batch sizes:
when batch_size=1, a single sample cannot define correlation. The buffer
collects detached predictions/targets across accumulation steps and only
computes the loss when the buffer is full (= grad_accum samples). This
avoids the catastrophic noise of computing PLCC on 2 samples (where
Pearson r is always ±1). Gradients only flow through the current (live)
batch; the loss is scaled by buffer_size to compensate for the /grad_accum
division in the trainer.
"""

import torch
import torch.nn as nn
from collections import deque


class PLCCLoss(nn.Module):
    """
    Differentiable PLCC-based loss: loss = 1 - PLCC(pred, target).

    Args:
        eps: numerical stability epsilon.
        buffer_size: number of accumulation steps to collect before computing.
            Set to grad_accum so the loss fires once per optimizer step.
            0 = no buffering (original behaviour, needs batch_size >= 2).
    """

    def __init__(self, eps: float = 1e-8, buffer_size: int = 0):
        super().__init__()
        self.eps = eps
        self.buffer_size = buffer_size
        # buffers store *detached* tensors from previous steps
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
            pred: [B] predicted scores  (live, has grad)
            target: [B] ground-truth MOS

        Returns:
            scalar loss = (1 - PLCC) * buffer_size   when buffer full
            0.0                                       otherwise
        """
        # ---- buffered mode: collect, compute only when full ----
        if self.buffer_size > 0:
            self._pred_buf.append(pred.detach())
            self._tgt_buf.append(target.detach())

            if len(self._pred_buf) < self.buffer_size:
                # Not enough samples yet — return zero (no gradient)
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

            # Buffer full — assemble all samples, replace last with live pred
            all_preds = list(self._pred_buf)
            all_preds[-1] = pred  # live tensor for gradient flow
            all_pred = torch.cat(all_preds)
            all_tgt = torch.cat(list(self._tgt_buf))

            loss = self._compute(all_pred, all_tgt, pred.device)
            # Scale by buffer_size to compensate for /grad_accum in trainer
            return loss * self.buffer_size

        # ---- unbuffered mode ----
        return self._compute(pred, target, pred.device)

    # ------------------------------------------------------------------
    def _compute(self, pred: torch.Tensor, target: torch.Tensor,
                 device: torch.device) -> torch.Tensor:
        if pred.numel() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pred_c = pred - pred.mean()
        target_c = target - target.mean()

        pred_var = (pred_c ** 2).sum()
        tgt_var = (target_c ** 2).sum()
        if pred_var < self.eps or tgt_var < self.eps:
            return torch.tensor(0.0, device=device, requires_grad=True)

        num = (pred_c * target_c).sum()
        den = torch.sqrt(pred_var * tgt_var).clamp(min=self.eps)
        plcc = (num / den).clamp(-1.0, 1.0)
        return 1.0 - plcc

    def compute_unbuffered(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute PLCC loss directly on the provided tensors (no internal buffer)."""
        return self._compute(pred, target, pred.device)
