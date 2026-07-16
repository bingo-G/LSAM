"""
Vectorized preprocess helpers for the VQA dataset.

Two utilities, both proven **bit-exact equivalent** to the legacy loop /
double-interpolate code in ``src.data.datamodule``:

1. ``crop_stack_patches_vec(x_3t, positions, ph, pw)``
   Replaces::

        patches = []
        for (y, x) in positions:
            p = x_3t[:, :, y:y+ph, x:x+pw]
            if p.shape[2] != ph or p.shape[3] != pw:
                p = F.pad(p, (0, pw - p.shape[3], 0, ph - p.shape[2]))
            patches.append(p)
        return torch.stack(patches, dim=0)      # [P, 3, T, ph, pw]

   with a single advanced-index gather when all positions stay within the
   frame bounds (the common case — GMS samplers already clamp positions to
   [0, H-ph] × [0, W-pw]). Falls back to the legacy loop verbatim when any
   position overflows, preserving F.pad zero-padding semantics.

2. ``resize_3t_hw_pair(dis_3t, ref_3t, tgt_h, tgt_w, antialias)``
   Batches the two ``F.interpolate`` calls for dis and ref into a single
   call along the N (frame) dimension. ``F.interpolate`` is independent
   per-image, so batching gives identical per-frame results.

Both functions live in a *separate* module so datamodule.py (217 KB) stays
small-edit-friendly: callers just ``from .sampling.preprocess_vec import …``.

The equivalence is covered by ``tests/run_preprocess_equivalence.py``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def crop_stack_patches_vec(
    x_3t: torch.Tensor,
    positions: Sequence[Tuple[int, int]],
    ph: int,
    pw: int,
) -> torch.Tensor:
    """Vectorized patch crop + stack.

    Args:
        x_3t: ``[3, T, H, W]`` source tensor.
        positions: sequence of ``(y, x)`` top-left coords (Python ints).
        ph: patch height.
        pw: patch width.

    Returns:
        ``[P, 3, T, ph, pw]`` contiguous tensor.

    Bit-exact equivalent of the legacy loop (see module docstring).
    """
    P = len(positions)
    if P == 0:
        # Empty result — preserves a sane shape. Callers never hit this in
        # practice (GMS/legacy grid always asks for >=1 patch).
        return x_3t.new_empty((0, x_3t.shape[0], x_3t.shape[1], ph, pw))

    _, _T, H, W = x_3t.shape

    # Fast-path check: all positions inside [0, H-ph] × [0, W-pw]?
    any_overflow = False
    for (y, x) in positions:
        if y < 0 or x < 0 or y + ph > H or x + pw > W:
            any_overflow = True
            break

    if not any_overflow:
        # Build [P, ph] and [P, pw] index grids then advanced-index once.
        # Result of x_3t[:, :, idx_h[:, :, None], idx_w[:, None, :]]:
        #   [3, T, P, ph, pw]   → permute → [P, 3, T, ph, pw]
        ys = torch.as_tensor([p[0] for p in positions],
                             dtype=torch.long, device=x_3t.device)
        xs = torch.as_tensor([p[1] for p in positions],
                             dtype=torch.long, device=x_3t.device)
        dh = torch.arange(ph, dtype=torch.long, device=x_3t.device)
        dw = torch.arange(pw, dtype=torch.long, device=x_3t.device)
        idx_h = ys.unsqueeze(1) + dh.unsqueeze(0)          # [P, ph]
        idx_w = xs.unsqueeze(1) + dw.unsqueeze(0)          # [P, pw]
        gathered = x_3t[:, :, idx_h.unsqueeze(-1), idx_w.unsqueeze(-2)]
        return gathered.permute(2, 0, 1, 3, 4).contiguous()

    # Fallback: preserve exact legacy semantics (F.pad zero-padding).
    patches: List[torch.Tensor] = []
    for (y, x) in positions:
        patch = x_3t[:, :, y:y + ph, x:x + pw]
        if patch.shape[2] != ph or patch.shape[3] != pw:
            patch = F.pad(patch, (0, pw - patch.shape[3], 0, ph - patch.shape[2]))
        patches.append(patch)
    return torch.stack(patches, dim=0)


def _interpolate_bilinear(
    frames: torch.Tensor,
    tgt_h: int,
    tgt_w: int,
    antialias: bool,
) -> torch.Tensor:
    """Thin wrapper that matches the exact kwargs used across the codebase."""
    if antialias:
        try:
            return F.interpolate(
                frames, size=(tgt_h, tgt_w),
                mode='bilinear', align_corners=False, antialias=True,
            )
        except TypeError:
            return F.interpolate(
                frames, size=(tgt_h, tgt_w),
                mode='bilinear', align_corners=False,
            )
    return F.interpolate(
        frames, size=(tgt_h, tgt_w),
        mode='bilinear', align_corners=False,
    )


def resize_3t_hw_pair(
    dis_3t: torch.Tensor,
    ref_3t: Optional[torch.Tensor],
    tgt_h: int,
    tgt_w: int,
    antialias: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Batch-resize a dis/ref pair of ``[3, T, H, W]`` tensors in one call.

    Bit-exact equivalent of calling the single-tensor resize twice —
    ``F.interpolate(mode='bilinear', align_corners=False[, antialias])``
    processes each frame independently, so batching along the N-dim does
    not change per-frame numerics.

    Same-size skip semantics mirror the legacy ``_resize_3t_hw``: if input
    already matches ``(tgt_h, tgt_w)`` we return it unchanged (no
    interpolate noise, critical for pre-resized YUV inputs).

    Args:
        dis_3t: ``[3, T_dis, H, W]`` distorted tensor.
        ref_3t: ``[3, T_ref, H, W]`` reference tensor or ``None``.
        tgt_h: target height.
        tgt_w: target width.
        antialias: passthrough to ``F.interpolate(antialias=…)``.

    Returns:
        ``(dis_out, ref_out)`` where each has shape ``[3, T, tgt_h, tgt_w]``
        (or original if same-size skipped). ``ref_out`` is ``None`` when
        ``ref_3t`` is ``None``.
    """
    dis_skip = (dis_3t.shape[2] == tgt_h and dis_3t.shape[3] == tgt_w)

    if ref_3t is None:
        if dis_skip:
            return dis_3t, None
        dis_frames = dis_3t.permute(1, 0, 2, 3).float()
        dis_r = _interpolate_bilinear(dis_frames, tgt_h, tgt_w, antialias)
        return dis_r.permute(1, 0, 2, 3).contiguous(), None

    ref_skip = (ref_3t.shape[2] == tgt_h and ref_3t.shape[3] == tgt_w)

    if dis_skip and ref_skip:
        return dis_3t, ref_3t

    if dis_skip:
        ref_frames = ref_3t.permute(1, 0, 2, 3).float()
        ref_r = _interpolate_bilinear(ref_frames, tgt_h, tgt_w, antialias)
        return dis_3t, ref_r.permute(1, 0, 2, 3).contiguous()

    if ref_skip:
        dis_frames = dis_3t.permute(1, 0, 2, 3).float()
        dis_r = _interpolate_bilinear(dis_frames, tgt_h, tgt_w, antialias)
        return dis_r.permute(1, 0, 2, 3).contiguous(), ref_3t

    # Both need resize. Only batch if source (H, W) matches — interpolate
    # requires a single output size per call.
    if dis_3t.shape[2] != ref_3t.shape[2] or dis_3t.shape[3] != ref_3t.shape[3]:
        dis_frames = dis_3t.permute(1, 0, 2, 3).float()
        ref_frames = ref_3t.permute(1, 0, 2, 3).float()
        dis_r = _interpolate_bilinear(dis_frames, tgt_h, tgt_w, antialias)
        ref_r = _interpolate_bilinear(ref_frames, tgt_h, tgt_w, antialias)
        return (dis_r.permute(1, 0, 2, 3).contiguous(),
                ref_r.permute(1, 0, 2, 3).contiguous())

    # [3, T, H, W] -> [T, 3, H, W], cat along N -> [2T, 3, H, W]
    T_dis = dis_3t.shape[1]
    dis_frames = dis_3t.permute(1, 0, 2, 3).float()
    ref_frames = ref_3t.permute(1, 0, 2, 3).float()
    both = torch.cat([dis_frames, ref_frames], dim=0)
    both_r = _interpolate_bilinear(both, tgt_h, tgt_w, antialias)
    dis_r = both_r[:T_dis].permute(1, 0, 2, 3).contiguous()
    ref_r = both_r[T_dis:].permute(1, 0, 2, 3).contiguous()
    return dis_r, ref_r
