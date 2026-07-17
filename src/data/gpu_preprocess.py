"""
GPU-side preprocess for the semantic `gmsavg + legacy_pe` branch.

Scope (intentionally narrow for the first iteration):
  - Only the `semantic_sampler=gmsavg*` + `semantic_gms_legacy_pe=True` path
    on 1080p (and lower) inputs.
  - Only when the dataset emitted raw Y + U_half + V_half tensors (the new
    `raw_yuv_half_dis`/`raw_yuv_half_ref` batch keys, which replace the
    CPU-side `align_yuv` work).

What this module does, on the GPU, per clip, per sample:
  1. UV bicubic upsample   U_half[T,H/2,W/2] → U_up[T,H,W] (same for V)
  2. torch.stack([Y, U_up, V_up]) → [3, T, H, W]  (= dis_base, same as CPU path)
  3. (1080p fast-path) same-size skip instead of F.interpolate
  4. _crop_stack_patches_vec: [3, T, H, W] → [P, 3, T, ph, pw]
  5. No semantic_target_size resize (ph == target_size == 224 in the released
     LSAM config)

All ops above are bit-exact or numerically equivalent (float rounding < 1e-5)
to the existing CPU path in `_sample_semantic_gmsavg(legacy_pe)`; the UV
bicubic-upsample of a [T,1,H/2,W/2] float32 tensor produces the same numerical
result on CPU and GPU (non-antialias `F.interpolate(mode='bicubic')` is the
same algorithm, floating-point rounding may differ at the ULP level).

Equivalence is verified in `tests/run_gpu_preprocess_equivalence.py`.

Timing pattern used by evaluator
--------------------------------
The caller should wrap this entire function in a CUDA event / synchronize pair
so that the wall time recorded is the *GPU preprocess* wall time alone.  Any
H2D transfer done *before* this function belongs to data_load (and is already
overlapped with model forward via DataLoader workers).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _emulate_offline_uhd_grid(x_3t: torch.Tensor) -> torch.Tensor:
    """GPU 复刻 CPU 侧 ``_resize_3t_hw`` 的 UHD→sub-UHD 字节级模拟。

    与 ``datamodule._emulate_offline_uhd_grid`` 完全一致：模拟
    shift8 → resize → reverse_shift8 → 写 YUV → 重新读回 → shift8 的离线链路
    （tools/build_train_4k_resize.py 生成 Phase2_Resize1080p 的过程），
    使 on-the-fly 的 GPU resize 与预先离线降采样的 YUV 文件在数值上对齐。

    仅在源为 UHD(>=1500 行)、目标 <1500 行(即 4K→1080p)时调用。
    输入/输出均为已 stack 的 ``[3, T, H, W]`` (Y/U_up/V_up)。
    """
    _, T, tgt_h, tgt_w = x_3t.shape
    dst_h2, dst_w2 = tgt_h // 2, tgt_w // 2
    y = x_3t[0:1]                                            # [1, T, H, W]
    uv = x_3t[1:3]                                           # [2, T, H, W]
    # 把 UV 下采样回色度分辨率 (dst_h/2, dst_w/2)
    uv_small = F.interpolate(
        uv.permute(1, 0, 2, 3),                              # [T, 2, H, W]
        size=(dst_h2, dst_w2), mode='bilinear', align_corners=False,
    )
    # Y(全分辨率) + UV(色度分辨率) 量化到 n/255 网格 (uint8)
    y_q = torch.clamp((y * 255.0).round(), 0.0, 255.0) / 255.0
    uv_q_small = torch.clamp((uv_small * 255.0).round(), 0.0, 255.0) / 255.0
    # 量化后的 UV 再上采样回全分辨率 —— 对应读取离线文件时的 align_yuv
    uv_q_full = F.interpolate(
        uv_q_small, size=(tgt_h, tgt_w),
        mode='bilinear', align_corners=False,
    ).permute(1, 0, 2, 3).contiguous()                        # [2, T, H, W]
    out = torch.cat([y_q, uv_q_full], dim=0)                  # [3, T, H, W]
    return out.contiguous()


def _legacy_grid_positions(
    H: int, W: int,
    patch_h: int, patch_w: int,
    grid_size: int,
    patches_per_frame: int,
) -> List[Tuple[int, int]]:
    """Bit-exact copy of the CPU-side ``VQADataset._legacy_grid_positions`` for
    eval mode (``is_train=False``).

    Behaviour:
      1. Build a ``grid_size × grid_size`` grid of cells over (H, W).
      2. In each cell, center-crop a ``patch_h × patch_w`` window
         (``y_off = (gh - patch_h) // 2``).
      3. Clamp so the patch fits inside ``[0, H - patch_h]`` etc.
      4. If ``len(positions) > patches_per_frame``:
            deterministic evenly-spaced selection
            ``positions[int(i * step)] for i in range(N)``
            where ``step = len(positions) / patches_per_frame``.

    This must stay numerically identical to the CPU path at line 4500~ of
    ``datamodule.py::VQADataset._legacy_grid_positions`` whenever
    ``self.is_train is False``.
    """
    gs = max(1, int(grid_size))
    gh = max(1, H // gs)
    gw = max(1, W // gs)

    positions: List[Tuple[int, int]] = []
    for r in range(gs):
        for c in range(gs):
            y_s = r * gh
            x_s = c * gw
            y_off = max(0, (gh - patch_h) // 2)
            x_off = max(0, (gw - patch_w) // 2)
            y = min(max(y_s + y_off, 0), max(0, H - patch_h))
            x = min(max(x_s + x_off, 0), max(0, W - patch_w))
            positions.append((int(y), int(x)))

    max_patches = int(patches_per_frame) if patches_per_frame and patches_per_frame > 0 else len(positions)
    if len(positions) > max_patches:
        step = len(positions) / max_patches
        positions = [positions[int(i * step)] for i in range(max_patches)]
    return positions


def _crop_stack_patches_vec(
    x_3t: torch.Tensor,            # [3, T, H, W]
    positions: List[Tuple[int, int]],
    ph: int,
    pw: int,
) -> torch.Tensor:
    """Vectorised patch gather that is bit-exact to the CPU path (advanced index).

    Returns tensor of shape [P, 3, T, ph, pw] on the same device as ``x_3t``.
    """
    C, T, H, W = x_3t.shape
    device = x_3t.device
    P = len(positions)

    ys = torch.as_tensor([p[0] for p in positions], dtype=torch.long, device=device)  # [P]
    xs = torch.as_tensor([p[1] for p in positions], dtype=torch.long, device=device)  # [P]
    dh = torch.arange(ph, dtype=torch.long, device=device)
    dw = torch.arange(pw, dtype=torch.long, device=device)
    idx_h = ys.unsqueeze(1) + dh.unsqueeze(0)   # [P, ph]
    idx_w = xs.unsqueeze(1) + dw.unsqueeze(0)   # [P, pw]

    # Advanced index gather: x_3t[:, :, idx_h[:, :, None], idx_w[:, None, :]]
    # produces [C, T, P, ph, pw] → permute to [P, C, T, ph, pw].
    gathered = x_3t[:, :, idx_h.unsqueeze(-1), idx_w.unsqueeze(-2)]
    return gathered.permute(2, 0, 1, 3, 4).contiguous()


def _upsample_uv_generic(
    U_half: torch.Tensor,   # [T, H/2, W/2]
    V_half: torch.Tensor,   # [T, H/2, W/2]
    target_h: int,
    target_w: int,
    mode: str = 'bilinear',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upsample U/V planes to [T, target_h, target_w].

    Must match the behaviour of ``src.data.yuv.yuv_align.upsample_uv``:
      - ``'nearest'`` → no ``align_corners`` kwarg
      - ``'bilinear'`` / ``'bicubic'`` → ``align_corners=False``
    """
    U4 = U_half.unsqueeze(1).float()
    V4 = V_half.unsqueeze(1).float()
    mode = str(mode or 'bilinear').lower()
    if mode == 'nearest':
        U_up = F.interpolate(U4, size=(target_h, target_w), mode='nearest')
        V_up = F.interpolate(V4, size=(target_h, target_w), mode='nearest')
    else:
        # 'bilinear' or 'bicubic'
        U_up = F.interpolate(U4, size=(target_h, target_w), mode=mode, align_corners=False)
        V_up = F.interpolate(V4, size=(target_h, target_w), mode=mode, align_corners=False)
    return U_up.squeeze(1), V_up.squeeze(1)


def _upsample_uv_bicubic(
    U_half: torch.Tensor,   # [T, H/2, W/2]
    V_half: torch.Tensor,   # [T, H/2, W/2]
    target_h: int,
    target_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-compat wrapper: bicubic upsample (kept for tests)."""
    return _upsample_uv_generic(U_half, V_half, target_h, target_w, 'bicubic')


def build_resize_dis_legacy_pe(
    Y: torch.Tensor,              # [T, H, W]
    U_half: torch.Tensor,         # [T, H/2, W/2]
    V_half: torch.Tensor,         # [T, H/2, W/2]
    *,
    patch_h: int = 224,
    patch_w: int = 224,
    grid_size: int = 7,
    patches_per_frame: int = 8,
    semantic_gms_mode: str = 'stack',    # 'stack' | 'avg' | 'random1'
    semantic_target_size: Optional[int] = 224,
    uv_upsample: str = 'bilinear',       # 'nearest' | 'bilinear' | 'bicubic'
) -> torch.Tensor:
    """Rebuild `resize_dis` on GPU for the `gmsavg + legacy_pe` path.

    Returns
    -------
    resize_dis : torch.Tensor
        Shape depends on ``semantic_gms_mode``:
          * ``'stack'``   → ``[P, 3, T, ts, ts]``
          * ``'avg'``     → ``[3, T, ts, ts]``
          * ``'random1'`` → ``[3, T, ts, ts]``

    Notes
    -----
    For 1080p inputs (W=1920, H=1080) the function *skips* the 4K→1080p resize
    (same-size skip) to stay bit-exact with the CPU fast-path.  4K inputs are
    handled by `F.interpolate(mode='bilinear')` (same as CPU `_resize_3t_hw`).

    The released LSAM eval config uses ``semantic_gms_mode='stack'`` with
    ``patch_h == semantic_target_size``, so *no* semantic-side resize is
    performed (matching ``_sample_semantic_gmsavg`` behaviour).
    """
    assert Y.dim() == 3 and U_half.dim() == 3 and V_half.dim() == 3, \
        "expect Y[T,H,W], U_half/V_half[T,H/2,W/2]"

    T, H, W = Y.shape
    # Target shape: 1080p landscape / portrait — exactly matches the CPU branch
    if W >= H:
        tgt_h, tgt_w = 1080, 1920
    else:
        tgt_h, tgt_w = 1920, 1080

    # --- Step 1: UV upsample (matches CPU reader.uv_upsample, default bilinear) ---
    U_up, V_up = _upsample_uv_generic(U_half, V_half, H, W, uv_upsample)  # [T, H, W] each

    # --- Step 2: stack Y/U/V into [3, T, H, W] ---
    yuv = torch.stack([Y, U_up, V_up], dim=0)  # [3, T, H, W]

    # --- Step 3: same-size skip OR bilinear resize to 1080p ---
    if H == tgt_h and W == tgt_w:
        dis_base = yuv  # 1080p fast-path, bit-exact
    else:
        # [3,T,H,W] -> [T,3,H,W] for interpolate; [T,3,H,W] -> [3,T,H,W] back.
        frames = yuv.permute(1, 0, 2, 3).float()            # [T,3,H,W]
        frames_r = F.interpolate(
            frames, size=(tgt_h, tgt_w),
            mode='bilinear', align_corners=False,
        )
        dis_base = frames_r.permute(1, 0, 2, 3).contiguous()  # [3,T,tgt_h,tgt_w]
        # UHD(>=1500 行) → sub-UHD 降采样与离线预处理(Phase2_Resize1080p)
        # 保持一致的字节级对齐，行为与 CPU 侧 datamodule._resize_3t_hw 相同。
        if H >= 1500 and tgt_h < 1500:
            dis_base = _emulate_offline_uhd_grid(dis_base)

    # --- Step 4: legacy_pe grid patch positions ---
    positions = _legacy_grid_positions(
        H=tgt_h, W=tgt_w,
        patch_h=patch_h, patch_w=patch_w,
        grid_size=int(grid_size),
        patches_per_frame=int(patches_per_frame),
    )
    patches = _crop_stack_patches_vec(dis_base, positions, patch_h, patch_w)  # [P,3,T,ph,pw]

    # --- Step 5: mode reduction ---
    if semantic_gms_mode == 'stack':
        out = patches
    elif semantic_gms_mode == 'avg':
        out = patches.mean(dim=0)                                         # [3,T,ph,pw]
    else:  # random1 — eval uses center pick
        p = int(patches.shape[0])
        pick_idx = p // 2
        out = patches[pick_idx]                                           # [3,T,ph,pw]

    # --- Step 6: semantic_target_size resize (skipped when ph == ts) ---
    ts = int(semantic_target_size or patch_h)
    if patch_h != ts:
        if semantic_gms_mode == 'stack':
            # [P,3,T,ph,pw] → resize T/H/W each
            P, C, Tn, _, _ = out.shape
            flat = out.reshape(P * C * Tn, 1, patch_h, patch_w)
            flat_r = F.interpolate(flat, size=(ts, ts), mode='bilinear', align_corners=False)
            out = flat_r.reshape(P, C, Tn, ts, ts)
        else:
            C, Tn, _, _ = out.shape
            flat = out.reshape(C * Tn, 1, patch_h, patch_w)
            flat_r = F.interpolate(flat, size=(ts, ts), mode='bilinear', align_corners=False)
            out = flat_r.reshape(C, Tn, ts, ts)

    return out
