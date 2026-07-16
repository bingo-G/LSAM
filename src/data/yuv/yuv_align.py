"""
YUV Alignment: UV upsampling and stacking to unified [3, H, W] tensor.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union


def upsample_uv(
    U: torch.Tensor,
    V: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = 'bicubic',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Upsample U, V from [H/2, W/2] to [H, W].

    Args:
        U: [H/2, W/2] or [T, H/2, W/2]
        V: same shape as U
        target_h, target_w: output spatial dims
        mode: 'nearest' | 'bilinear' | 'bicubic'

    Returns:
        U_up, V_up: same shape but [H, W] or [T, H, W]
    """
    had_batch = U.dim() == 3  # [T, H/2, W/2]
    if U.dim() == 2:
        U = U.unsqueeze(0).unsqueeze(0)  # [1, 1, H/2, W/2]
        V = V.unsqueeze(0).unsqueeze(0)
    elif U.dim() == 3:
        T = U.shape[0]
        U = U.unsqueeze(1)  # [T, 1, H/2, W/2]
        V = V.unsqueeze(1)

    if mode == 'nearest':
        U_up = F.interpolate(U.float(), size=(target_h, target_w), mode=mode)
        V_up = F.interpolate(V.float(), size=(target_h, target_w), mode=mode)
    else:
        U_up = F.interpolate(U.float(), size=(target_h, target_w), mode=mode, align_corners=False)
        V_up = F.interpolate(V.float(), size=(target_h, target_w), mode=mode, align_corners=False)

    if had_batch:
        return U_up.squeeze(1), V_up.squeeze(1)  # [T, H, W]
    else:
        return U_up.squeeze(0).squeeze(0), V_up.squeeze(0).squeeze(0)  # [H, W]


def align_yuv(
    Y: 'Union[np.ndarray, torch.Tensor]',
    U: 'Union[np.ndarray, torch.Tensor]',
    V: 'Union[np.ndarray, torch.Tensor]',
    uv_upsample: str = 'bilinear',
    method: str = None,
) -> torch.Tensor:
    """
    Convert Y, U, V arrays/tensors to a unified YUV tensor with UV upsampled.

    Args:
        Y: float32 [T, H, W] or [H, W]  — numpy array or torch.Tensor
        U: float32 [T, H/2, W/2] or [H/2, W/2]  — numpy array or torch.Tensor
        V: same as U
        uv_upsample: interpolation mode

    Returns:
        yuv: float32 [3, T, H, W] or [3, H, W] depending on input
    """
    if method is not None:
        uv_upsample = method

    # Accept both numpy arrays and torch tensors (avoids numpy→torch round-trip
    # when the caller already provides tensors, e.g. from torch-accelerated decode).
    if isinstance(Y, np.ndarray):
        Y_t = torch.from_numpy(np.asarray(Y, dtype=np.float32)).float()
    else:
        Y_t = Y.float()
    if isinstance(U, np.ndarray):
        U_t = torch.from_numpy(np.asarray(U, dtype=np.float32)).float()
    else:
        U_t = U.float()
    if isinstance(V, np.ndarray):
        V_t = torch.from_numpy(np.asarray(V, dtype=np.float32)).float()
    else:
        V_t = V.float()

    if Y_t.dim() == 2:
        H, W = Y_t.shape
        U_up, V_up = upsample_uv(U_t, V_t, H, W, uv_upsample)
        return torch.stack([Y_t, U_up, V_up], dim=0)  # [3, H, W]
    else:
        T, H, W = Y_t.shape
        U_up, V_up = upsample_uv(U_t, V_t, H, W, uv_upsample)
        return torch.stack([Y_t, U_up, V_up], dim=0)  # [3, T, H, W]
