"""
Colorspace conversion utilities: RGB <-> YUV (BT.709 / BT.601).
All operations are on float32 [0, 1] tensors.
"""

import torch
import numpy as np

# BT.709 conversion matrix (from RGB to YUV)
# Y =  0.2126 R + 0.7152 G + 0.0722 B
# U = -0.1146 R - 0.3854 G + 0.5000 B  (Cb, offset by +0.5 for unsigned)
# V =  0.5000 R - 0.4542 G - 0.0458 B  (Cr, offset by +0.5 for unsigned)

# For unsigned representation (U, V in [0, 1]):
_RGB_TO_YUV_709 = torch.tensor([
    [ 0.2126,  0.7152,  0.0722],
    [-0.1146, -0.3854,  0.5000],
    [ 0.5000, -0.4542, -0.0458],
], dtype=torch.float32)

# Inverse: YUV (with U,V centered at 0.5) -> RGB
# We first subtract 0.5 from U,V, then:
# R = Y + 1.5748 * V'
# G = Y - 0.1873 * U' - 0.4681 * V'
# B = Y + 1.8556 * U'
_YUV_TO_RGB_709 = torch.tensor([
    [1.0,  0.0000,  1.5748],
    [1.0, -0.1873, -0.4681],
    [1.0,  1.8556,  0.0000],
], dtype=torch.float32)

# BT.601 conversion matrix (RGB -> YUV)
# Y =  0.2990 R + 0.5870 G + 0.1140 B
# U = -0.168736 R - 0.331264 G + 0.500000 B
# V =  0.500000 R - 0.418688 G - 0.081312 B
_RGB_TO_YUV_601 = torch.tensor([
    [ 0.299000,  0.587000,  0.114000],
    [-0.168736, -0.331264,  0.500000],
    [ 0.500000, -0.418688, -0.081312],
], dtype=torch.float32)

# BT.601 inverse: YUV (U,V centered) -> RGB
_YUV_TO_RGB_601 = torch.tensor([
    [1.0,  0.000000,  1.402000],
    [1.0, -0.344136, -0.714136],
    [1.0,  1.772000,  0.000000],
], dtype=torch.float32)


def yuv_to_rgb_bt709(yuv: torch.Tensor) -> torch.Tensor:
    """
    Convert YUV [0,1] to RGB [0,1] using BT.709.
    U and V are assumed unsigned [0,1] and will be centered first.

    Args:
        yuv: float tensor [..., 3, H, W] where channels are (Y, U, V)

    Returns:
        rgb: float tensor [..., 3, H, W]
    """
    # Move color channel to last dim for matmul
    # yuv shape: [..., 3, H, W] -> [..., H, W, 3]
    yuv = yuv.float()
    shape = yuv.shape
    # Channels are at dim -3
    yuv_perm = yuv.movedim(-3, -1)  # [..., H, W, 3]

    # Center U, V
    offset = torch.tensor([0.0, 0.5, 0.5], device=yuv.device, dtype=torch.float32)
    yuv_centered = yuv_perm - offset

    mat = _YUV_TO_RGB_709.to(device=yuv.device, dtype=torch.float32)
    rgb = torch.matmul(yuv_centered, mat.T)  # [..., H, W, 3]

    # Move back
    rgb = rgb.movedim(-1, -3)  # [..., 3, H, W]
    return rgb


def rgb_to_yuv_bt709(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB [0,1] to YUV [0,1] using BT.709.
    Output U, V are unsigned [0,1] (offset by 0.5).

    Args:
        rgb: float tensor [..., 3, H, W]

    Returns:
        yuv: float tensor [..., 3, H, W]
    """
    rgb = rgb.float()
    rgb_perm = rgb.movedim(-3, -1)  # [..., H, W, 3]

    mat = _RGB_TO_YUV_709.to(device=rgb.device, dtype=torch.float32)
    yuv = torch.matmul(rgb_perm, mat.T)  # [..., H, W, 3]

    # Offset U, V back to unsigned
    offset = torch.tensor([0.0, 0.5, 0.5], device=rgb.device, dtype=torch.float32)
    yuv = yuv + offset

    yuv = yuv.movedim(-1, -3)  # [..., 3, H, W]
    return yuv


def yuv_to_rgb_bt601(yuv: torch.Tensor) -> torch.Tensor:
    """
    Convert YUV [0,1] to RGB [0,1] using BT.601.
    U and V are assumed unsigned [0,1] and will be centered first.
    """
    yuv = yuv.float()
    yuv_perm = yuv.movedim(-3, -1)  # [..., H, W, 3]
    offset = torch.tensor([0.0, 0.5, 0.5], device=yuv.device, dtype=torch.float32)
    yuv_centered = yuv_perm - offset
    mat = _YUV_TO_RGB_601.to(device=yuv.device, dtype=torch.float32)
    rgb = torch.matmul(yuv_centered, mat.T)
    return rgb.movedim(-1, -3)


def rgb_to_yuv_bt601(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB [0,1] to YUV [0,1] using BT.601.
    Output U, V are unsigned [0,1] (offset by 0.5).
    """
    rgb = rgb.float()
    rgb_perm = rgb.movedim(-3, -1)  # [..., H, W, 3]
    mat = _RGB_TO_YUV_601.to(device=rgb.device, dtype=torch.float32)
    yuv = torch.matmul(rgb_perm, mat.T)
    offset = torch.tensor([0.0, 0.5, 0.5], device=rgb.device, dtype=torch.float32)
    yuv = yuv + offset
    return yuv.movedim(-1, -3)


def yuv_to_rgb_bt709_np(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    NumPy version: Convert YUV float32 [0,1] arrays to RGB float32 [0,1].
    U, V are unsigned [0,1]. Returns RGB [H, W, 3].
    """
    U_c = U - 0.5
    V_c = V - 0.5
    R = Y + 1.5748 * V_c
    G = Y - 0.1873 * U_c - 0.4681 * V_c
    B = Y + 1.8556 * U_c
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
