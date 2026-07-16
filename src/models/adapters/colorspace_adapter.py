"""
ColorSpaceAdapter: Non-learnable BT.709 YUV->RGB + ImageNet normalize.

Placed at the model entry point, before all backbone processing.
- VIF branch BYPASSES this (uses only Y channel directly).
- Detail and Semantic branches go through this adapter.
- FR mode: both ref and dis use the same adapter.
"""

import torch
import torch.nn as nn


# BT.709 YUV (with U,V centered) -> RGB matrix
# Y' = Y,  U' = U - 0.5,  V' = V - 0.5
# R = Y' + 1.5748 * V'
# G = Y' - 0.1873 * U' - 0.4681 * V'
# B = Y' + 1.8556 * U'
_YUV_TO_RGB_709 = torch.tensor([
    [1.0,  0.0000,  1.5748],
    [1.0, -0.1873, -0.4681],
    [1.0,  1.8556,  0.0000],
], dtype=torch.float32)

# BT.601 YUV (with U,V centered) -> RGB matrix
_YUV_TO_RGB_601 = torch.tensor([
    [1.0,  0.0000,  1.4020],
    [1.0, -0.344136, -0.714136],
    [1.0,  1.7720,  0.0000],
], dtype=torch.float32)

# ImageNet normalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# CLIP / OpenAI ViT normalization (also used by many CLIP-pretrained models)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)


class ColorSpaceAdapter(nn.Module):
    """
    Non-learnable module: YUV [0,1] -> BT.709 RGB -> ImageNet normalized.

    Input shapes: [..., 3, H, W] where channels are (Y, U, V) in [0, 1].
    Output shapes: [..., 3, H, W] with ImageNet normalization applied.

    Supports:
        - [B, 3, H, W]
        - [B, N, 3, H, W]
        - [B*T, 3, H, W]
        Any leading dims are OK.

    AMP safety: internally converts to float32 for matmul, then back.
    """

    def __init__(self, mode: str = 'bt709_imagenet', standard: str = None):
        super().__init__()
        # Backward-compatible alias: historical API used `standard='bt709'`.
        if standard is not None and (mode is None or mode == 'bt709_imagenet'):
            s = str(standard).strip().lower()
            if s in ('bt709', '709'):
                mode = 'bt709_imagenet'
            elif s in ('bt601', '601'):
                mode = 'bt601_imagenet'
            elif s in ('none', 'identity'):
                mode = 'none'
        self.mode = mode
        # Register as buffers so they move with model.to(device)
        mode_l = str(mode).lower()
        if mode_l.startswith('bt601'):
            mat = _YUV_TO_RGB_601
        else:
            mat = _YUV_TO_RGB_709
        self.register_buffer('yuv_to_rgb_mat', mat.clone(), persistent=False)
        # Select normalization constants based on mode suffix
        if mode_l.endswith('_clip'):
            norm_mean, norm_std = _CLIP_MEAN, _CLIP_STD
        else:
            norm_mean, norm_std = _IMAGENET_MEAN, _IMAGENET_STD
        self.register_buffer('norm_mean', norm_mean.clone(), persistent=False)
        self.register_buffer('norm_std', norm_std.clone(), persistent=False)
        self.register_buffer('uv_offset', torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def forward(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Convert YUV [0,1] to ImageNet-normalized RGB.

        Args:
            yuv: [..., 3, H, W] YUV float tensor

        Returns:
            rgb_norm: [..., 3, H, W] ImageNet-normalized RGB
        """
        mode_l = str(self.mode).lower()
        if mode_l in ('none', 'identity'):
            return yuv

        original_dtype = yuv.dtype

        # Work in float32 for numerical stability
        yuv = yuv.float()

        # [..., 3, H, W] -> [..., H, W, 3]
        yuv_hwc = yuv.movedim(-3, -1)

        # Center U, V: subtract [0, 0.5, 0.5]
        yuv_centered = yuv_hwc - self.uv_offset

        # Matrix multiply: [..., H, W, 3] @ [3, 3]^T -> [..., H, W, 3]
        rgb = torch.matmul(yuv_centered, self.yuv_to_rgb_mat.T)

        # Optional normalization by mode.
        if mode_l.endswith('_imagenet') or mode_l.endswith('_clip'):
            rgb_norm = (rgb - self.norm_mean) / self.norm_std
        elif mode_l.endswith('_raw'):
            rgb_norm = rgb
        else:
            # Backward compatible default
            rgb_norm = (rgb - self.norm_mean) / self.norm_std

        # Back to [..., 3, H, W]
        rgb_norm = rgb_norm.movedim(-1, -3)

        return rgb_norm.to(original_dtype)

    def forward_raw_rgb(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Convert YUV to RGB WITHOUT ImageNet normalization.
        Useful for debug visualization.

        Returns: [..., 3, H, W] in approximately [0, 1].
        """
        if self.mode == 'none':
            return yuv

        yuv = yuv.float()
        yuv_hwc = yuv.movedim(-3, -1)
        yuv_centered = yuv_hwc - self.uv_offset
        rgb = torch.matmul(yuv_centered, self.yuv_to_rgb_mat.T)
        rgb = rgb.movedim(-1, -3)
        return rgb.clamp(0, 1)

    def extra_repr(self):
        return f'mode={self.mode}'
