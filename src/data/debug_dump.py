"""
Debug frame dumping utility.
Saves YUV Y-channel, raw RGB, GMS patch RGB, and resize RGB for visual inspection.
Only rank0 saves.
"""

import os
import random
import logging
from typing import List, Optional, Dict

import torch
import numpy as np

logger = logging.getLogger('hmf_vqa.debug_dump')


def save_tensor_as_image(tensor: torch.Tensor, path: str):
    """Save a [C, H, W] or [H, W] tensor as PNG image."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available, skipping debug frame save")
        return

    t = tensor.detach().cpu().float()
    if t.dim() == 2:
        # Grayscale
        arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
    elif t.dim() == 3:
        if t.shape[0] == 1:
            arr = (t[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
        elif t.shape[0] == 3:
            arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode='RGB')
        else:
            return
    else:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def yuv_to_rgb_for_debug(yuv: torch.Tensor) -> torch.Tensor:
    """
    Convert YUV [0,1] to RGB [0,1] using BT.709 for debug visualization.
    Input: [..., 3, H, W], Output: [..., 3, H, W] clamped to [0,1].
    """
    mat = torch.tensor([
        [1.0,  0.0000,  1.5748],
        [1.0, -0.1873, -0.4681],
        [1.0,  1.8556,  0.0000],
    ], dtype=torch.float32)
    offset = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)

    yuv = yuv.float()
    yuv_hwc = yuv.movedim(-3, -1)  # [..., H, W, 3]
    yuv_centered = yuv_hwc - offset
    rgb = torch.matmul(yuv_centered, mat.T)
    rgb = rgb.movedim(-1, -3)  # [..., 3, H, W]
    return rgb.clamp(0, 1)


class DebugFrameDumper:
    """
    Saves debug frames (YUV Y-channel and RGB) for visual inspection.
    Only operates on rank0.
    """

    def __init__(
        self,
        save_dir: str,
        enabled: bool = True,
        num_videos_train: int = 2,
        num_videos_test: int = 2,
        num_frames: int = 1,
        every_epochs: int = 1,
        seed: int = 42,
        rank: int = 0,
    ):
        self.save_dir = save_dir
        self.enabled = enabled and (rank == 0)
        self.num_videos_train = num_videos_train
        self.num_videos_test = num_videos_test
        self.num_frames = num_frames
        self.every_epochs = every_epochs
        self.rng = random.Random(seed)
        self.rank = rank

    def should_save(self, epoch: int) -> bool:
        return self.enabled and (epoch % self.every_epochs == 0)

    def save_batch_debug(
        self,
        phase: str,  # 'train' or 'test'
        epoch_or_stage: str,
        dataset_name: str,
        video_ids: List[str],
        yuv_batch: torch.Tensor,  # [B, 3, T, H, W] or [B, 3, H, W]
        rgb_batch: Optional[torch.Tensor] = None,  # [B, 3, T, H, W] before ImageNet norm
        gms_batch: Optional[torch.Tensor] = None,   # [B, P, 3, T, ph, pw] GMS patches
        resize_batch: Optional[torch.Tensor] = None, # [B, 3, T, ts, ts] resized frames
        max_videos: Optional[int] = None,
    ):
        """
        Save debug frames for a batch.

        Saves:
          - yuv_y_f{i}.png:  Y-channel grayscale from original YUV
          - rgb_full_f{i}.png: Full-resolution RGB (YUV->BT.709 RGB, NO ImageNet norm)
          - gms_rgb_p{p}_f{i}.png: GMS patch as RGB (model detail branch input)
          - resize_rgb_f{i}.png: Resized frame as RGB (model semantic branch input)
        """
        if not self.enabled:
            return

        n_max = max_videos or (self.num_videos_train if phase == 'train' else self.num_videos_test)
        n_save = min(n_max, len(video_ids))

        if n_save <= 0:
            return

        # Pick subset
        indices = list(range(len(video_ids)))
        if len(indices) > n_save:
            indices = self.rng.sample(indices, n_save)

        saved_paths = []

        for idx in indices:
            vid = video_ids[idx]
            vid_safe = vid.replace('/', '_').replace('\\', '_').replace(' ', '_')
            base_dir = os.path.join(
                self.save_dir, 'debug_frames', phase, str(epoch_or_stage),
                dataset_name, vid_safe
            )

            yuv = yuv_batch[idx]  # [3, T, H, W] or [3, H, W]

            if yuv.dim() == 3:
                # Single frame [3, H, W]
                y_channel = yuv[0]  # [H, W]
                path_y = os.path.join(base_dir, 'yuv_y_f0.png')
                save_tensor_as_image(y_channel, path_y)
                saved_paths.append(path_y)

                # Convert full YUV to RGB for visualization
                rgb_full = yuv_to_rgb_for_debug(yuv)  # [3, H, W]
                path_rgb = os.path.join(base_dir, 'rgb_full_f0.png')
                save_tensor_as_image(rgb_full, path_rgb)
                saved_paths.append(path_rgb)

            elif yuv.dim() == 4:
                # Multiple frames [3, T, H, W]
                T = yuv.shape[1]
                frames_to_save = min(self.num_frames, T)
                frame_indices = list(range(0, T, max(1, T // frames_to_save)))[:frames_to_save]

                for fi, t in enumerate(frame_indices):
                    # Y channel
                    y_channel = yuv[0, t]  # [H, W]
                    path_y = os.path.join(base_dir, f'yuv_y_f{fi}.png')
                    save_tensor_as_image(y_channel, path_y)
                    saved_paths.append(path_y)

                    # Full-res RGB (YUV->RGB, no ImageNet norm)
                    yuv_frame = yuv[:, t]  # [3, H, W]
                    rgb_full = yuv_to_rgb_for_debug(yuv_frame)
                    path_rgb = os.path.join(base_dir, f'rgb_full_f{fi}.png')
                    save_tensor_as_image(rgb_full, path_rgb)
                    saved_paths.append(path_rgb)

            # ---- GMS patches (detail branch input) ----
            if gms_batch is not None and idx < gms_batch.shape[0]:
                gms = gms_batch[idx]  # [P, 3, T, ph, pw]
                P = gms.shape[0]
                T_gms = gms.shape[2]
                num_patches_to_save = min(P, 4)  # save up to 4 patches
                for p in range(num_patches_to_save):
                    # Save first frame of each patch
                    patch_yuv = gms[p, :, 0]  # [3, ph, pw]
                    patch_rgb = yuv_to_rgb_for_debug(patch_yuv)
                    path_p = os.path.join(base_dir, f'gms_rgb_p{p}_f0.png')
                    save_tensor_as_image(patch_rgb, path_p)
                    saved_paths.append(path_p)

            # ---- Resized frames (semantic branch input) ----
            if resize_batch is not None and idx < resize_batch.shape[0]:
                resize = resize_batch[idx]  # [3, T, ts, ts] or [P, 3, T, ts, ts]
                if resize.dim() == 5:
                    # Patch-stacked semantic input: visualize first patch.
                    resize = resize[0]
                T_rs = resize.shape[1]
                frames_to_save = min(self.num_frames, T_rs)
                frame_indices_rs = list(range(0, T_rs, max(1, T_rs // frames_to_save)))[:frames_to_save]
                for fi, t in enumerate(frame_indices_rs):
                    rs_yuv = resize[:, t]  # [3, ts, ts]
                    rs_rgb = yuv_to_rgb_for_debug(rs_yuv)
                    path_rs = os.path.join(base_dir, f'resize_rgb_f{fi}.png')
                    save_tensor_as_image(rs_rgb, path_rs)
                    saved_paths.append(path_rs)

        if saved_paths:
            logger.info(f"[DebugDump] Saved {len(saved_paths)} debug frames to {os.path.dirname(saved_paths[0])}")
