"""
VideoReaderFactory: auto-detect raw vs container and produce unified YUV output.

Output contract: YUV float32 [3, T, H, W] in [0, 1], channel order (Y, U, V).
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..yuv.yuv_reader import (
    read_yuv420_frames,
    read_yuv420_frames_y,
    read_yuv420_frames_rgb_legacy,
    get_yuv_frame_count,
)
from ..yuv.yuv_align import align_yuv
from ..yuv.colorspace import rgb_to_yuv_bt709, rgb_to_yuv_bt601
from .container_reader import (
    read_container_pyav,
    read_container_decord,
    read_container_ffmpeg,
    read_container_ffmpeg_yuv,
    get_container_frame_count,
    _probe_video_info,
)

logger = logging.getLogger('hmf_vqa.video_reader')

RAW_EXTENSIONS = {'.yuv', '.y4m', '.nv12', '.raw'}
CONTAINER_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.flv', '.ts', '.m4v'}


def is_raw_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in RAW_EXTENSIONS


def is_container_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in CONTAINER_EXTENSIONS


class VideoReaderFactory:
    """
    Unified video reading interface.
    Auto-detects raw YUV vs container and outputs YUV float32 [3, T, H, W] in [0,1].
    """

    def __init__(
        self,
        container_decoder: str = 'pyav',
        auto_probe: bool = True,
        default_width: Optional[int] = None,
        default_height: Optional[int] = None,
        default_bitdepth: int = 10,
        default_pix_fmt: str = 'yuv420p10le',
        default_signal_range: str = 'auto',
        default_tenbit_mode: str = 'normalize',
        uv_upsample: str = 'bicubic',
        raw_yuv_backend: str = 'native',
        raw_yuv_matrix: str = 'bt709',
        container_yuv_matrix: str = 'bt709',
        container_yuv_direct: bool = False,
    ):
        self.container_decoder = container_decoder
        self.auto_probe = auto_probe
        self.default_width = default_width
        self.default_height = default_height
        self.default_bitdepth = default_bitdepth
        self.default_pix_fmt = default_pix_fmt
        self.default_signal_range = str(default_signal_range or 'auto')
        self.default_tenbit_mode = str(default_tenbit_mode or 'shift8')
        self.uv_upsample = uv_upsample
        self.raw_yuv_backend = str(raw_yuv_backend or 'native').lower().strip()
        if self.raw_yuv_backend not in ('native', 'legacy_cv2'):
            self.raw_yuv_backend = 'native'
        self.raw_yuv_matrix = str(raw_yuv_matrix or 'bt709').lower().strip()
        if self.raw_yuv_matrix not in ('bt709', 'bt601'):
            self.raw_yuv_matrix = 'bt709'
        self.container_yuv_matrix = str(container_yuv_matrix or 'bt709').lower().strip()
        if self.container_yuv_matrix not in ('bt709', 'bt601'):
            self.container_yuv_matrix = 'bt709'
        self.container_yuv_direct = bool(container_yuv_direct)

    def get_frame_count(
        self,
        path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bitdepth: Optional[int] = None,
    ) -> int:
        """Get total frame count for any video file."""
        if is_raw_video(path):
            w = width or self.default_width
            h = height or self.default_height
            bd = bitdepth or self.default_bitdepth
            if w is None or h is None:
                logger.warning(f"Cannot determine frame count for raw file {path}: missing w/h")
                return 0
            return get_yuv_frame_count(path, w, h, bd)
        else:
            return get_container_frame_count(path, self.container_decoder)

    def read_frames(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        bitdepth: Optional[int] = None,
        pix_fmt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Read specified frames and return YUV float32 [3, T, H, W] in [0,1].

        For raw YUV: reads directly as YUV.
        For containers: decodes to RGB then converts to YUV via BT.709.
        """
        if is_raw_video(path):
            return self._read_raw(path, frame_indices, width, height, bitdepth, pix_fmt)
        elif is_container_video(path) or not is_raw_video(path):
            return self._read_container(path, frame_indices, width, height, bitdepth, pix_fmt)
        else:
            raise ValueError(f"Unsupported video format: {path}")

    def read_frames_y(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        bitdepth: Optional[int] = None,
        pix_fmt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Read specified frames and return only Y channel [1, T, H, W] in [0,1].
        """
        if is_raw_video(path):
            return self._read_raw_y(path, frame_indices, width, height, bitdepth, pix_fmt)
        elif is_container_video(path) or not is_raw_video(path):
            yuv = self._read_container(path, frame_indices, width, height, bitdepth, pix_fmt)
            return yuv[0:1].contiguous()
        else:
            raise ValueError(f"Unsupported video format: {path}")

    def read_frames_raw_yuv420(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        bitdepth: Optional[int] = None,
        pix_fmt: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read YUV420 frames and return the *unpacked* (Y, U_half, V_half)
        tensors, i.e. WITHOUT the bicubic UV upsample + stack that
        ``read_frames`` performs via ``align_yuv``.

        This exists to support GPU-side preprocess (see
        ``src/data/gpu_preprocess.py``) which prefers to receive raw YUV420
        planes and do UV upsample on-device.

        Only supports the native raw-YUV path for now — the fast-path used by
        CVQM eval.  For containers / legacy_cv2 backends, falls back to
        ``read_frames`` and strides-down UV from the aligned YUV tensor, which
        is NOT bit-exact; callers should not enable GPU preprocess on those
        paths.

        Returns
        -------
        Y       : torch.FloatTensor [T, H, W],         in [0, 1]
        U_half  : torch.FloatTensor [T, H // 2, W // 2], in [0, 1]
        V_half  : torch.FloatTensor [T, H // 2, W // 2], in [0, 1]
        """
        if is_raw_video(path) and self.raw_yuv_backend == 'native':
            w = width or self.default_width
            h = height or self.default_height
            bd = bitdepth or self.default_bitdepth
            pf = pix_fmt or self.default_pix_fmt
            if w is None or h is None:
                raise ValueError(
                    f"Width/height required for raw YUV file {path}. "
                    f"Provide via SampleMeta or CLI (--width, --height)."
                )
            Y_np, U_np, V_np = read_yuv420_frames(
                file_path=path,
                width=w,
                height=h,
                frame_indices=frame_indices,
                bitdepth=bd,
                pix_fmt=pf,
                signal_range=self.default_signal_range,
                tenbit_mode=self.default_tenbit_mode,
            )
            Y_t = Y_np if isinstance(Y_np, torch.Tensor) else torch.from_numpy(np.asarray(Y_np, dtype=np.float32))
            U_t = U_np if isinstance(U_np, torch.Tensor) else torch.from_numpy(np.asarray(U_np, dtype=np.float32))
            V_t = V_np if isinstance(V_np, torch.Tensor) else torch.from_numpy(np.asarray(V_np, dtype=np.float32))
            return Y_t.float(), U_t.float(), V_t.float()

        # Fallback: use the regular aligned-YUV path, then stride-down U/V.
        # WARNING: this is NOT bit-exact with the aligned path the rest of
        # the dataloader uses, so callers should only enable GPU preprocess
        # for paths where the native raw-YUV branch above applies.
        yuv = self.read_frames(path, frame_indices, width, height, bitdepth, pix_fmt)
        Y_t = yuv[0].contiguous()
        U_t = yuv[1, :, ::2, ::2].contiguous()
        V_t = yuv[2, :, ::2, ::2].contiguous()
        return Y_t, U_t, V_t


    def _read_raw(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
    ) -> torch.Tensor:
        """Read raw YUV and return [3, T, H, W]."""
        w = width or self.default_width
        h = height or self.default_height
        bd = bitdepth or self.default_bitdepth
        pf = pix_fmt or self.default_pix_fmt

        if w is None or h is None:
            raise ValueError(
                f"Width/height required for raw YUV file {path}. "
                f"Provide via SampleMeta or CLI (--width, --height)."
            )

        # Full legacy path used by old PE dataset loader:
        # raw yuv420(10bit)->shift8->cv2 I420 RGB, then convert to YUV proxy.
        if self.raw_yuv_backend == 'legacy_cv2' and self.default_tenbit_mode == 'shift8':
            rgb_np = read_yuv420_frames_rgb_legacy(
                file_path=path,
                width=w,
                height=h,
                frame_indices=frame_indices,
                bitdepth=bd,
                pix_fmt=pf,
                tenbit_mode=self.default_tenbit_mode,
            )  # [T,H,W,3], [0,1]
            rgb_tensor = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2)  # [T,3,H,W]
            if self.raw_yuv_matrix == 'bt601':
                yuv_tensor = rgb_to_yuv_bt601(rgb_tensor)
            else:
                yuv_tensor = rgb_to_yuv_bt709(rgb_tensor)
            return yuv_tensor.permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)

        Y, U, V = read_yuv420_frames(
            file_path=path,
            width=w,
            height=h,
            frame_indices=frame_indices,
            bitdepth=bd,
            pix_fmt=pf,
            signal_range=self.default_signal_range,
            tenbit_mode=self.default_tenbit_mode,
        )
        # Y: [T, H, W], U: [T, H/2, W/2], V: [T, H/2, W/2]
        yuv = align_yuv(Y, U, V, self.uv_upsample)  # [3, T, H, W]
        return yuv

    def _read_raw_y(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
    ) -> torch.Tensor:
        """Read raw YUV and return only Y [1, T, H, W]."""
        w = width or self.default_width
        h = height or self.default_height
        bd = bitdepth or self.default_bitdepth
        pf = pix_fmt or self.default_pix_fmt

        if w is None or h is None:
            raise ValueError(
                f"Width/height required for raw YUV file {path}. "
                f"Provide via SampleMeta or CLI (--width, --height)."
            )

        Y = read_yuv420_frames_y(
            file_path=path,
            width=w,
            height=h,
            frame_indices=frame_indices,
            bitdepth=bd,
            pix_fmt=pf,
            signal_range=self.default_signal_range,
            tenbit_mode=self.default_tenbit_mode,
        )
        return torch.from_numpy(Y).float().unsqueeze(0)  # [1,T,H,W]

    def _read_container(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
    ) -> torch.Tensor:
        """Read container video and return YUV [3, T, H, W] in [0,1].

        When container_yuv_direct=True, ffmpeg outputs raw YUV420 directly,
        avoiding the double color-space conversion (YUV→RGB→YUV).
        Otherwise, falls back to the original RGB decode + rgb_to_yuv path.
        """
        # --- YUV direct path: ffmpeg outputs YUV420p/YUV420p10le ---
        if self.container_yuv_direct:
            Y_np, U_np, V_np, h, w = read_container_ffmpeg_yuv(
                path, frame_indices, width, height,
                pix_fmt_hint=pix_fmt,
                uv_upsample=self.uv_upsample,
            )
            # Y: [T, H, W], U: [T, H/2, W/2], V: [T, H/2, W/2]
            yuv = align_yuv(Y_np, U_np, V_np, self.uv_upsample)  # [3, T, H, W]
            return yuv.clamp(0, 1)

        # --- Original RGB path: ffmpeg→RGB then rgb_to_yuv ---
        import os as _os
        _decoder_override = _os.environ.get('HMF_CONTAINER_DECODER', '').strip().lower()
        decoder = _decoder_override if _decoder_override else self.container_decoder
        # Default to ffmpeg when num_workers > 0 (fork mode)
        if decoder == 'pyav' and not _decoder_override:
            decoder = 'ffmpeg'

        if decoder == 'pyav':
            rgb_np, h, w = read_container_pyav(path, frame_indices, pix_fmt_hint=pix_fmt)
        elif decoder == 'decord':
            rgb_np, h, w = read_container_decord(path, frame_indices)
        elif decoder == 'ffmpeg':
            rgb_np, h, w = read_container_ffmpeg(
                path, frame_indices, width, height, pix_fmt_hint=pix_fmt,
            )
        else:
            raise ValueError(f"Unknown decoder: {decoder}")

        # rgb_np: [T, H, W, 3] float32 in [0, 1]
        # Convert to torch: [T, 3, H, W]
        rgb_tensor = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2)

        # Convert RGB -> YUV with configurable matrix.
        if self.container_yuv_matrix == 'bt601':
            yuv_tensor = rgb_to_yuv_bt601(rgb_tensor)
        else:
            yuv_tensor = rgb_to_yuv_bt709(rgb_tensor)

        # Rearrange to [3, T, H, W]
        yuv_tensor = yuv_tensor.permute(1, 0, 2, 3)  # [3, T, H, W]
        return yuv_tensor.clamp(0, 1)
