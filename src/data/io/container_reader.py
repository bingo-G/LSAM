"""
Container video reader: reads mp4/mkv/mov/webm via PyAV or ffmpeg subprocess.
Always outputs YUV float32 [0,1] tensors to match the unified pipeline.
"""

import os
import re
import subprocess
import logging
import time
from functools import lru_cache
import numpy as np
import torch
from typing import List, Optional, Tuple

_logger = logging.getLogger('hmf_vqa.container_reader')

# Files larger than this threshold (MB) use ffmpeg subprocess instead of PyAV
# to avoid PyAV + fork + NFS deadlock on large 4K videos over NAS.
# Set to 0 to ALWAYS use ffmpeg (recommended for NAS/NFS environments).
# Override via env: HMF_PYAV_SIZE_LIMIT_MB=0 (default 0, i.e., always ffmpeg)
_PYAV_SIZE_LIMIT_MB = float(os.environ.get('HMF_PYAV_SIZE_LIMIT_MB', '0'))

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

try:
    from decord import VideoReader as DecordVideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


@lru_cache(maxsize=4096)
def _probe_video_info_cached(path: str) -> Tuple[int, int, int, str, str, float]:
    """Use ffprobe to get cached video metadata."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            stream = info.get('streams', [{}])[0]
            return (
                int(stream.get('width', 0)),
                int(stream.get('height', 0)),
                int(stream.get('nb_frames', 0)) if stream.get('nb_frames') else 0,
                str(stream.get('codec_name', '') or ''),
                str(stream.get('pix_fmt', '') or ''),
                float(stream.get('duration', 0)) if stream.get('duration') else 0.0,
            )
    except Exception:
        pass
    return (0, 0, 0, '', '', 0.0)


def _probe_video_info(path: str) -> dict:
    """Use ffprobe to get video metadata."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    width, height, nb_frames, codec, pix_fmt, duration = _probe_video_info_cached(abs_path)
    return {
        'width': width,
        'height': height,
        'nb_frames': nb_frames,
        'codec': codec,
        'pix_fmt': pix_fmt,
        'duration': duration,
    }


def _is_high_bitdepth_pix_fmt(pix_fmt: Optional[str]) -> bool:
    pix_fmt = str(pix_fmt or '').strip().lower()
    if not pix_fmt:
        return False
    return re.search(r'(?:p|gray|gbrp)(10|12|14|16)(?:le|be)?$', pix_fmt) is not None


def _container_rgb_format(pix_fmt_hint: Optional[str], path: str) -> Tuple[str, np.dtype, float, int]:
    pix_fmt = str(pix_fmt_hint or '').strip().lower()
    if not pix_fmt:
        pix_fmt = str(_probe_video_info(path).get('pix_fmt', '') or '').strip().lower()
    if _is_high_bitdepth_pix_fmt(pix_fmt):
        return 'rgb48le', np.uint16, 65535.0, 2
    return 'rgb24', np.uint8, 255.0, 1


def read_container_pyav(
    path: str,
    frame_indices: List[int],
    pix_fmt_hint: Optional[str] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Read frames using PyAV. Returns RGB float32 [T, H, W, 3] in [0, 1].
    For large files (>_PYAV_SIZE_LIMIT_MB), automatically falls back to ffmpeg
    subprocess to avoid PyAV + multiprocessing fork + NFS deadlock.
    """
    if not HAS_PYAV:
        raise ImportError("PyAV is required for container reading. Install: pip install av")

    # Auto-fallback: large files over NAS are prone to PyAV+fork deadlock
    try:
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        file_size_mb = 0
    if file_size_mb > _PYAV_SIZE_LIMIT_MB:
        _logger.info(
            "[PyAV→ffmpeg] Large file (%.0fMB > %.0fMB limit), using ffmpeg subprocess: %s",
            file_size_mb, _PYAV_SIZE_LIMIT_MB, os.path.basename(path),
        )
        return read_container_ffmpeg(path, frame_indices, pix_fmt_hint=pix_fmt_hint)

    t_open_start = time.monotonic()
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    t_open = time.monotonic() - t_open_start

    frames_dict = {}
    target_set = set(frame_indices)
    max_idx = max(frame_indices) if frame_indices else 0
    rgb_fmt, _dtype, scale, _bytes_per_chan = _container_rgb_format(pix_fmt_hint, path)

    # Warn if opening was slow (NAS latency)
    if t_open > 5.0:
        _logger.warning(
            "[PyAV] Slow open (%.1fs): %s", t_open, os.path.basename(path),
        )

    t_decode_start = time.monotonic()
    t_last_log = t_decode_start
    SLOW_FRAME_THRESHOLD = 2.0  # warn if a single frame takes >2s
    STALL_LOG_INTERVAL = 30.0   # log progress every 30s if decode is slow

    t_frame_start = time.monotonic()
    for i, frame in enumerate(container.decode(video=0)):
        t_now = time.monotonic()
        frame_elapsed = t_now - t_frame_start

        if i in target_set:
            img = frame.to_ndarray(format=rgb_fmt).astype(np.float32) / scale
            frames_dict[i] = img

        # Log slow frames and periodic progress for long decodes
        if frame_elapsed > SLOW_FRAME_THRESHOLD and i > 0:
            if t_now - t_last_log > STALL_LOG_INTERVAL:
                total_elapsed = t_now - t_decode_start
                _logger.warning(
                    "[PyAV] Slow decode: frame %d/%d (%.1fs total, %.1fs/frame) %s",
                    i, max_idx, total_elapsed, frame_elapsed, os.path.basename(path),
                )
                t_last_log = t_now

        if i >= max_idx:
            break
        t_frame_start = time.monotonic()

    t_decode_total = time.monotonic() - t_decode_start
    container.close()

    # Warn if total decode was slow
    if t_decode_total > 10.0:
        _logger.warning(
            "[PyAV] Decode took %.1fs (%d frames, %.2fs/frame): %s",
            t_decode_total, max_idx + 1, t_decode_total / max(1, max_idx + 1),
            os.path.basename(path),
        )

    # Assemble in order, handle missing frames
    H, W = 0, 0
    result = []
    for idx in frame_indices:
        if idx in frames_dict:
            result.append(frames_dict[idx])
            H, W = frames_dict[idx].shape[:2]
        else:
            if result:
                result.append(result[-1].copy())  # repeat last
            else:
                # If first frame is missing, try to get any available
                if frames_dict:
                    fallback = list(frames_dict.values())[0]
                    result.append(fallback.copy())
                    H, W = fallback.shape[:2]
                else:
                    raise RuntimeError(f"No frames read from {path}")

    return np.stack(result), H, W


def read_container_decord(
    path: str,
    frame_indices: List[int],
) -> Tuple[np.ndarray, int, int]:
    """
    Read frames using Decord. Returns RGB float32 [T, H, W, 3] in [0, 1].
    """
    if not HAS_DECORD:
        raise ImportError("Decord is required. Install: pip install decord")

    vr = DecordVideoReader(path, ctx=cpu(0))
    total = len(vr)

    # Clamp indices
    indices = [min(idx, total - 1) for idx in frame_indices]
    frames = vr.get_batch(indices)  # [T, H, W, 3] uint8

    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    frames_np = frames.astype(np.float32) / 255.0

    H, W = frames_np.shape[1], frames_np.shape[2]
    return frames_np, H, W


def read_container_ffmpeg(
    path: str,
    frame_indices: List[int],
    width: Optional[int] = None,
    height: Optional[int] = None,
    pix_fmt_hint: Optional[str] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Read frames using ffmpeg subprocess. Returns RGB float32 [T, H, W, 3] in [0, 1].
    """
    # ALWAYS probe container videos for true resolution.
    # The caller may pass dis resolution for a ref video (e.g., Waterloo4K:
    # dis=540p/1080p but ref is always 4K). ffprobe gives the actual resolution.
    info = _probe_video_info(path)
    probed_w = info.get('width', 0)
    probed_h = info.get('height', 0)
    if probed_w > 0 and probed_h > 0:
        width = probed_w
        height = probed_h
    elif width is None or height is None:
        width = width or 1920
        height = height or 1080

    rgb_fmt, dtype, scale, bytes_per_chan = _container_rgb_format(pix_fmt_hint, path)

    cmd = [
        'ffmpeg', '-i', path,
        '-f', 'rawvideo', '-pix_fmt', rgb_fmt,
        '-v', 'quiet', '-'
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = height * width * 3 * bytes_per_chan
    frames_dict = {}
    target_set = set(frame_indices)
    max_idx = max(frame_indices) if frame_indices else 0

    t_start = time.monotonic()
    idx = 0
    while idx <= max_idx:
        raw = process.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        if idx in target_set:
            frame = np.frombuffer(raw, dtype=dtype).reshape(height, width, 3)
            frames_dict[idx] = frame.astype(np.float32) / scale
        idx += 1

    process.stdout.close()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        _logger.warning("[ffmpeg] Process killed after timeout: %s", os.path.basename(path))

    t_elapsed = time.monotonic() - t_start
    if t_elapsed > 10.0:
        _logger.warning(
            "[ffmpeg] Decode took %.1fs (%d frames, %dx%d): %s",
            t_elapsed, idx, width, height, os.path.basename(path),
        )

    result = []
    for i in frame_indices:
        if i in frames_dict:
            result.append(frames_dict[i])
        elif result:
            result.append(result[-1].copy())
        elif frames_dict:
            result.append(list(frames_dict.values())[0].copy())

    if not result:
        raise RuntimeError(f"No frames decoded from {path}")

    return np.stack(result), height, width


def _container_yuv_format(pix_fmt_hint: Optional[str], path: str) -> Tuple[str, np.dtype, float, int]:
    """Determine ffmpeg output YUV format based on source bit depth.

    Returns:
        (ffmpeg_pix_fmt, numpy_dtype, scale_factor, bytes_per_sample)
    """
    pix_fmt = str(pix_fmt_hint or '').strip().lower()
    if not pix_fmt:
        pix_fmt = str(_probe_video_info(path).get('pix_fmt', '') or '').strip().lower()
    if _is_high_bitdepth_pix_fmt(pix_fmt):
        return 'yuv420p10le', np.uint16, 1023.0, 2
    return 'yuv420p', np.uint8, 255.0, 1


def read_container_ffmpeg_yuv(
    path: str,
    frame_indices: List[int],
    width: Optional[int] = None,
    height: Optional[int] = None,
    pix_fmt_hint: Optional[str] = None,
    uv_upsample: str = 'bicubic',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Read frames using ffmpeg subprocess, directly outputting YUV420 planes.

    This avoids the double color-space conversion (ffmpeg YUV→RGB + rgb_to_yuv)
    by letting ffmpeg output raw YUV420p/YUV420p10le, then normalizing to [0,1].

    A ``scale=in_range=tv:out_range=pc`` filter is applied so the output
    matches the full-range [0,1] values that the RGB decode path produces
    (ffmpeg automatically converts limited→full when decoding to RGB).

    Returns:
        Y:  float32 [T, H, W]     in [0, 1]
        U:  float32 [T, H/2, W/2] in [0, 1]
        V:  float32 [T, H/2, W/2] in [0, 1]
        H:  int
        W:  int
    """
    # Probe actual resolution
    info = _probe_video_info(path)
    probed_w = info.get('width', 0)
    probed_h = info.get('height', 0)
    if probed_w > 0 and probed_h > 0:
        width = probed_w
        height = probed_h
    elif width is None or height is None:
        width = width or 1920
        height = height or 1080

    yuv_fmt, dtype, scale, bytes_per_sample = _container_yuv_format(pix_fmt_hint, path)

    # Use scale filter to convert limited range (tv) → full range (pc).
    # This matches ffmpeg's implicit behavior when decoding to RGB.
    cmd = [
        'ffmpeg', '-i', path,
        '-vf', 'scale=in_range=tv:out_range=pc',
        '-f', 'rawvideo', '-pix_fmt', yuv_fmt,
        '-v', 'quiet', '-'
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    h, w = height, width
    h2, w2 = h // 2, w // 2
    y_size = h * w
    uv_size = h2 * w2
    frame_size_bytes = (y_size + 2 * uv_size) * bytes_per_sample

    Ys_dict = {}
    Us_dict = {}
    Vs_dict = {}
    target_set = set(frame_indices)
    max_idx = max(frame_indices) if frame_indices else 0

    t_start = time.monotonic()
    idx = 0
    while idx <= max_idx:
        raw = process.stdout.read(frame_size_bytes)
        if len(raw) < frame_size_bytes:
            break
        if idx in target_set:
            data = np.frombuffer(raw, dtype=dtype)
            Y = (data[:y_size].reshape(h, w).astype(np.float32) / scale)
            U = (data[y_size:y_size + uv_size].reshape(h2, w2).astype(np.float32) / scale)
            V = (data[y_size + uv_size:].reshape(h2, w2).astype(np.float32) / scale)
            Ys_dict[idx] = Y
            Us_dict[idx] = U
            Vs_dict[idx] = V
        idx += 1

    process.stdout.close()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        _logger.warning("[ffmpeg-yuv] Process killed after timeout: %s", os.path.basename(path))

    t_elapsed = time.monotonic() - t_start
    if t_elapsed > 10.0:
        _logger.warning(
            "[ffmpeg-yuv] Decode took %.1fs (%d frames, %dx%d): %s",
            t_elapsed, idx, width, height, os.path.basename(path),
        )

    # Assemble in order
    Ys, Us, Vs = [], [], []
    for i in frame_indices:
        if i in Ys_dict:
            Ys.append(Ys_dict[i])
            Us.append(Us_dict[i])
            Vs.append(Vs_dict[i])
        elif Ys:
            Ys.append(Ys[-1].copy())
            Us.append(Us[-1].copy())
            Vs.append(Vs[-1].copy())
        elif Ys_dict:
            first_key = next(iter(Ys_dict))
            Ys.append(Ys_dict[first_key].copy())
            Us.append(Us_dict[first_key].copy())
            Vs.append(Vs_dict[first_key].copy())

    if not Ys:
        raise RuntimeError(f"No frames decoded from {path}")

    return np.stack(Ys), np.stack(Us), np.stack(Vs), height, width


def get_container_frame_count(path: str, decoder: str = 'pyav') -> int:
    """Get total frame count of a container video."""
    info = _probe_video_info(path)
    if int(info.get('nb_frames', 0) or 0) > 0:
        return int(info['nb_frames'])

    if decoder == 'pyav' and HAS_PYAV:
        try:
            container = av.open(path)
            stream = container.streams.video[0]
            count = stream.frames
            if count == 0:
                count = sum(1 for _ in container.decode(video=0))
            container.close()
            return count
        except Exception:
            pass
    elif decoder == 'decord' and HAS_DECORD:
        try:
            vr = DecordVideoReader(path, ctx=cpu(0))
            return len(vr)
        except Exception:
            pass

    # Fallback: ffprobe
    return info.get('nb_frames', 0)
