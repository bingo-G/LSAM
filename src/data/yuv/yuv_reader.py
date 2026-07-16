"""
YUV Reader: Reads raw YUV files (8-bit and 10-bit, planar format).
Supports: yuv420p, yuv420p10le
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple, List
import cv2


def _normalize_raw_args(
    bitdepth: int,
    bit_depth: Optional[int],
    pix_fmt: str,
) -> Tuple[int, str]:
    """
    Backward-compatible arg normalization.
    Handles accidental positional call style where pix_fmt is passed
    into bit_depth slot (e.g., bit_depth='yuv420p10le').
    """
    if isinstance(bit_depth, str):
        # Legacy bug path: read_xxx(..., bitdepth, pix_fmt)
        if pix_fmt == 'yuv420p10le':
            pix_fmt = bit_depth
        bit_depth = None

    if bit_depth is not None:
        bitdepth = int(bit_depth)

    return int(bitdepth), str(pix_fmt)


def _resolve_signal_range(pix_fmt: str, signal_range: str) -> str:
    sr = str(signal_range or 'auto').strip().lower()
    if sr in ('limited', 'tv', 'mpeg'):
        return 'limited'
    if sr in ('full', 'pc', 'jpeg'):
        return 'full'
    pf = str(pix_fmt or '').strip().lower()
    # yuvj* or explicit full-range tags imply full range.
    if ('yuvj' in pf) or ('full' in pf):
        return 'full'
    # Most raw yuv420p/yuv420p10le files are studio/limited range.
    return 'limited'


def _range_params(bitdepth: int, pix_fmt: str, signal_range: str) -> Tuple[float, float, float, float]:
    max_val = float((1 << int(bitdepth)) - 1)
    mode = _resolve_signal_range(pix_fmt, signal_range)
    if mode == 'full':
        return 0.0, max_val, 0.0, max_val

    scale = float(1 << max(int(bitdepth) - 8, 0))
    y_min = 16.0 * scale
    y_max = 235.0 * scale
    c_min = 16.0 * scale
    c_max = 240.0 * scale
    return y_min, max(y_max - y_min, 1.0), c_min, max(c_max - c_min, 1.0)


def read_yuv420_frame(
    file_path: str,
    width: int,
    height: int,
    frame_idx: int,
    bitdepth: int = 10,
    bit_depth: Optional[int] = None,
    pix_fmt: str = 'yuv420p10le',
    signal_range: str = 'auto',
    tenbit_mode: str = 'shift8',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a single YUV420 frame from a raw file.

    Returns:
        Y: float32 [H, W] in [0, 1]
        U: float32 [H/2, W/2] in [0, 1]
        V: float32 [H/2, W/2] in [0, 1]
    """
    bitdepth, _ = _normalize_raw_args(bitdepth, bit_depth, pix_fmt)

    h, w = height, width
    h2, w2 = h // 2, w // 2

    if bitdepth > 8:
        # 10-bit stored as 16-bit little-endian
        bytes_per_sample = 2
        dtype = np.uint16
    else:
        bytes_per_sample = 1
        dtype = np.uint8

    y_size = h * w
    u_size = h2 * w2
    v_size = h2 * w2
    frame_size_samples = y_size + u_size + v_size
    frame_size_bytes = frame_size_samples * bytes_per_sample

    offset = frame_idx * frame_size_bytes

    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw = f.read(frame_size_bytes)

    if len(raw) < frame_size_bytes:
        raise ValueError(
            f"Incomplete frame at index {frame_idx} in {file_path}. "
            f"Expected {frame_size_bytes} bytes, got {len(raw)}."
        )

    data = np.frombuffer(raw, dtype=dtype)

    mode = str(tenbit_mode or 'normalize').lower().strip()
    if bitdepth > 8 and mode == 'shift8':
        shift = max(int(bitdepth) - 8, 0)
        Y = ((data[:y_size].reshape(h, w).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
        U = ((data[y_size:y_size + u_size].reshape(h2, w2).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
        V = ((data[y_size + u_size:].reshape(h2, w2).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
    else:
        y_off, y_den, c_off, c_den = _range_params(bitdepth, pix_fmt, signal_range)
        Y = ((data[:y_size].reshape(h, w).astype(np.float32) - y_off) / y_den).clip(0.0, 1.0)
        U = ((data[y_size:y_size + u_size].reshape(h2, w2).astype(np.float32) - c_off) / c_den).clip(0.0, 1.0)
        V = ((data[y_size + u_size:].reshape(h2, w2).astype(np.float32) - c_off) / c_den).clip(0.0, 1.0)

    return Y, U, V


def read_yuv420_luma_frame(
    file_path: str,
    width: int,
    height: int,
    frame_idx: int,
    bitdepth: int = 10,
    bit_depth: Optional[int] = None,
    pix_fmt: str = 'yuv420p10le',
    signal_range: str = 'auto',
    tenbit_mode: str = 'shift8',
) -> np.ndarray:
    """
    Read only the Y plane from a single YUV420 frame.

    Returns:
        Y: float32 [H, W] in [0, 1]
    """
    bitdepth, _ = _normalize_raw_args(bitdepth, bit_depth, pix_fmt)

    h, w = height, width
    h2, w2 = h // 2, w // 2

    if bitdepth > 8:
        bytes_per_sample = 2
        dtype = np.uint16
    else:
        bytes_per_sample = 1
        dtype = np.uint8

    y_size = h * w
    u_size = h2 * w2
    v_size = h2 * w2
    frame_size_samples = y_size + u_size + v_size
    frame_size_bytes = frame_size_samples * bytes_per_sample
    y_bytes = y_size * bytes_per_sample

    offset = frame_idx * frame_size_bytes

    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw_y = f.read(y_bytes)

    if len(raw_y) < y_bytes:
        raise ValueError(
            f"Incomplete frame at index {frame_idx} in {file_path}. "
            f"Expected {y_bytes} Y bytes, got {len(raw_y)}."
        )

    data = np.frombuffer(raw_y, dtype=dtype)
    mode = str(tenbit_mode or 'shift8').lower().strip()
    if bitdepth > 8 and mode == 'shift8':
        shift = max(int(bitdepth) - 8, 0)
        Y = ((data.reshape(h, w).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
    else:
        y_off, y_den, _c_off, _c_den = _range_params(bitdepth, pix_fmt, signal_range)
        Y = ((data.reshape(h, w).astype(np.float32) - y_off) / y_den).clip(0.0, 1.0)
    return Y


def get_yuv_frame_count(
    file_path: str,
    width: int,
    height: int,
    bitdepth: int = 10,
    bit_depth: Optional[int] = None,
) -> int:
    """Get total number of frames in a raw YUV file."""
    bitdepth, _ = _normalize_raw_args(bitdepth, bit_depth, 'yuv420p10le')
    h, w = height, width
    h2, w2 = h // 2, w // 2
    bytes_per_sample = 2 if bitdepth > 8 else 1
    frame_size = (h * w + h2 * w2 * 2) * bytes_per_sample
    file_size = os.path.getsize(file_path)
    return file_size // frame_size


def _is_consecutive(indices: List[int]) -> bool:
    """Check if frame indices form a consecutive sequence."""
    if len(indices) <= 1:
        return True
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            return False
    return True


def _decode_yuv_frame(
    raw: bytes,
    offset_in_raw: int,
    frame_size_bytes: int,
    h: int, w: int, h2: int, w2: int,
    y_size: int, u_size: int,
    dtype: type,
    bitdepth: int,
    tenbit_mode: str,
    pix_fmt: str,
    signal_range: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a single YUV420 frame from a raw buffer at the given byte offset."""
    data = np.frombuffer(raw, dtype=dtype,
                         offset=offset_in_raw, count=y_size + u_size + u_size)

    mode = str(tenbit_mode or 'normalize').lower().strip()
    if bitdepth > 8 and mode == 'shift8':
        shift = max(int(bitdepth) - 8, 0)
        Y = ((data[:y_size].reshape(h, w).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
        U = ((data[y_size:y_size + u_size].reshape(h2, w2).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
        V = ((data[y_size + u_size:].reshape(h2, w2).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
    else:
        y_off, y_den, c_off, c_den = _range_params(bitdepth, pix_fmt, signal_range)
        Y = ((data[:y_size].reshape(h, w).astype(np.float32) - y_off) / y_den).clip(0.0, 1.0)
        U = ((data[y_size:y_size + u_size].reshape(h2, w2).astype(np.float32) - c_off) / c_den).clip(0.0, 1.0)
        V = ((data[y_size + u_size:].reshape(h2, w2).astype(np.float32) - c_off) / c_den).clip(0.0, 1.0)

    return Y, U, V


def _bulk_decode_yuv_frames(
    raw: bytes,
    n: int,
    h: int, w: int, h2: int, w2: int,
    y_size: int, u_size: int,
    dtype: type,
    bitdepth: int,
    tenbit_mode: str,
    pix_fmt: str,
    signal_range: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bulk-decode n consecutive YUV420 frames from a single raw buffer.

    Equivalent (bit-exact) to calling _decode_yuv_frame n times and stacking,
    but avoids the per-frame Python/numpy overhead by doing a single shift and a
    single float-cast over the entire buffer. Heavy-lifting numpy ops run once,
    not 3*n times.

    Returns:
        Y: float32 [n, h, w]
        U: float32 [n, h2, w2]
        V: float32 [n, h2, w2]
    """
    frame_size_samples = y_size + u_size + u_size

    # Full contiguous view over all frames' samples.
    arr = np.frombuffer(raw, dtype=dtype, count=n * frame_size_samples)

    mode = str(tenbit_mode or 'normalize').lower().strip()
    if bitdepth > 8 and mode == 'shift8':
        shift = max(int(bitdepth) - 8, 0)
        # Single fused shift + cast + divide + clip over the whole buffer.
        # np.right_shift returns the same dtype (uint16) without an extra view-copy
        # that .astype(np.uint16) would introduce; the subsequent .astype(np.float32)
        # is the one unavoidable cast, now performed exactly once (vs 3*n times).
        flat = (np.right_shift(arr, shift).astype(np.float32) / 255.0)
        np.clip(flat, 0.0, 1.0, out=flat)
    else:
        y_off, y_den, c_off, c_den = _range_params(bitdepth, pix_fmt, signal_range)
        # Plane-wise offsets/denominators differ, so we still need per-plane math,
        # but we can do a single float-cast over the whole buffer first.
        flat = arr.astype(np.float32)
        # Per-frame slicing loop below will apply the per-plane (off, den) + clip.

    # Slice per frame. This is O(n) but cheap: each slice is a view.
    Ys = np.empty((n, h, w), dtype=np.float32)
    Us = np.empty((n, h2, w2), dtype=np.float32)
    Vs = np.empty((n, h2, w2), dtype=np.float32)

    if bitdepth > 8 and mode == 'shift8':
        # shift8 path: `flat` is already normalized & clipped.
        for i in range(n):
            base = i * frame_size_samples
            Ys[i] = flat[base:base + y_size].reshape(h, w)
            Us[i] = flat[base + y_size:base + y_size + u_size].reshape(h2, w2)
            Vs[i] = flat[base + y_size + u_size:base + frame_size_samples].reshape(h2, w2)
    else:
        # normalize path: apply per-plane (val - off) / den, then clip.
        for i in range(n):
            base = i * frame_size_samples
            y_plane = flat[base:base + y_size]
            u_plane = flat[base + y_size:base + y_size + u_size]
            v_plane = flat[base + y_size + u_size:base + frame_size_samples]
            Ys[i] = ((y_plane - y_off) / y_den).clip(0.0, 1.0).reshape(h, w)
            Us[i] = ((u_plane - c_off) / c_den).clip(0.0, 1.0).reshape(h2, w2)
            Vs[i] = ((v_plane - c_off) / c_den).clip(0.0, 1.0).reshape(h2, w2)

    return Ys, Us, Vs


def _bulk_decode_yuv_frames_y_only(
    raw: bytes,
    n: int,
    h: int, w: int, h2: int, w2: int,
    y_size: int, u_size: int,
    dtype: type,
    bitdepth: int,
    tenbit_mode: str,
    pix_fmt: str,
    signal_range: str,
) -> np.ndarray:
    """Bulk-decode Y plane only from n consecutive YUV420 frames.

    Bit-exact equivalent to iterating per-frame Y extraction, but performs the
    shift + float cast exactly once over the entire raw buffer.

    Returns:
        Y: float32 [n, h, w]
    """
    frame_size_samples = y_size + u_size + u_size
    arr = np.frombuffer(raw, dtype=dtype, count=n * frame_size_samples)

    mode = str(tenbit_mode or 'shift8').lower().strip()
    if bitdepth > 8 and mode == 'shift8':
        shift = max(int(bitdepth) - 8, 0)
        flat = (np.right_shift(arr, shift).astype(np.float32) / 255.0)
        np.clip(flat, 0.0, 1.0, out=flat)
        Ys = np.empty((n, h, w), dtype=np.float32)
        for i in range(n):
            base = i * frame_size_samples
            Ys[i] = flat[base:base + y_size].reshape(h, w)
        return Ys
    else:
        y_off, y_den, _, _ = _range_params(bitdepth, pix_fmt, signal_range)
        flat = arr.astype(np.float32)
        Ys = np.empty((n, h, w), dtype=np.float32)
        for i in range(n):
            base = i * frame_size_samples
            y_plane = flat[base:base + y_size]
            Ys[i] = ((y_plane - y_off) / y_den).clip(0.0, 1.0).reshape(h, w)
        return Ys


def read_yuv420_frames(
    file_path: str,
    width: int,
    height: int,
    frame_indices: List[int],
    bitdepth: int = 10,
    bit_depth: Optional[int] = None,
    pix_fmt: str = 'yuv420p10le',
    signal_range: str = 'auto',
    tenbit_mode: str = 'shift8',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read multiple YUV420 frames.

    Optimized: single file open, with bulk-read for consecutive frames.

    Returns:
        Y: float32 [T, H, W] in [0, 1]
        U: float32 [T, H/2, W/2] in [0, 1]
        V: float32 [T, H/2, W/2] in [0, 1]
    """
    bitdepth, pix_fmt = _normalize_raw_args(bitdepth, bit_depth, pix_fmt)

    h, w = height, width
    h2, w2 = h // 2, w // 2
    bytes_per_sample = 2 if bitdepth > 8 else 1
    dtype = np.uint16 if bitdepth > 8 else np.uint8
    y_size = h * w
    u_size = h2 * w2
    frame_size_samples = y_size + u_size + u_size
    frame_size_bytes = frame_size_samples * bytes_per_sample

    n = len(frame_indices)
    Ys, Us, Vs = [], [], []

    with open(file_path, 'rb') as f:
        # Fast path: consecutive frames → single bulk read
        if n > 1 and _is_consecutive(frame_indices):
            f.seek(frame_indices[0] * frame_size_bytes)
            total_bytes = n * frame_size_bytes
            raw = f.read(total_bytes)
            if len(raw) < total_bytes:
                raise ValueError(
                    f"Incomplete bulk read in {file_path}. "
                    f"Expected {total_bytes} bytes, got {len(raw)}."
                )
            # Bulk decode: single shift + single float-cast over all n frames
            # (bit-exact equivalent to per-frame _decode_yuv_frame stacking).
            Y_arr, U_arr, V_arr = _bulk_decode_yuv_frames(
                raw, n, h, w, h2, w2, y_size, u_size, dtype,
                bitdepth, tenbit_mode, pix_fmt, signal_range,
            )
            return Y_arr, U_arr, V_arr
        else:
            # General path: single open, per-frame seek/read
            for idx in frame_indices:
                f.seek(idx * frame_size_bytes)
                raw = f.read(frame_size_bytes)
                if len(raw) < frame_size_bytes:
                    raise ValueError(
                        f"Incomplete frame at index {idx} in {file_path}. "
                        f"Expected {frame_size_bytes} bytes, got {len(raw)}."
                    )
                Y, U, V = _decode_yuv_frame(
                    raw, 0, frame_size_bytes,
                    h, w, h2, w2, y_size, u_size, dtype,
                    bitdepth, tenbit_mode, pix_fmt, signal_range,
                )
                Ys.append(Y)
                Us.append(U)
                Vs.append(V)

    return np.stack(Ys), np.stack(Us), np.stack(Vs)


def read_yuv420_frames_y(
    file_path: str,
    width: int,
    height: int,
    frame_indices: List[int],
    bitdepth: int = 10,
    bit_depth: Optional[int] = None,
    pix_fmt: str = 'yuv420p10le',
    signal_range: str = 'auto',
    tenbit_mode: str = 'shift8',
) -> np.ndarray:
    """
    Read only Y planes for multiple YUV420 frames.

    Optimized: single file open, with bulk-read for consecutive frames.

    Returns:
        Y: float32 [T, H, W] in [0, 1]
    """
    bitdepth, pix_fmt = _normalize_raw_args(bitdepth, bit_depth, pix_fmt)

    h, w = height, width
    h2, w2 = h // 2, w // 2
    bytes_per_sample = 2 if bitdepth > 8 else 1
    dtype = np.uint16 if bitdepth > 8 else np.uint8
    y_size = h * w
    u_size = h2 * w2
    frame_size_samples = y_size + u_size + u_size
    frame_size_bytes = frame_size_samples * bytes_per_sample
    y_bytes = y_size * bytes_per_sample

    n = len(frame_indices)
    Ys = []

    mode = str(tenbit_mode or 'shift8').lower().strip()

    with open(file_path, 'rb') as f:
        # Fast path: consecutive frames → single bulk read (read full frames, extract Y)
        if n > 1 and _is_consecutive(frame_indices):
            f.seek(frame_indices[0] * frame_size_bytes)
            total_bytes = n * frame_size_bytes
            raw = f.read(total_bytes)
            if len(raw) < total_bytes:
                raise ValueError(
                    f"Incomplete bulk read in {file_path}. "
                    f"Expected {total_bytes} bytes, got {len(raw)}."
                )
            # Bulk decode Y plane: single shift + single float-cast over all frames.
            return _bulk_decode_yuv_frames_y_only(
                raw, n, h, w, h2, w2, y_size, u_size, dtype,
                bitdepth, tenbit_mode, pix_fmt, signal_range,
            )
        else:
            # General path: single open, per-frame seek/read (only Y plane)
            for idx in frame_indices:
                f.seek(idx * frame_size_bytes)
                raw_y = f.read(y_bytes)
                if len(raw_y) < y_bytes:
                    raise ValueError(
                        f"Incomplete frame at index {idx} in {file_path}. "
                        f"Expected {y_bytes} Y bytes, got {len(raw_y)}."
                    )
                data = np.frombuffer(raw_y, dtype=dtype)
                if bitdepth > 8 and mode == 'shift8':
                    shift = max(int(bitdepth) - 8, 0)
                    Y = ((data.reshape(h, w).astype(np.uint16) >> shift).astype(np.float32) / 255.0).clip(0.0, 1.0)
                else:
                    y_off, y_den, _, _ = _range_params(bitdepth, pix_fmt, signal_range)
                    Y = ((data.reshape(h, w).astype(np.float32) - y_off) / y_den).clip(0.0, 1.0)
                Ys.append(Y)

    return np.stack(Ys)


def read_yuv420_frames_rgb_legacy(
    file_path: str,
    width: int,
    height: int,
    frame_indices: List[int],
    bitdepth: int = 10,
    bit_depth: Optional[int] = None,
    pix_fmt: str = 'yuv420p10le',
    tenbit_mode: str = 'shift8',
) -> np.ndarray:
    """
    Legacy PE-compatible raw YUV420 -> RGB decode path.

    Behavior:
      - 8-bit: direct I420 decode with OpenCV
      - 10-bit + shift8: right-shift to 8-bit, then OpenCV I420 decode

    Returns:
        RGB float32 [T, H, W, 3] in [0, 1].
    """
    bitdepth, _pix_fmt = _normalize_raw_args(bitdepth, bit_depth, pix_fmt)
    mode = str(tenbit_mode or 'shift8').lower().strip()
    if bitdepth > 8 and mode != 'shift8':
        raise ValueError(
            f"Legacy RGB decoder supports tenbit_mode=shift8 for >8-bit YUV, got '{tenbit_mode}'."
        )

    h, w = int(height), int(width)
    h2, w2 = h // 2, w // 2
    bytes_per_sample = 2 if bitdepth > 8 else 1
    y_size = h * w
    u_size = h2 * w2
    v_size = h2 * w2
    frame_size_samples = y_size + u_size + v_size
    frame_size_bytes = frame_size_samples * bytes_per_sample

    rgbs: List[np.ndarray] = []
    with open(file_path, 'rb') as f:
        for idx in frame_indices:
            offset = int(idx) * frame_size_bytes
            f.seek(offset)
            raw = f.read(frame_size_bytes)
            if len(raw) < frame_size_bytes:
                raise ValueError(
                    f"Incomplete frame at index {idx} in {file_path}. "
                    f"Expected {frame_size_bytes} bytes, got {len(raw)}."
                )

            if bitdepth > 8:
                yuv = np.frombuffer(raw, dtype='<u2')
                shift = max(int(bitdepth) - 8, 0)
                yuv = (yuv >> shift).astype(np.uint8)
            else:
                yuv = np.frombuffer(raw, dtype=np.uint8)

            yuv_frame = np.array(yuv, copy=True).reshape((h * 3 // 2, w))
            rgb = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
            rgbs.append((rgb.astype(np.float32) / 255.0).clip(0.0, 1.0))

    return np.stack(rgbs, axis=0)
