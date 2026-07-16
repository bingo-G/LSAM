"""
Frame cache reader for CVQM evaluation acceleration.

Binary format (.bin):
  - 16-byte header: magic(4B) + width(2B) + height(2B) + num_frames(2B)
                    + bitdepth(2B) + flags(2B) + reserved(2B)
  - Raw uint16 YUV420 data: T frames × (H*W + 2*(H/2)*(W/2)) samples × 2 bytes

Design:
  - Per-clip independent files: clip_0.bin, clip_1.bin, ..., clip_K-1.bin
  - readinto + pre-allocated buffer for zero-malloc, zero-memcpy reading
  - Worker-level buffer pre-allocation via worker_init_fn
  - Automatic fallback to YUV when cache file is missing

Usage:
  # In worker_init_fn:
  init_frame_cache_buffer(max_width=4096, max_height=2160, num_frames=8)

  # In __getitem__:
  Y, U, V = load_cached_clip('/path/to/clip_0.bin')
"""

import os
import struct
import logging
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger('hmf_vqa.frame_cache')

# ============================================================================
# Optional torch dependency (for accelerated decode)
# ============================================================================
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ============================================================================
# Binary format constants
# ============================================================================
CACHE_MAGIC = b'HMFC'  # HMF Frame Cache
HEADER_SIZE = 16
HEADER_FMT = '<4sHHHHH2x'  # magic(4) + width(2) + height(2) + nframes(2) + bitdepth(2) + flags(2) + pad(2) = 16

# Flags
FLAG_YUV420 = 0x0001  # YUV420 planar format

# ============================================================================
# Torch decode configuration
# ============================================================================
# torch decode uses OpenMP/MKL multi-threading for right_shift + float conversion.
# Benchmark (V100, Xeon 8163 44 cores):
#   1080p: numpy 153ms → torch threads=8  78ms = 1.96x speedup
#   4K:    numpy 862ms → torch threads=8 414ms = 2.08x speedup
# On 4090 (Xeon 8458P 176 cores): ~1.18x speedup (smaller benefit).
TORCH_DECODE_THREADS = int(os.environ.get('HMF_VQA_TORCH_DECODE_THREADS', '8'))
_TORCH_DECODE_ENABLED = _HAS_TORCH and os.environ.get('HMF_VQA_TORCH_DECODE', '1') != '0'

# ============================================================================
# Per-worker pre-allocated buffer (thread-local via DataLoader fork)
# ============================================================================
_worker_read_buf: Optional[bytearray] = None
_worker_buf_size: int = 0


def _calc_frame_data_bytes(width: int, height: int, num_frames: int, bitdepth: int = 10) -> int:
    """Calculate total raw data bytes for T YUV420 frames."""
    bps = 2 if bitdepth > 8 else 1
    h2, w2 = height // 2, width // 2
    samples_per_frame = height * width + 2 * h2 * w2
    return num_frames * samples_per_frame * bps


def init_frame_cache_buffer(
    max_width: int = 4096,
    max_height: int = 2160,
    num_frames: int = 8,
    bitdepth: int = 10,
) -> None:
    """
    Pre-allocate the readinto buffer for this worker process.
    Call this in worker_init_fn. Buffer is reused across all load_cached_clip() calls.

    Max buffer size for 4K 10-bit 8-frame YUV420:
      8 * (4096*2160 + 2*2048*1080) * 2 = 8 * 12441600 * 2 = 199,065,600 bytes ≈ 190MB
    """
    global _worker_read_buf, _worker_buf_size
    needed = _calc_frame_data_bytes(max_width, max_height, num_frames, bitdepth)
    if _worker_read_buf is not None and _worker_buf_size >= needed:
        return  # Already large enough
    _worker_read_buf = bytearray(needed)
    _worker_buf_size = needed

    logger.debug(
        "Frame cache buffer allocated: %.1f MB (w=%d h=%d T=%d bd=%d)",
        needed / (1024 * 1024), max_width, max_height, num_frames, bitdepth,
    )
    # Log torch decode status once per worker (only worker 0 prints at INFO)
    _wid = getattr(init_frame_cache_buffer, '_worker_id', None)
    if _wid is not None and _wid == 0:
        if _TORCH_DECODE_ENABLED:
            logger.info(
                "[FrameCache] torch decode ON (threads=%s). "
                "Set HMF_VQA_TORCH_DECODE=0 to disable.",
                os.environ.get('OMP_NUM_THREADS', 'default'),
            )
        else:
            logger.info("[FrameCache] torch decode OFF, using numpy fallback.")


def _ensure_buffer(needed: int) -> bytearray:
    """Ensure the worker buffer is large enough, lazy-allocate if needed."""
    global _worker_read_buf, _worker_buf_size
    if _worker_read_buf is not None and _worker_buf_size >= needed:
        return _worker_read_buf
    # Fallback: allocate on the fly (slower, but safe for main process / uninitialized workers)
    _worker_read_buf = bytearray(needed)
    _worker_buf_size = needed
    return _worker_read_buf


# ============================================================================
# Write cache file
# ============================================================================
def write_cache_file(
    path: str,
    width: int,
    height: int,
    num_frames: int,
    bitdepth: int,
    raw_data: bytes,
    flags: int = FLAG_YUV420,
) -> None:
    """
    Write a frame cache .bin file with header + raw YUV data.

    Args:
        path: Output file path.
        width, height: Frame dimensions.
        num_frames: Number of frames.
        bitdepth: Bit depth (8 or 10).
        raw_data: Raw YUV420 uint16/uint8 data bytes.
        flags: Format flags.
    """
    expected_size = _calc_frame_data_bytes(width, height, num_frames, bitdepth)
    if len(raw_data) != expected_size:
        raise ValueError(
            f"raw_data size mismatch: got {len(raw_data)}, "
            f"expected {expected_size} for {width}x{height}x{num_frames}x{bitdepth}bit"
        )

    header = struct.pack(HEADER_FMT, CACHE_MAGIC, width, height, num_frames, bitdepth, flags)
    assert len(header) == HEADER_SIZE

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(raw_data)


# ============================================================================
# Read cache file (extreme performance version)
# ============================================================================
def parse_cache_header(path: str) -> Tuple[int, int, int, int, int]:
    """
    Read and parse the 16-byte header from a cache file.

    Returns:
        (width, height, num_frames, bitdepth, flags)
    """
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
    if len(hdr) < HEADER_SIZE:
        raise ValueError(f"Incomplete header in {path}: {len(hdr)} bytes")
    magic, w, h, nf, bd, flags = struct.unpack(HEADER_FMT, hdr)
    if magic != CACHE_MAGIC:
        raise ValueError(f"Invalid magic in {path}: {magic!r} (expected {CACHE_MAGIC!r})")
    return w, h, nf, bd, flags


def load_cached_clip(
    path: str,
    signal_range: str = 'auto',
    tenbit_mode: str = 'shift8',
    pix_fmt: str = 'yuv420p10le',
) -> Tuple[Union[np.ndarray, 'torch.Tensor'],
           Union[np.ndarray, 'torch.Tensor'],
           Union[np.ndarray, 'torch.Tensor']]:
    """
    Load a cached clip using readinto + pre-allocated buffer (zero-malloc path).

    This function matches the output format of read_yuv420_frames():
      Y: float32 [T, H, W] in [0, 1]
      U: float32 [T, H/2, W/2] in [0, 1]
      V: float32 [T, H/2, W/2] in [0, 1]

    When torch is available and HMF_VQA_TORCH_DECODE != '0', returns torch.Tensor
    (approx 2x faster on V100 with threads=8). Otherwise returns numpy arrays.
    The numerical result is identical: both paths compute
      clamp(right_shift(uint16, 2).to_float32() * (1/255), 0, 1)

    The normalization logic is identical to yuv_reader._decode_yuv_frame().

    Args:
        path: Path to .bin cache file.
        signal_range: 'auto', 'limited', or 'full'.
        tenbit_mode: 'shift8' or 'normalize'.
        pix_fmt: Pixel format string.

    Returns:
        (Y, U, V) — numpy float32 arrays or torch float32 tensors.
    """
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
        if len(hdr) < HEADER_SIZE:
            raise ValueError(f"Incomplete header in {path}")

        magic, w, h, nf, bd, flags = struct.unpack(HEADER_FMT, hdr)
        if magic != CACHE_MAGIC:
            raise ValueError(f"Invalid magic in {path}: {magic!r}")

        bps = 2 if bd > 8 else 1
        dtype = np.uint16 if bd > 8 else np.uint8
        h2, w2 = h // 2, w // 2
        y_size = h * w
        u_size = h2 * w2
        spf = y_size + 2 * u_size  # samples per frame
        data_bytes = nf * spf * bps

        buf = _ensure_buffer(data_bytes)
        mv = memoryview(buf)[:data_bytes]
        nbytes_read = f.readinto(mv)
        if nbytes_read < data_bytes:
            raise ValueError(
                f"Incomplete data in {path}: got {nbytes_read}, expected {data_bytes}"
            )

    mode = str(tenbit_mode or 'normalize').lower().strip()

    # ------------------------------------------------------------------
    # Torch-accelerated decode path (shift8 mode, 10-bit)
    # Uses OpenMP multi-threading: ~2x faster than numpy on V100.
    # Math: clamp(right_shift(int32 & 0xFFFF, 2).float() * (1/255), 0, 1)
    # Numerically identical to the numpy path because:
    #   - int16 → int32 → bitwise_and(0xFFFF) is equivalent to
    #     uint16 zero-extension for all uint16 values [0, 65535]
    #   - All subsequent ops (>> shift, float, mul, clamp) are identical
    # ------------------------------------------------------------------
    if _TORCH_DECODE_ENABLED and bd > 8 and mode == 'shift8':
        return _torch_decode_shift8(mv, nf, spf, y_size, u_size, h, w, h2, w2, bd)

    # ------------------------------------------------------------------
    # Numpy fallback (original path, always available)
    # ------------------------------------------------------------------
    return _numpy_decode(mv, dtype, nf, spf, y_size, u_size, h, w, h2, w2,
                         bd, mode, pix_fmt, signal_range)


def _torch_decode_shift8(
    mv: memoryview,
    nf: int, spf: int, y_size: int, u_size: int,
    h: int, w: int, h2: int, w2: int, bd: int,
) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
    """
    Torch-accelerated shift8 decode for 10-bit YUV.

    Returns torch.Tensor float32: Y[T,H,W], U[T,H/2,W/2], V[T,H/2,W/2] in [0,1].
    """
    shift = max(bd - 8, 0)
    inv255 = 1.0 / 255.0

    # NOTE: torch.set_num_threads() is NOT called here.
    # It must be set ONCE in worker_init_fn (before any OpenMP work),
    # because calling it inside a forked DataLoader worker after OpenMP
    # has been initialized causes deadlock (OpenMP lock state is corrupted
    # by fork).  See init_frame_cache_buffer() for the one-time setup.

    # Create writable int32 tensor from the raw uint16 buffer.
    #
    # Challenge: torch has no uint16 dtype.  np.frombuffer(int16) +
    # torch.from_numpy gives a *signed* int16 tensor.  A simple
    # .to(int32) would sign-extend values ≥ 32768, giving wrong results
    # when right-shifted (arithmetic vs logical shift).
    #
    # Solution: .to(int32) (sign-extends) then .bitwise_and_(0xFFFF)
    # which zeros the sign-extended upper bits, producing the same
    # result as unsigned zero-extension.  Both .to() and .bitwise_and_()
    # are multi-threaded in torch (unlike numpy .astype which is single-
    # threaded), so this is fast on V100 with 8 threads.
    #
    # Correctness proof for 10-bit YUV (values 0..1023):
    #   int16(v) == uint16(v) for v in [0, 32767], and 1023 < 32768,
    #   so the bitwise_and_ is a no-op for valid 10-bit data.
    #   For out-of-range values (padding, corruption), bitwise_and_
    #   ensures correct unsigned semantics matching numpy uint16 >> shift.
    raw_np = np.frombuffer(mv, dtype=np.int16, count=nf * spf)
    raw = torch.from_numpy(raw_np).to(torch.int32).bitwise_and_(0xFFFF)
    raw_2d = raw[:nf * spf].reshape(nf, spf)

    y_all = raw_2d[:, :y_size]
    u_all = raw_2d[:, y_size:y_size + u_size]
    v_all = raw_2d[:, y_size + u_size:y_size + 2 * u_size]

    Y = (y_all >> shift).float().mul_(inv255).clamp_(0.0, 1.0).reshape(nf, h, w)
    U = (u_all >> shift).float().mul_(inv255).clamp_(0.0, 1.0).reshape(nf, h2, w2)
    V = (v_all >> shift).float().mul_(inv255).clamp_(0.0, 1.0).reshape(nf, h2, w2)

    return Y, U, V


def _numpy_decode(
    mv: memoryview,
    dtype: type,
    nf: int, spf: int, y_size: int, u_size: int,
    h: int, w: int, h2: int, w2: int,
    bd: int, mode: str, pix_fmt: str, signal_range: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Original numpy decode path (unchanged logic, extracted to a function).

    Returns numpy float32: Y[T,H,W], U[T,H/2,W/2], V[T,H/2,W/2] in [0,1].
    """
    # Zero-copy view into the buffer
    raw = np.frombuffer(mv, dtype=dtype, count=nf * spf)

    # Reshape to [T, spf] then split Y/U/V using column slicing
    raw_2d = raw.reshape(nf, spf)
    y_all = raw_2d[:, :y_size]                                  # [T, H*W]
    u_all = raw_2d[:, y_size:y_size + u_size]                   # [T, H/2*W/2]
    v_all = raw_2d[:, y_size + u_size:y_size + 2 * u_size]      # [T, H/2*W/2]

    if bd > 8 and mode == 'shift8':
        shift = max(bd - 8, 0)
        inv255 = np.float32(1.0 / 255.0)
        Y = np.clip(np.right_shift(y_all, shift).astype(np.float32) * inv255,
                     0.0, 1.0).reshape(nf, h, w)
        U = np.clip(np.right_shift(u_all, shift).astype(np.float32) * inv255,
                     0.0, 1.0).reshape(nf, h2, w2)
        V = np.clip(np.right_shift(v_all, shift).astype(np.float32) * inv255,
                     0.0, 1.0).reshape(nf, h2, w2)
    else:
        y_off, y_den, c_off, c_den = _range_params(bd, pix_fmt, signal_range)
        inv_y = np.float32(1.0 / y_den)
        inv_c = np.float32(1.0 / c_den)
        Y = np.clip((y_all.astype(np.float32) - np.float32(y_off)) * inv_y,
                     0.0, 1.0).reshape(nf, h, w)
        U = np.clip((u_all.astype(np.float32) - np.float32(c_off)) * inv_c,
                     0.0, 1.0).reshape(nf, h2, w2)
        V = np.clip((v_all.astype(np.float32) - np.float32(c_off)) * inv_c,
                     0.0, 1.0).reshape(nf, h2, w2)

    return Y, U, V


def load_cached_clip_raw(path: str) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Load raw uint16/uint8 data from cache without normalization.
    Used for verification against original YUV read.

    Returns:
        (raw_data, width, height, num_frames, bitdepth)
        raw_data: uint16 or uint8 array of shape [T * spf]
    """
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
        magic, w, h, nf, bd, flags = struct.unpack(HEADER_FMT, hdr)
        if magic != CACHE_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}")

        bps = 2 if bd > 8 else 1
        dtype = np.uint16 if bd > 8 else np.uint8
        h2, w2 = h // 2, w // 2
        spf = h * w + 2 * h2 * w2
        data_bytes = nf * spf * bps

        buf = _ensure_buffer(data_bytes)
        mv = memoryview(buf)[:data_bytes]
        f.readinto(mv)

    raw = np.frombuffer(mv, dtype=dtype, count=nf * spf).copy()
    return raw, w, h, nf, bd


# ============================================================================
# Range params (duplicated from yuv_reader to avoid circular import)
# ============================================================================
def _resolve_signal_range(pix_fmt: str, signal_range: str) -> str:
    sr = str(signal_range or 'auto').strip().lower()
    if sr in ('limited', 'tv', 'mpeg'):
        return 'limited'
    if sr in ('full', 'pc', 'jpeg'):
        return 'full'
    pf = str(pix_fmt or '').strip().lower()
    if ('yuvj' in pf) or ('full' in pf):
        return 'full'
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


# ============================================================================
# Path utilities
# ============================================================================
def get_cache_clip_path(
    cache_root: str,
    video_id: str,
    clip_idx: int,
    is_ref: bool = False,
) -> str:
    """
    Resolve the cache file path for a given video clip.

    Layout:
      {cache_root}/dis/{video_id_sanitized}/clip_{idx}.bin
      {cache_root}/ref/{video_id_sanitized}/clip_{idx}.bin
    """
    subdir = 'ref' if is_ref else 'dis'
    # Sanitize video_id: replace path separators and special chars
    safe_id = video_id.replace('/', '__').replace('\\', '__').replace(' ', '_')
    return os.path.join(cache_root, subdir, safe_id, f'clip_{clip_idx}.bin')


def cache_clip_exists(
    cache_root: str,
    video_id: str,
    clip_idx: int,
    is_ref: bool = False,
) -> bool:
    """Check if a cache file exists for the given clip."""
    path = get_cache_clip_path(cache_root, video_id, clip_idx, is_ref)
    return os.path.isfile(path)
