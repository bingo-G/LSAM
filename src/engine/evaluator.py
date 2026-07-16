"""
Evaluator: run inference on a dataloader and compute VQA metrics.

Supports:
  - Standard evaluation (single pass, returns per-video results)
  - CVQM two-stage evaluation (phase1 / phase2 / all, separately)
  - AMP inference
  - dry_run mode
  - All-gather across DDP ranks
  - Save per-video CSV inference results
  - Rescale predictions to MOS scale before computing metrics
  - Per-video timing profiling (data loading, model inference, per-frame)
"""

import csv
import logging
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ..utils.dist import is_main_process
from ..losses.metrics import compute_all_metrics, compute_cvqm_stage_metrics

try:
    from ..data.gpu_preprocess import build_resize_dis_legacy_pe as _gpu_build_resize_dis_legacy_pe
    _HAS_GPU_PREP = True
except Exception:
    _gpu_build_resize_dis_legacy_pe = None
    _HAS_GPU_PREP = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 0.001:
        return f'{seconds * 1000000:.0f}μs'
    if seconds < 1.0:
        return f'{seconds * 1000:.1f}ms'
    if seconds < 60:
        return f'{seconds:.2f}s'
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f'{minutes}m{secs:.1f}s'


def _format_fps(num_frames: int, seconds: float) -> str:
    """Format frames-per-second."""
    if seconds < 1e-9:
        return 'N/A'
    fps = num_frames / seconds
    return f'{fps:.2f} fps'


def _apply_gpu_preprocess_legacy_pe(
    batch: Dict,
    spatial_info: Dict,
    device: torch.device,
    has_cuda: bool,
) -> Optional[float]:
    """Rebuild ``spatial_info['resize_dis']`` (and optionally ``'resize_ref'``)
    on GPU from raw YUV420 planes carried in the batch.

    Activated only when:
      1. The GPU preprocess helper was imported successfully.
      2. The batch contains ``raw_y_dis`` / ``raw_u_half_dis`` / ``raw_v_half_dis``
         tensors (emitted by ``VQADataset`` when
         ``HMF_VQA_GPU_PREPROCESS`` is enabled).
      3. The existing ``resize_dis`` has the legacy_pe + gmsavg + stack layout
         ``[B, P, 3, T, ph, pw]`` (bypassed otherwise — keeps the CPU result).

    Returns
    -------
    elapsed_sec : Optional[float]
        GPU preprocess wall-clock time (seconds), or ``None`` when the
        helper did not run (feature disabled, missing batch keys, etc.).
    """
    if not _HAS_GPU_PREP or _gpu_build_resize_dis_legacy_pe is None:
        return None

    raw_y_dis = batch.get('raw_y_dis', None)
    raw_u_dis = batch.get('raw_u_half_dis', None)
    raw_v_dis = batch.get('raw_v_half_dis', None)
    if not (isinstance(raw_y_dis, torch.Tensor) and
            isinstance(raw_u_dis, torch.Tensor) and
            isinstance(raw_v_dis, torch.Tensor)):
        return None

    # Only support the SD3-eval layout: single-clip, stack mode (5D+B = 6D).
    existing = spatial_info.get('resize_dis', None)
    if not (isinstance(existing, torch.Tensor) and existing.dim() == 6):
        return None  # not the legacy_pe + stack layout → skip silently.

    # Raw Y tensor after collate: [B, T, H, W]
    if raw_y_dis.dim() != 4 or raw_u_dis.dim() != 4 or raw_v_dis.dim() != 4:
        return None
    B = raw_y_dis.shape[0]
    # Derive stack-layout hyperparameters from existing resize_dis
    _, P, C, T, ph, pw = existing.shape
    if C != 3:
        return None

    # Read back GMS config from the model/dataset defaults used in
    # ``_sample_semantic_gmsavg`` (legacy_pe).  We read them off the
    # existing resize_dis shape + the raw Y tensor; grid_size/ppf have
    # to be provided externally because they aren't shape-encoded.
    # Fall back to the canonical SD3 defaults (grid=7, ppf=P).
    grid_size = 7
    patches_per_frame = int(P)
    semantic_target_size = int(ph)  # SD3: ph == ts == 224

    # Decode UV upsample mode from the tagged int the dataset attached
    # (0=bilinear, 1=bicubic, 2=nearest).  Must match the CPU reader so
    # that the GPU rebuild is numerically equivalent to the CPU path.
    _uv_code_t = batch.get('uv_upsample_code', None)
    _uv_code = 0
    if isinstance(_uv_code_t, torch.Tensor) and _uv_code_t.numel() >= 1:
        try:
            _uv_code = int(_uv_code_t.reshape(-1)[0].item())
        except Exception:
            _uv_code = 0
    _uv_mode_str = {0: 'bilinear', 1: 'bicubic', 2: 'nearest'}.get(_uv_code, 'bilinear')

    # Raw tensors should already be on the target device (moved by the
    # evaluator's H2D phase before t_model_start).  If they are still on
    # CPU (e.g. called from a test harness), move them now as a fallback.
    raw_y_d = raw_y_dis if raw_y_dis.device == device else raw_y_dis.to(device, non_blocking=True)
    raw_u_d = raw_u_dis if raw_u_dis.device == device else raw_u_dis.to(device, non_blocking=True)
    raw_v_d = raw_v_dis if raw_v_dis.device == device else raw_v_dis.to(device, non_blocking=True)

    # Record GPU wall-clock with CUDA events (preferred) or perf_counter (CPU).
    # No synchronize() needed here — the caller already synced after H2D.
    if has_cuda and device.type == 'cuda':
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
    else:
        ev_start = ev_end = None
        _t0 = time.perf_counter()

    out_list: List[torch.Tensor] = []
    for b in range(B):
        out_b = _gpu_build_resize_dis_legacy_pe(
            raw_y_d[b], raw_u_d[b], raw_v_d[b],
            patch_h=int(ph), patch_w=int(pw),
            grid_size=grid_size,
            patches_per_frame=patches_per_frame,
            semantic_gms_mode='stack',
            semantic_target_size=semantic_target_size,
            uv_upsample=_uv_mode_str,
        )
        out_list.append(out_b)
    new_resize_dis = torch.stack(out_list, dim=0)  # [B, P, 3, T, ph, pw]
    spatial_info['resize_dis'] = new_resize_dis

    # Optional: rebuild resize_ref when raw ref planes are present.
    raw_y_ref = batch.get('raw_y_ref', None)
    raw_u_ref = batch.get('raw_u_half_ref', None)
    raw_v_ref = batch.get('raw_v_half_ref', None)
    if (isinstance(raw_y_ref, torch.Tensor) and
            isinstance(raw_u_ref, torch.Tensor) and
            isinstance(raw_v_ref, torch.Tensor) and
            raw_y_ref.dim() == 4):
        # Ref raw tensors should already be on device (moved by evaluator H2D phase).
        raw_y_r = raw_y_ref if raw_y_ref.device == device else raw_y_ref.to(device, non_blocking=True)
        raw_u_r = raw_u_ref if raw_u_ref.device == device else raw_u_ref.to(device, non_blocking=True)
        raw_v_r = raw_v_ref if raw_v_ref.device == device else raw_v_ref.to(device, non_blocking=True)
        out_r_list: List[torch.Tensor] = []
        for b in range(raw_y_r.shape[0]):
            out_b = _gpu_build_resize_dis_legacy_pe(
                raw_y_r[b], raw_u_r[b], raw_v_r[b],
                patch_h=int(ph), patch_w=int(pw),
                grid_size=grid_size,
                patches_per_frame=patches_per_frame,
                semantic_gms_mode='stack',
                semantic_target_size=semantic_target_size,
                uv_upsample=_uv_mode_str,
            )
            out_r_list.append(out_b)
        new_resize_ref = torch.stack(out_r_list, dim=0)
        spatial_info['resize_ref'] = new_resize_ref

    if ev_start is not None:
        ev_end.record()
        torch.cuda.synchronize(device)
        elapsed_sec = float(ev_start.elapsed_time(ev_end)) / 1000.0
    else:
        elapsed_sec = float(time.perf_counter() - _t0)

    return elapsed_sec


def _apply_gpu_preprocess_legacy_pe_clip(
    batch: Dict,
    spatial_clip: Dict,
    clip_idx: int,
    device: torch.device,
    has_cuda: bool,
) -> Optional[float]:
    """Multi-clip variant of :func:`_apply_gpu_preprocess_legacy_pe`.

    Rebuilds ``spatial_clip['resize_dis']`` (and optionally ``'resize_ref'``)
    for ONE clip ``clip_idx`` from the multi-clip raw YUV420 planes carried in
    the batch with layout ``[B, K, T, H, W]`` (Y) and ``[B, K, T, H/2, W/2]``
    (U/V half).

    Timing semantics match the single-clip helper: the per-clip H2D of the raw
    planes happens BEFORE the CUDA-event window, so the returned wall time is
    the GPU rebuild compute only (UV upsample + resize + patch gather), with
    H2D excluded.

    Returns the GPU rebuild wall time (seconds), or ``None`` when it did not run
    (feature disabled, raw planes missing, or unexpected layout).
    """
    if not _HAS_GPU_PREP or _gpu_build_resize_dis_legacy_pe is None:
        return None

    raw_y = batch.get('raw_y_dis', None)
    raw_u = batch.get('raw_u_half_dis', None)
    raw_v = batch.get('raw_v_half_dis', None)
    if not (isinstance(raw_y, torch.Tensor) and isinstance(raw_u, torch.Tensor)
            and isinstance(raw_v, torch.Tensor)):
        return None
    # Multi-clip raw layout: [B, K, T, H, W]
    if raw_y.dim() != 5 or raw_u.dim() != 5 or raw_v.dim() != 5:
        return None

    existing = spatial_clip.get('resize_dis', None)
    if not (isinstance(existing, torch.Tensor) and existing.dim() == 6):
        return None  # not the legacy_pe + stack layout for this clip → skip.
    B, P, C, T, ph, pw = existing.shape
    if C != 3:
        return None
    K = raw_y.shape[1]
    if clip_idx < 0 or clip_idx >= K:
        return None

    # UV upsample mode (match the CPU reader for numerical parity).
    _uv_code_t = batch.get('uv_upsample_code', None)
    _uv_code = 0
    if isinstance(_uv_code_t, torch.Tensor) and _uv_code_t.numel() >= 1:
        try:
            _uv_code = int(_uv_code_t.reshape(-1)[0].item())
        except Exception:
            _uv_code = 0
    _uv_mode_str = {0: 'bilinear', 1: 'bicubic', 2: 'nearest'}.get(_uv_code, 'bilinear')

    def _build_group(ry, ru, rv):
        outs = []
        for b in range(ry.shape[0]):
            outs.append(_gpu_build_resize_dis_legacy_pe(
                ry[b], ru[b], rv[b],
                patch_h=int(ph), patch_w=int(pw),
                grid_size=7, patches_per_frame=int(P),
                semantic_gms_mode='stack',
                semantic_target_size=int(ph),
                uv_upsample=_uv_mode_str,
            ))
        return torch.stack(outs, dim=0)

    # ── H2D of this clip's raw planes (EXCLUDED from the timed window) ──
    raw_y_c = raw_y[:, clip_idx]
    raw_u_c = raw_u[:, clip_idx]
    raw_v_c = raw_v[:, clip_idx]
    if raw_y_c.device != device:
        raw_y_c = raw_y_c.to(device, non_blocking=True)
        raw_u_c = raw_u_c.to(device, non_blocking=True)
        raw_v_c = raw_v_c.to(device, non_blocking=True)

    # Optional ref planes for this clip.
    raw_y_r = batch.get('raw_y_ref', None)
    raw_u_r = batch.get('raw_u_half_ref', None)
    raw_v_r = batch.get('raw_v_half_ref', None)
    _has_ref = (isinstance(raw_y_r, torch.Tensor) and raw_y_r.dim() == 5 and
                isinstance(raw_u_r, torch.Tensor) and isinstance(raw_v_r, torch.Tensor)
                and clip_idx < raw_y_r.shape[1])
    if _has_ref:
        ry = raw_y_r[:, clip_idx]
        ru = raw_u_r[:, clip_idx]
        rv = raw_v_r[:, clip_idx]
        if ry.device != device:
            ry = ry.to(device, non_blocking=True)
            ru = ru.to(device, non_blocking=True)
            rv = rv.to(device, non_blocking=True)

    # Time only the GPU rebuild (H2D above is already done + synced below).
    if has_cuda and device.type == 'cuda':
        torch.cuda.synchronize(device)  # ensure H2D completed before timing
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
    else:
        ev_start = ev_end = None
        _t0 = time.perf_counter()

    spatial_clip['resize_dis'] = _build_group(raw_y_c, raw_u_c, raw_v_c)
    if _has_ref:
        spatial_clip['resize_ref'] = _build_group(ry, ru, rv)

    if ev_start is not None:
        ev_end.record()
        torch.cuda.synchronize(device)
        return float(ev_start.elapsed_time(ev_end)) / 1000.0
    return float(time.perf_counter() - _t0)


def _compute_timing_stats(timing_records: List[Dict], dataset_name: str = '') -> Dict:
    """
    Compute comprehensive timing statistics from per-video timing records.

    Each record should contain:
      - video_id, height, width, num_frames
      - data_load_time, model_infer_time, total_time
      - (optional, fine-grained) io_decode_time, preprocess_time, sample_total_time
      - (optional) orig_height, orig_width
      - (optional) stage

    Returns a dict with overall + resolution-grouped stats.

    Concepts:
      data_load_time   = DataLoader `next(iter)` wall time (IO + decode + preprocess + collate)
      model_infer_time = H2D transfer + model forward (all clips summed)
      io_decode_time   = disk IO + YUV decode (dataset __getitem__, before preprocessing)
      preprocess_time  = Resize + patch sampling + semantic branch prep (dataset __getitem__)
      method_infer_time = preprocess_time + model_infer_time
                          → the cost that actually belongs to "our method"
                            when reporting inference FPS.
    """
    if not timing_records:
        return {}

    # Overall stats
    total_data_time = sum(r['data_load_time'] for r in timing_records)
    total_model_time = sum(r['model_infer_time'] for r in timing_records)
    # Pure model forward: only the model(...) call itself, excluding per-clip
    # H2D transfer, CPU tensor slicing, and torch.cuda.empty_cache() cost.
    # 0.0 when the record didn't emit it (older callers, or non-timed runs).
    total_pure_forward_time = sum(
        float(r.get('pure_forward_time', 0.0) or 0.0) for r in timing_records
    )
    has_pure_forward = total_pure_forward_time > 1e-9
    total_time = sum(r['total_time'] for r in timing_records)
    total_frames = sum(r.get('num_frames', 0) for r in timing_records)
    n_videos = len(timing_records)

    # Fine-grained totals (may be 0 when dataset didn't inject them).
    # IMPORTANT caveat:
    #   Dataset-side timings (io_decode_time, preprocess_time) are measured
    #   inside each DataLoader worker's __getitem__ with time.perf_counter().
    #   When num_workers > 1 these workers run in parallel processes; summing
    #   their per-sample times therefore overcounts by ~num_workers.
    #   → We keep the raw sums for diagnostics, but ALSO scale them down to
    #     the real DataLoader wall clock (`data_load_sec`) using each phase's
    #     proportional share.  This gives numbers that are self-consistent
    #     with wall_clock.total_sec (io+pre+model ≈ total).
    total_io_time = sum(float(r.get('io_decode_time', 0.0) or 0.0) for r in timing_records)
    total_pre_time = sum(float(r.get('preprocess_time', 0.0) or 0.0) for r in timing_records)
    total_sample_time = sum(float(r.get('sample_total_time', 0.0) or 0.0) for r in timing_records)
    # GPU preprocess time is recorded on the main (evaluator) thread, so it
    # is a wall-clock time — no worker-parallelism scaling needed.
    total_gpu_prep_time = sum(float(r.get('gpu_preprocess_time', 0.0) or 0.0) for r in timing_records)
    has_fine = total_sample_time > 1e-9 or total_pre_time > 1e-9 or total_io_time > 1e-9
    has_gpu_prep = total_gpu_prep_time > 1e-9

    # Wall-clock scaled values: preserve proportions but rescale so that
    # io_wall + pre_wall ≈ data_load_sec.
    if has_fine and (total_io_time + total_pre_time) > 1e-9:
        _scale = total_data_time / max(total_io_time + total_pre_time, 1e-9)
    else:
        _scale = 1.0
    total_io_wall = total_io_time * _scale
    total_pre_wall = total_pre_time * _scale
    # method_infer = preprocess (wall-clock scaled) + model forward
    total_method_time = total_pre_wall + total_model_time if has_fine else total_model_time

    # Observed worker parallelism (≈ how many workers ran in parallel on average).
    # Equal to _scale^-1 when data_load_sec captures all worker time.
    est_parallelism = 1.0 / max(_scale, 1e-9) if has_fine else 1.0

    stats = {
        'dataset_name': dataset_name,
        'num_videos': n_videos,
        'total_frames': total_frames,
        'wall_clock': {
            'total_sec': round(total_time, 3),
            'data_load_sec': round(total_data_time, 3),
            'model_infer_sec': round(total_model_time, 3),
            'pure_forward_sec': round(total_pure_forward_time, 3) if has_pure_forward else 0.0,
            'other_sec': round(total_time - total_data_time - total_model_time, 3),
            'data_load_pct': round(total_data_time / max(total_time, 1e-9) * 100, 1),
            'model_infer_pct': round(total_model_time / max(total_time, 1e-9) * 100, 1),
        },
        'per_video_avg': {
            'total_ms': round(total_time / max(n_videos, 1) * 1000, 1),
            'data_load_ms': round(total_data_time / max(n_videos, 1) * 1000, 1),
            'model_infer_ms': round(total_model_time / max(n_videos, 1) * 1000, 1),
            'pure_forward_ms': (round(total_pure_forward_time / max(n_videos, 1) * 1000, 1)
                                if has_pure_forward else 0.0),
        },
        'throughput': {
            'videos_per_sec': round(n_videos / max(total_time, 1e-9), 2),
            'frames_per_sec': round(total_frames / max(total_model_time, 1e-9), 2),
            'frames_per_sec_total': round(total_frames / max(total_time, 1e-9), 2),
            'pure_forward_frames_per_sec': (
                round(total_frames / max(total_pure_forward_time, 1e-9), 2)
                if has_pure_forward else 0.0
            ),
        },
    }

    if has_fine:
        # ── wall_clock: use scaled-to-data_load_sec values ──
        stats['wall_clock'].update({
            'io_decode_sec': round(total_io_wall, 3),
            'preprocess_sec': round(total_pre_wall, 3),
            'method_infer_sec': round(total_method_time, 3),
            'io_decode_pct': round(total_io_wall / max(total_time, 1e-9) * 100, 1),
            'preprocess_pct': round(total_pre_wall / max(total_time, 1e-9) * 100, 1),
            'method_infer_pct': round(total_method_time / max(total_time, 1e-9) * 100, 1),
            # Raw (worker-accumulated) values, kept for diagnostics.
            'io_decode_raw_sec': round(total_io_time, 3),
            'preprocess_raw_sec': round(total_pre_time, 3),
            'sample_total_raw_sec': round(total_sample_time, 3),
            'data_load_scale': round(_scale, 4),
            'est_dataloader_parallelism': round(est_parallelism, 2),
        })
        if has_gpu_prep:
            stats['wall_clock'].update({
                'gpu_preprocess_sec': round(total_gpu_prep_time, 3),
                'gpu_preprocess_pct': round(
                    total_gpu_prep_time / max(total_time, 1e-9) * 100, 1),
            })
        # ── per_video_avg:
        #   *_ms      : single-stream physical time per video
        #                (what the CPU actually spends on one video; divided by n_videos).
        #   *_wall_ms : how many ms of wall clock one video adds on average
        #                (single-stream_ms / parallelism).
        stats['per_video_avg'].update({
            'io_decode_ms': round(total_io_time / max(n_videos, 1) * 1000, 1),
            'preprocess_ms': round(total_pre_time / max(n_videos, 1) * 1000, 1),
            'sample_total_ms': round(total_sample_time / max(n_videos, 1) * 1000, 1),
            'method_infer_ms': round(
                (total_pre_time + total_model_time) / max(n_videos, 1) * 1000, 1),
            'io_decode_wall_ms': round(total_io_wall / max(n_videos, 1) * 1000, 1),
            'preprocess_wall_ms': round(total_pre_wall / max(n_videos, 1) * 1000, 1),
            'method_infer_wall_ms': round(total_method_time / max(n_videos, 1) * 1000, 1),
        })
        stats['throughput'].update({
            # "Method FPS" here is a *wall-clock* throughput:  total_frames
            # divided by the wall clock you'd have spent if only preprocess
            # and model forward were in the critical path (io_decode excluded).
            'method_frames_per_sec': round(total_frames / max(total_method_time, 1e-9), 2),
            # "Single-stream method FPS" assumes num_workers=1 and no IO overlap.
            # This is the most meaningful number to report as "our method's
            # processing speed per stream".
            'method_frames_per_sec_single_stream': round(
                total_frames / max(total_pre_time + total_model_time, 1e-9), 2),
            # ── Standalone preprocess FPS (IO/decode excluded) ──
            # single_stream: frames / preprocess_raw  (CPU physical time per worker,
            #                 i.e. what one single-threaded deployment would see)
            # wall:          frames / preprocess_wall (scaled to DataLoader wall clock,
            #                 reflecting N-worker parallel preprocessing throughput)
            'preprocess_frames_per_sec_single_stream': round(
                total_frames / max(total_pre_time, 1e-9), 2),
            'preprocess_frames_per_sec': round(
                total_frames / max(total_pre_wall, 1e-9), 2),
            # ── Standalone model-forward FPS (H2D + GPU forward only) ──
            # Identical numerical value to `frames_per_sec` above (which is already
            # model-only), duplicated here under a clearer name for symmetry with
            # `preprocess_frames_per_sec` and `method_frames_per_sec`.
            'model_frames_per_sec': round(
                total_frames / max(total_model_time, 1e-9), 2),
        })

    # Per-frame stats (model inference only, **single-stream physical time**)
    if total_frames > 0:
        _model_ms = total_model_time / total_frames * 1000
        stats['per_frame_avg'] = {
            'model_infer_ms': round(_model_ms, 2),
            # Direct FPS readings (= 1000 / per-frame-ms), duplicated here for
            # convenience so callers don't have to compute it from *_ms fields.
            'model_fps': round(1000.0 / max(_model_ms, 1e-9), 2),
        }
        if has_pure_forward:
            _pf_ms = total_pure_forward_time / total_frames * 1000
            stats['per_frame_avg'].update({
                'pure_forward_ms': round(_pf_ms, 2),
                'pure_forward_fps': round(1000.0 / max(_pf_ms, 1e-9), 2),
            })
        if has_fine:
            _pre_ms = total_pre_time / total_frames * 1000
            _method_ms = (total_pre_time + total_model_time) / total_frames * 1000
            # These three are "CPU time per frame" measured inside a single
            # worker - they are NOT divided by num_workers.  They represent
            # what a single-stream deployment would actually see per frame.
            stats['per_frame_avg'].update({
                'io_decode_ms': round(total_io_time / total_frames * 1000, 2),
                'preprocess_ms': round(_pre_ms, 2),
                'method_infer_ms': round(_method_ms, 2),
                # Direct FPS readings corresponding to the three *_ms fields.
                # Excludes IO/decode — this is the "fair processing speed".
                'preprocess_fps': round(1000.0 / max(_pre_ms, 1e-9), 2),
                'method_fps': round(1000.0 / max(_method_ms, 1e-9), 2),
            })
        if has_gpu_prep and total_frames > 0:
            _gpu_prep_ms = total_gpu_prep_time / total_frames * 1000
            # "Method itself" = GPU preprocess + PURE forward, both measured with
            # CUDA events and EXCLUDING H2D.
            # ⚠ total_model_time ALREADY includes the GPU-preprocess wall time
            # (the rebuild runs after t_model_start) plus per-clip H2D; adding
            # gpu_prep to model_time would double-count it. So we add gpu_prep to
            # PURE forward (H2D-free) instead — the clean algorithm cost.
            _fwd_ms = (total_pure_forward_time / total_frames * 1000
                       if has_pure_forward else _model_ms)
            _fwd_time = total_pure_forward_time if has_pure_forward else total_model_time
            _method_gpu_ms = _gpu_prep_ms + _fwd_ms
            stats['per_frame_avg'].update({
                'gpu_preprocess_ms': round(_gpu_prep_ms, 3),
                'gpu_preprocess_fps': round(1000.0 / max(_gpu_prep_ms, 1e-9), 2),
                'method_with_gpu_prep_ms': round(_method_gpu_ms, 3),
                'method_with_gpu_prep_fps': round(1000.0 / max(_method_gpu_ms, 1e-9), 2),
            })
            stats['throughput'].update({
                'gpu_preprocess_frames_per_sec': round(
                    total_frames / max(total_gpu_prep_time, 1e-9), 2),
                'method_with_gpu_prep_frames_per_sec': round(
                    total_frames / max(total_gpu_prep_time + _fwd_time, 1e-9), 2),
            })

    # Resolution-grouped stats.
    # Primary bucketing is by *original* resolution (pre-resize, from annotation),
    # so Waterloo4K read via a Resize1080p cache is still correctly counted as 4K.
    # Falls back to decoded tensor resolution when orig_height/width are missing.
    def _bucket_of(h, w):
        try:
            h = int(h); w = int(w)
        except Exception:
            return 'unknown'
        if h <= 0 or w <= 0:
            return 'unknown'
        if h <= 1200 and w <= 2100:
            return '1080p'
        if h <= 1600 and w <= 2600:
            return '2K'
        return '4K'

    def _build_res_group(records, key_h='height', key_w='width'):
        groups = {}
        for r in records:
            label = _bucket_of(r.get(key_h, 0), r.get(key_w, 0))
            groups.setdefault(label, []).append(r)
        result = {}
        for label, recs in sorted(groups.items()):
            n = len(recs)
            dt = sum(r['data_load_time'] for r in recs)
            mt = sum(r['model_infer_time'] for r in recs)
            pft = sum(float(r.get('pure_forward_time', 0.0) or 0.0) for r in recs)
            has_pft = pft > 1e-9
            tt = sum(r['total_time'] for r in recs)
            nf = sum(r.get('num_frames', 0) for r in recs)
            # IMPORTANT: io_t and pr_t are accumulated across workers (overcounted).
            # Per-frame / per-video *single-stream physical* values are still correct
            # (each sample's timing is self-consistent within its own worker).
            io_t = sum(float(r.get('io_decode_time', 0.0) or 0.0) for r in recs)
            pr_t = sum(float(r.get('preprocess_time', 0.0) or 0.0) for r in recs)
            # GPU preprocess time (main-thread wall clock, no worker scaling).
            gp_t = sum(float(r.get('gpu_preprocess_time', 0.0) or 0.0) for r in recs)
            has_gp = gp_t > 1e-9
            # Wall-clock scaled to this bucket (proportional split of data_load_time).
            if has_fine and (io_t + pr_t) > 1e-9:
                sc = dt / max(io_t + pr_t, 1e-9)
            else:
                sc = 1.0
            io_wall = io_t * sc
            pr_wall = pr_t * sc
            method_t = pr_wall + mt if has_fine else mt
            # Single-stream method time = preprocess_raw + model_forward
            method_single_t = pr_t + mt if has_fine else mt
            entry = {
                'num_videos': n,
                'total_frames': nf,
                'avg_total_ms': round(tt / max(n, 1) * 1000, 1),
                'avg_data_load_ms': round(dt / max(n, 1) * 1000, 1),
                'avg_model_infer_ms': round(mt / max(n, 1) * 1000, 1),
                'model_fps': round(nf / max(mt, 1e-9), 2),
                'total_fps': round(nf / max(tt, 1e-9), 2),
            }
            if has_pft:
                entry['avg_pure_forward_ms'] = round(pft / max(n, 1) * 1000, 1)
                entry['pure_forward_fps'] = round(nf / max(pft, 1e-9), 2)
            if nf > 0:
                entry['per_frame_model_ms'] = round(mt / nf * 1000, 2)
                if has_pft:
                    entry['per_frame_pure_forward_ms'] = round(pft / nf * 1000, 2)
            if has_fine:
                entry.update({
                    # Per-video single-stream physical time (what a single worker
                    # spends on one video).
                    'avg_io_decode_ms': round(io_t / max(n, 1) * 1000, 1),
                    'avg_preprocess_ms': round(pr_t / max(n, 1) * 1000, 1),
                    'avg_method_infer_ms': round(method_single_t / max(n, 1) * 1000, 1),
                    # Wall-clock scaled (proportional within data_load_sec).
                    'avg_io_decode_wall_ms': round(io_wall / max(n, 1) * 1000, 1),
                    'avg_preprocess_wall_ms': round(pr_wall / max(n, 1) * 1000, 1),
                    'avg_method_infer_wall_ms': round(method_t / max(n, 1) * 1000, 1),
                    # Single-stream FPS: the number to report as our processing speed.
                    'method_fps': round(nf / max(method_single_t, 1e-9), 2),
                    # Wall-clock FPS: what you'd observe with current parallelism.
                    'method_fps_wall': round(nf / max(method_t, 1e-9), 2),
                    # Standalone preprocess FPS (both single-stream & wall-clock)
                    # so each stage (preprocess / model / method) has a paired
                    # time+FPS readout in the same bucket.
                    'preprocess_fps': round(nf / max(pr_t, 1e-9), 2),
                    'preprocess_fps_wall': round(nf / max(pr_wall, 1e-9), 2),
                    'data_load_scale': round(sc, 4),
                })
                if nf > 0:
                    entry.update({
                        'per_frame_io_decode_ms': round(io_t / nf * 1000, 2),
                        'per_frame_preprocess_ms': round(pr_t / nf * 1000, 2),
                        'per_frame_method_ms': round(method_single_t / nf * 1000, 2),
                    })
            # GPU preprocess (single-clip only) — per-res gpu_prep + method(gpu_prep+forward).
            if has_gp and nf > 0:
                # method = gpu_prep + PURE forward (H2D-free). Do NOT use mt here:
                # model_infer already contains the gpu_prep wall + per-clip H2D.
                _fwd = pft if has_pft else mt
                entry.update({
                    'avg_gpu_preprocess_ms': round(gp_t / max(n, 1) * 1000, 2),
                    'per_frame_gpu_preprocess_ms': round(gp_t / nf * 1000, 3),
                    'per_frame_method_with_gpu_prep_ms': round((gp_t + _fwd) / nf * 1000, 3),
                    'method_with_gpu_prep_fps': round(nf / max(gp_t + _fwd, 1e-9), 2),
                })
            result[label] = entry
        return result

    has_orig = any(
        int(r.get('orig_height', 0) or 0) > 0 and int(r.get('orig_width', 0) or 0) > 0
        for r in timing_records
    )
    # Primary bucketing: original (annotation) resolution if available.
    if has_orig:
        stats['by_resolution'] = _build_res_group(
            timing_records, key_h='orig_height', key_w='orig_width',
        )
        # Also keep decoded-tensor bucket for reference (shape after frame-cache/resize).
        stats['by_decoded_resolution'] = _build_res_group(
            timing_records, key_h='height', key_w='width',
        )
    else:
        stats['by_resolution'] = _build_res_group(
            timing_records, key_h='height', key_w='width',
        )

    # ── readable_summary: concise, human-readable timing stats for reports ──
    # Keep only the most essential fields; for each resolution provide:
    #   io_decode_ms, preprocess_ms/fps, model_forward_ms/fps, method_ms/fps
    # When GPU preprocess is enabled, preprocess uses GPU time; otherwise CPU time.
    readable = {
        'overview': {
            'num_videos': n_videos,
            'total_frames': total_frames,
            'wall_clock_sec': round(total_time, 1),
        },
    }
    if total_frames > 0:
        _r_model_ms = total_model_time / total_frames * 1000
        if has_gpu_prep:
            _r_prep_ms = total_gpu_prep_time / total_frames * 1000
            # model_infer_time already includes gpu_preprocess, so
            # pure model forward = model_infer - gpu_preprocess.
            _r_fwd_ms = max(0.0, _r_model_ms - _r_prep_ms)
            # method = model_infer (which is gpu_prep + forward)
            _r_method_ms = _r_model_ms
        elif has_fine:
            _r_prep_ms = total_pre_time / total_frames * 1000
            _r_fwd_ms = _r_model_ms
            _r_method_ms = _r_prep_ms + _r_model_ms
        else:
            _r_prep_ms = 0.0
            _r_fwd_ms = _r_model_ms
            _r_method_ms = _r_model_ms
        _r_io_ms = (total_io_time / total_frames * 1000) if has_fine else 0.0
        readable['per_frame'] = {
            'io_decode_ms': round(_r_io_ms, 2),
            'preprocess_ms': round(_r_prep_ms, 2),
            'preprocess_fps': round(1000.0 / max(_r_prep_ms, 1e-9), 2) if _r_prep_ms > 0.01 else 0.0,
            'model_forward_ms': round(_r_fwd_ms, 2),
            'model_forward_fps': round(1000.0 / max(_r_fwd_ms, 1e-9), 2),
            'method_ms': round(_r_method_ms, 2),
            'method_fps': round(1000.0 / max(_r_method_ms, 1e-9), 2),
            'preprocess_source': 'gpu' if has_gpu_prep else 'cpu',
        }

    # Concise per-resolution-bucket version
    readable_by_res = {}
    for res_label, recs in sorted(
        {_bucket_of(r.get('orig_height', 0) if has_orig else r.get('height', 0),
                     r.get('orig_width', 0) if has_orig else r.get('width', 0)): []
         for r in timing_records}.items()
    ):
        pass  # just init keys
    # Re-group properly
    _res_groups: Dict[str, list] = {}
    for r in timing_records:
        if has_orig:
            lbl = _bucket_of(r.get('orig_height', 0), r.get('orig_width', 0))
        else:
            lbl = _bucket_of(r.get('height', 0), r.get('width', 0))
        _res_groups.setdefault(lbl, []).append(r)

    for res_label, recs in sorted(_res_groups.items()):
        n_r = len(recs)
        nf_r = sum(r.get('num_frames', 0) for r in recs)
        if nf_r <= 0:
            continue
        mt_r = sum(r['model_infer_time'] for r in recs)
        io_r = sum(float(r.get('io_decode_time', 0.0) or 0.0) for r in recs)
        pr_r = sum(float(r.get('preprocess_time', 0.0) or 0.0) for r in recs)
        gp_r = sum(float(r.get('gpu_preprocess_time', 0.0) or 0.0) for r in recs)
        _has_gp_r = gp_r > 1e-9

        _rm = mt_r / nf_r * 1000
        if _has_gp_r:
            _rp = gp_r / nf_r * 1000
            # model_infer_time includes gpu_preprocess; pure forward = model - gpu_prep
            _rfwd = max(0.0, _rm - _rp)
            _rmeth = _rm  # method = model_infer (gpu_prep + forward)
        elif has_fine:
            _rp = pr_r / nf_r * 1000
            _rfwd = _rm
            _rmeth = _rp + _rm
        else:
            _rp = 0.0
            _rfwd = _rm
            _rmeth = _rm
        _rio = (io_r / nf_r * 1000) if has_fine else 0.0

        readable_by_res[res_label] = {
            'num_videos': n_r,
            'total_frames': nf_r,
            'per_frame': {
                'io_decode_ms': round(_rio, 2),
                'preprocess_ms': round(_rp, 2),
                'preprocess_fps': round(1000.0 / max(_rp, 1e-9), 2) if _rp > 0.01 else 0.0,
                'model_forward_ms': round(_rfwd, 2),
                'model_forward_fps': round(1000.0 / max(_rfwd, 1e-9), 2),
                'method_ms': round(_rmeth, 2),
                'method_fps': round(1000.0 / max(_rmeth, 1e-9), 2),
                'preprocess_source': 'gpu' if _has_gp_r else 'cpu',
            },
        }
    readable['by_resolution'] = readable_by_res
    stats['readable_summary'] = readable

    return stats


def print_timing_report(timing_stats: Dict, log=None):
    """Print a human-readable timing report.

    Outputs two sections:
      1. Readable Summary — the key numbers you'd put in a paper/report.
      2. Detailed Breakdown — wall-clock diagnostics (compact).
    """
    _log = log or logger
    if not timing_stats:
        return

    ds = timing_stats.get('dataset_name', '')
    header = f'Timing Report [{ds}]' if ds else 'Timing Report'

    # ═══════════════════════════════════════════════════════════════════
    #  Section 1: Readable Summary (for paper / report)
    # ═══════════════════════════════════════════════════════════════════
    rs = timing_stats.get('readable_summary', {})
    if rs:
        ov = rs.get('overview', {})
        pf = rs.get('per_frame', {})
        by_res = rs.get('by_resolution', {})

        _log.info('=' * 70)
        _log.info(f'  {header}  —  Readable Summary')
        _log.info('=' * 70)
        _log.info(f'  Videos: {ov.get("num_videos", 0)}    '
                  f'Frames: {ov.get("total_frames", 0)}    '
                  f'Wall clock: {ov.get("wall_clock_sec", 0):.1f}s')

        if pf:
            prep_src = pf.get('preprocess_source', 'cpu').upper()
            _log.info('')
            _log.info(f'  ★ Per-frame average (preprocess on {prep_src}):')
            _log.info(f'    ┌─ IO + decode:       {pf.get("io_decode_ms", 0):>10.2f} ms   '
                      f'(not counted in method FPS)')
            _log.info(f'    ├─ Preprocess ({prep_src}):  {pf.get("preprocess_ms", 0):>10.2f} ms   '
                      f'→ {pf.get("preprocess_fps", 0):>8.2f} FPS')
            _log.info(f'    ├─ Model forward:     {pf.get("model_forward_ms", 0):>10.2f} ms   '
                      f'→ {pf.get("model_forward_fps", 0):>8.2f} FPS')
            _log.info(f'    └─ Method (pre+fwd):  {pf.get("method_ms", 0):>10.2f} ms   '
                      f'→ {pf.get("method_fps", 0):>8.2f} FPS  ★')

        if by_res:
            _log.info('')
            _log.info(f'  ★ Per-frame by resolution:')
            _log.info(f'    {"Resolution":<10} {"Videos":>6} {"Frames":>6}  '
                      f'{"IO ms":>8} {"Prep ms":>8} {"Prep FPS":>9} '
                      f'{"Model ms":>9} {"Model FPS":>10} '
                      f'{"Method ms":>10} {"Method FPS":>11}')
            _log.info(f'    {"-"*10} {"-"*6} {"-"*6}  '
                      f'{"-"*8} {"-"*8} {"-"*9} '
                      f'{"-"*9} {"-"*10} '
                      f'{"-"*10} {"-"*11}')
            for rl, rd in sorted(by_res.items()):
                rpf = rd.get('per_frame', {})
                _log.info(
                    f'    {rl:<10} {rd.get("num_videos", 0):>6} {rd.get("total_frames", 0):>6}  '
                    f'{rpf.get("io_decode_ms", 0):>8.2f} '
                    f'{rpf.get("preprocess_ms", 0):>8.2f} '
                    f'{rpf.get("preprocess_fps", 0):>9.2f} '
                    f'{rpf.get("model_forward_ms", 0):>9.2f} '
                    f'{rpf.get("model_forward_fps", 0):>10.2f} '
                    f'{rpf.get("method_ms", 0):>10.2f} '
                    f'{rpf.get("method_fps", 0):>11.2f}')

        _log.info('')
        _log.info('=' * 70)

    # ═══════════════════════════════════════════════════════════════════
    #  Section 2: Detailed Breakdown (diagnostics)
    # ═══════════════════════════════════════════════════════════════════
    wc = timing_stats.get('wall_clock', {})
    has_fine = 'preprocess_sec' in wc
    _log.info(f'  Detailed Breakdown:')
    _log.info(f'    Wall clock: {_format_time(wc.get("total_sec", 0))}  '
              f'(data_load={_format_time(wc.get("data_load_sec", 0))} '
              f'[{wc.get("data_load_pct", 0):.0f}%]  '
              f'model={_format_time(wc.get("model_infer_sec", 0))} '
              f'[{wc.get("model_infer_pct", 0):.0f}%])')
    if has_fine:
        _log.info(f'    DataLoader parallelism ≈ {wc.get("est_dataloader_parallelism", 1.0):.1f}x  '
                  f'(io_raw={_format_time(wc.get("io_decode_raw_sec", 0))}  '
                  f'pre_raw={_format_time(wc.get("preprocess_raw_sec", 0))})')
    if 'gpu_preprocess_sec' in wc:
        _log.info(f'    GPU preprocess wall: {_format_time(wc.get("gpu_preprocess_sec", 0))} '
                  f'({wc.get("gpu_preprocess_pct", 0):.1f}%)')

    by_res_detail = timing_stats.get('by_resolution', {})
    if by_res_detail:
        for res_label, rs_data in sorted(by_res_detail.items()):
            _log.info(f'    [{res_label}] {rs_data["num_videos"]}v/{rs_data["total_frames"]}f  '
                      f'model={rs_data.get("per_frame_model_ms", 0):.2f}ms/f  '
                      f'total_fps={rs_data.get("total_fps", 0):.2f}')

    _log.info('=' * 70)


# ---------------------------------------------------------------------------
# Temporal pooling strategies for Test-Time Augmentation (TTA)
# ---------------------------------------------------------------------------
# All pooling functions accept:
#   clip_scores: torch.Tensor  [B, K]  — per-clip scores for each video in the batch
# and return:
#   torch.Tensor  [B]  — pooled video-level score
#
# Strategies:
#   mean           — simple average (baseline)
#   recency        — exponential recency weighting (later clips matter more)
#   forgetting     — exponential forgetting memory weighting
#   softmin        — softmin worst-clip emphasis weighting
#   adaptive       — full Recency-Hysteresis Adaptive Temporal Pooling
#   adaptive_simple — simplified adaptive (no softmin fusion, just weighted mean)
#   percentile_X   — X-th percentile (e.g. percentile_10 = near-worst)
#   trimmed_mean_X — trimmed mean removing X% worst and best
# ---------------------------------------------------------------------------

# Registry: name → callable(clip_scores, **kwargs) → [B]
_TTA_POOLING_REGISTRY = {}


def register_tta_pooling(name: str):
    """Decorator to register a TTA pooling strategy."""
    def decorator(fn):
        _TTA_POOLING_REGISTRY[name] = fn
        return fn
    return decorator


def get_tta_pooling_fn(name: str):
    """Get a registered TTA pooling function by name."""
    if name not in _TTA_POOLING_REGISTRY:
        available = list(_TTA_POOLING_REGISTRY.keys())
        raise ValueError(f"Unknown TTA pooling strategy: '{name}'. Available: {available}")
    return _TTA_POOLING_REGISTRY[name]


def list_tta_pooling_strategies() -> list:
    """Return list of available TTA pooling strategy names."""
    return sorted(_TTA_POOLING_REGISTRY.keys())


@register_tta_pooling('mean')
def _pool_mean(clip_scores: torch.Tensor, **kwargs) -> torch.Tensor:
    """Simple mean pooling (baseline)."""
    return clip_scores.mean(dim=1)


@register_tta_pooling('recency')
def _pool_recency(clip_scores: torch.Tensor, beta: float = 1.5, **kwargs) -> torch.Tensor:
    """
    Recency-weighted pooling: later clips receive exponentially higher weight.
    r_t = exp(β * (t-1) / (T-1))
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    device = clip_scores.device
    t_idx = torch.arange(K, device=device, dtype=clip_scores.dtype)
    recency = torch.exp(beta * t_idx / (K - 1))  # [K]
    weights = recency / recency.sum()  # normalized [K]
    return (clip_scores * weights.unsqueeze(0)).sum(dim=1)  # [B]


@register_tta_pooling('forgetting')
def _pool_forgetting(clip_scores: torch.Tensor,
                     tau_f: float = 6.0,
                     lambda2: float = 1.0,
                     **kwargs) -> torch.Tensor:
    """
    Exponential Forgetting Memory weighting.
    Bad clips leave a residual memory that decays exponentially.
    w_t = 1 + λ2 * m_t, then normalized.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    device = clip_scores.device
    dtype = clip_scores.dtype

    # Normalize scores to [0,1] range per batch for badness computation
    q_min = clip_scores.min(dim=1, keepdim=True).values
    q_max = clip_scores.max(dim=1, keepdim=True).values
    q_range = (q_max - q_min).clamp(min=1e-8)
    q_norm = (clip_scores - q_min) / q_range  # [B, K] in [0, 1]

    d_bad = 1.0 - q_norm  # [B, K]
    rho = float(np.exp(-1.0 / tau_f))

    # Forward pass to compute memory
    memory = torch.zeros_like(clip_scores)  # [B, K]
    memory[:, 0] = d_bad[:, 0]
    for t in range(1, K):
        memory[:, t] = torch.maximum(
            torch.tensor(rho, device=device, dtype=dtype) * memory[:, t - 1],
            d_bad[:, t],
        )

    weights = 1.0 + lambda2 * memory  # [B, K]
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('softmin')
def _pool_softmin(clip_scores: torch.Tensor,
                  kappa: float = 10.0,
                  **kwargs) -> torch.Tensor:
    """
    Softmin worst-clip emphasis: weight each clip inversely to its quality.
    α_k = exp(-κ q_k) / Σ exp(-κ q_j)
    High κ → focus on worst clip; κ=0 → mean.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)

    # Normalize scores to [0, 1] per batch
    q_min = clip_scores.min(dim=1, keepdim=True).values
    q_max = clip_scores.max(dim=1, keepdim=True).values
    q_range = (q_max - q_min).clamp(min=1e-8)
    q_norm = (clip_scores - q_min) / q_range

    weights = torch.softmax(-kappa * q_norm, dim=1)  # [B, K]
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('adaptive_simple')
def _pool_adaptive_simple(clip_scores: torch.Tensor,
                          tau_f: float = 6.0,
                          beta: float = 1.5,
                          lambda1: float = 0.5,
                          lambda2: float = 1.0,
                          **kwargs) -> torch.Tensor:
    """
    Simplified Recency-Hysteresis Adaptive Pooling (no softmin fusion).
    w_t = r_t * (1 + λ1*d_t) * (1 + λ2*m_t)
    Q = Σ(w_t * q_t) / Σ(w_t)
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    device = clip_scores.device
    dtype = clip_scores.dtype

    # Normalize to [0, 1]
    q_min = clip_scores.min(dim=1, keepdim=True).values
    q_max = clip_scores.max(dim=1, keepdim=True).values
    q_range = (q_max - q_min).clamp(min=1e-8)
    q_norm = (clip_scores - q_min) / q_range

    d_bad = 1.0 - q_norm
    rho = float(np.exp(-1.0 / tau_f))

    # Memory
    memory = torch.zeros_like(clip_scores)
    memory[:, 0] = d_bad[:, 0]
    for t in range(1, K):
        memory[:, t] = torch.maximum(
            torch.tensor(rho, device=device, dtype=dtype) * memory[:, t - 1],
            d_bad[:, t],
        )

    # Recency
    t_idx = torch.arange(K, device=device, dtype=dtype)
    recency = torch.exp(beta * t_idx / max(K - 1, 1))  # [K]

    # Weights
    weights = recency.unsqueeze(0) * (1.0 + lambda1 * d_bad) * (1.0 + lambda2 * memory)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('adaptive')
def _pool_adaptive_full(clip_scores: torch.Tensor,
                        tau_f: float = 6.0,
                        beta: float = 1.5,
                        kappa: float = 10.0,
                        lambda1: float = 0.5,
                        lambda2: float = 1.0,
                        window: int = 5,
                        a_coef: float = 2.0,
                        b_coef: float = 1.0,
                        **kwargs) -> torch.Tensor:
    """
    Full Recency-Hysteresis Adaptive Temporal Pooling.
    Includes softmin local worst-clip emphasis and adaptive q* fusion.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    device = clip_scores.device
    dtype = clip_scores.dtype

    # Normalize to [0, 1]
    q_min = clip_scores.min(dim=1, keepdim=True).values
    q_max = clip_scores.max(dim=1, keepdim=True).values
    q_range = (q_max - q_min).clamp(min=1e-8)
    q_norm = (clip_scores - q_min) / q_range

    d_bad = 1.0 - q_norm
    rho = float(np.exp(-1.0 / tau_f))

    # Memory
    memory = torch.zeros_like(clip_scores)
    memory[:, 0] = d_bad[:, 0]
    for t in range(1, K):
        memory[:, t] = torch.maximum(
            torch.tensor(rho, device=device, dtype=dtype) * memory[:, t - 1],
            d_bad[:, t],
        )

    # Recency
    t_idx = torch.arange(K, device=device, dtype=dtype)
    recency = torch.exp(beta * t_idx / max(K - 1, 1))

    # Softmin local quality
    half = window // 2
    q_soft = torch.zeros_like(q_norm)
    for t in range(K):
        l = max(0, t - half)
        r = min(K, t + half + 1)
        local_q = q_norm[:, l:r]  # [B, local_len]
        w_soft = torch.softmax(-kappa * local_q, dim=1)
        q_soft[:, t] = (w_soft * local_q).sum(dim=1)

    # Adaptive fusion: γ = sigmoid(a*m + b*d)
    gamma_logits = a_coef * memory + b_coef * d_bad
    gamma = torch.sigmoid(gamma_logits)
    q_star_norm = (1.0 - gamma) * q_norm + gamma * q_soft

    # De-normalize q_star back to original score scale
    q_star = q_star_norm * q_range + q_min

    # Weights
    weights = recency.unsqueeze(0) * (1.0 + lambda1 * d_bad) * (1.0 + lambda2 * memory)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('recency_softmin')
def _pool_recency_softmin(clip_scores: torch.Tensor,
                          beta: float = 1.5,
                          kappa: float = 10.0,
                          **kwargs) -> torch.Tensor:
    """
    Combined recency + softmin weighting (no forgetting memory).
    w_t = r_t * exp(-κ * q_norm_t), normalized.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    device = clip_scores.device
    dtype = clip_scores.dtype

    # Normalize to [0, 1]
    q_min = clip_scores.min(dim=1, keepdim=True).values
    q_max = clip_scores.max(dim=1, keepdim=True).values
    q_range = (q_max - q_min).clamp(min=1e-8)
    q_norm = (clip_scores - q_min) / q_range

    t_idx = torch.arange(K, device=device, dtype=dtype)
    recency = torch.exp(beta * t_idx / max(K - 1, 1))
    softmin_w = torch.exp(-kappa * q_norm)  # [B, K]

    weights = recency.unsqueeze(0) * softmin_w
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (clip_scores * weights).sum(dim=1)


# ---------------------------------------------------------------------------
# New pooling strategies (matching test_time_vqa_multi_strategy.md)
# ---------------------------------------------------------------------------

@register_tta_pooling('mean_softmin_fused')
def _pool_mean_softmin_fused(clip_scores: torch.Tensor,
                              alpha: float = 0.3,
                              tau: float = 0.0,
                              **kwargs) -> torch.Tensor:
    """
    MD Scheme 2: Mean + Softmin fusion pooling.
    video_score = (1-α) * mean + α * softmin
    tau=0 → auto set to 0.1 * (max-min).
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    s_mean = clip_scores.mean(dim=1)  # [B]
    # Dynamic tau
    s_min = clip_scores.min(dim=1).values
    s_max = clip_scores.max(dim=1).values
    s_range = (s_max - s_min).clamp(min=1e-8)
    if tau <= 0:
        tau_val = 0.1 * s_range  # [B]
    else:
        tau_val = torch.full_like(s_mean, tau)
    # Softmin: -τ * log(mean(exp(-s/τ)))
    neg_s_over_tau = -clip_scores / tau_val.unsqueeze(1).clamp(min=1e-8)  # [B,K]
    log_mean_exp = torch.logsumexp(neg_s_over_tau, dim=1) - float(np.log(K))
    s_softmin = -tau_val * log_mean_exp  # [B]
    return (1 - alpha) * s_mean + alpha * s_softmin


@register_tta_pooling('mean_softmin_adaptive')
def _pool_mean_softmin_adaptive(clip_scores: torch.Tensor,
                                 alpha_max: float = 0.5,
                                 c_alpha: float = 5.0,
                                 tau: float = 0.0,
                                 **kwargs) -> torch.Tensor:
    """
    MD Scheme 2B: Mean + Softmin adaptive-α fusion.
    α = clip(c * std(scores), 0, α_max); when volatility is high, weight softmin more.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    s_mean = clip_scores.mean(dim=1)
    s_std = clip_scores.std(dim=1)
    alpha = (c_alpha * s_std).clamp(0, alpha_max)  # [B]
    s_min = clip_scores.min(dim=1).values
    s_max = clip_scores.max(dim=1).values
    s_range = (s_max - s_min).clamp(min=1e-8)
    tau_val = 0.1 * s_range if tau <= 0 else torch.full_like(s_mean, tau)
    neg_s_over_tau = -clip_scores / tau_val.unsqueeze(1).clamp(min=1e-8)
    log_mean_exp = torch.logsumexp(neg_s_over_tau, dim=1) - float(np.log(K))
    s_softmin = -tau_val * log_mean_exp
    return (1 - alpha) * s_mean + alpha * s_softmin


@register_tta_pooling('weighted_badness')
def _pool_weighted_badness(clip_scores: torch.Tensor,
                            beta_w: float = 0.7,
                            **kwargs) -> torch.Tensor:
    """
    MD Scheme 3: Badness-aware weighted mean.
    Low-score clips get larger weight: w_k = exp(-β * z_k), z = standardized score.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    mu = clip_scores.mean(dim=1, keepdim=True)
    sigma = clip_scores.std(dim=1, keepdim=True).clamp(min=1e-8)
    z = (clip_scores - mu) / sigma  # [B, K]
    weights = torch.exp(-beta_w * z)  # low score z<0 → larger weight
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('memory_v2')
def _pool_memory_v2(clip_scores: torch.Tensor,
                     beta_mem: float = 0.9,
                     lambda_m: float = 1.0,
                     lambda_b: float = 0.5,
                     **kwargs) -> torch.Tensor:
    """
    MD Scheme 5: Badness Memory weighted pooling (version B: joint weight of memory + current badness).
    m_k = max(β*m_{k-1}, b_k), w_k = exp(λ_m*m_k + λ_b*b_k).
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    dtype = clip_scores.dtype
    device = clip_scores.device
    # Compute badness after standardization
    mu = clip_scores.mean(dim=1, keepdim=True)
    sigma = clip_scores.std(dim=1, keepdim=True).clamp(min=1e-8)
    z = (clip_scores - mu) / sigma
    b = z.max(dim=1, keepdim=True).values - z  # badness [B,K]
    # Memory recurrence
    memory = torch.zeros_like(clip_scores)
    memory[:, 0] = b[:, 0]
    for t in range(1, K):
        memory[:, t] = torch.maximum(
            torch.tensor(beta_mem, device=device, dtype=dtype) * memory[:, t - 1],
            b[:, t],
        )
    weights = torch.exp(lambda_m * memory + lambda_b * b)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('local_softmin_mean')
def _pool_local_softmin_mean(clip_scores: torch.Tensor,
                              delta: float = 0.3,
                              tau_local: float = 0.0,
                              **kwargs) -> torch.Tensor:
    """
    MD Scheme 6: Local Softmin Smoothing → mean pooling.
    s̃_k = (1-δ)*s_k + δ*local_softmin_k, then take mean.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    s_range = (clip_scores.max(dim=1).values - clip_scores.min(dim=1).values).clamp(min=1e-8)
    if tau_local <= 0:
        tau_val = 0.1 * s_range  # [B]
    else:
        tau_val = torch.full((B,), tau_local, device=clip_scores.device, dtype=clip_scores.dtype)
    # Local softmin for each clip (window=3: k-1, k, k+1)
    enhanced = torch.zeros_like(clip_scores)
    for k in range(K):
        lo = max(0, k - 1)
        hi = min(K, k + 2)
        nbr = clip_scores[:, lo:hi]  # [B, 1~3]
        neg_over_tau = -nbr / tau_val.unsqueeze(1).clamp(min=1e-8)
        local_sm = -tau_val * (torch.logsumexp(neg_over_tau, dim=1) - float(np.log(hi - lo)))
        enhanced[:, k] = (1 - delta) * clip_scores[:, k] + delta * local_sm
    return enhanced.mean(dim=1)


@register_tta_pooling('memory_recency')
def _pool_memory_recency(clip_scores: torch.Tensor,
                          beta_mem: float = 0.9,
                          lambda_m: float = 1.0,
                          lambda_b: float = 0.5,
                          gamma: float = 0.2,
                          **kwargs) -> torch.Tensor:
    """
    MD Scheme 5+7 combination: Badness Memory + Recency Prior.
    w_k = exp(λ_m*m_k + λ_b*b_k) * exp(γ*k/K).
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)
    dtype = clip_scores.dtype
    device = clip_scores.device
    mu = clip_scores.mean(dim=1, keepdim=True)
    sigma = clip_scores.std(dim=1, keepdim=True).clamp(min=1e-8)
    z = (clip_scores - mu) / sigma
    b = z.max(dim=1, keepdim=True).values - z
    memory = torch.zeros_like(clip_scores)
    memory[:, 0] = b[:, 0]
    for t in range(1, K):
        memory[:, t] = torch.maximum(
            torch.tensor(beta_mem, device=device, dtype=dtype) * memory[:, t - 1],
            b[:, t],
        )
    t_idx = torch.arange(K, device=device, dtype=dtype)
    recency = torch.exp(gamma * t_idx / max(K - 1, 1))  # [K]
    weights = torch.exp(lambda_m * memory + lambda_b * b) * recency.unsqueeze(0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('coarse_to_fine')
def _pool_coarse_to_fine(clip_scores: torch.Tensor,
                          k_coarse: int = 5,
                          refine_boost: float = 2.0,
                          refine_radius: int = 1,
                          **kwargs) -> torch.Tensor:
    """
    MD Scheme 4: Coarse-to-Fine two-stage pooling.
    Without changing the data pipeline, simulate two stages within K clips:
      1. Coarse stage: uniformly select k_coarse clip positions (stride-based selection)
      2. Locate the worst clip position from the coarse stage
      3. Fine stage: assign refine_boost times higher weight to clips in the worst position's neighborhood
      4. Weighted average
    Prerequisite: the data pipeline has already uniformly sampled K clips (K >= k_coarse).
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)

    # Coarse stage: pick k_coarse clips at equal intervals from the K clips
    k_c = min(k_coarse, K)
    coarse_indices = torch.linspace(0, K - 1, k_c).long()
    coarse_scores = clip_scores[:, coarse_indices]  # [B, k_c]

    # Locate the position of the worst coarse-stage clip per batch (index in the original K sequence)
    worst_coarse_pos = coarse_indices[coarse_scores.argmin(dim=1)]  # [B]

    # Fine stage: assign higher weight to the neighborhood of the worst position
    weights = torch.ones_like(clip_scores)  # [B, K]
    for b_idx in range(B):
        wc = worst_coarse_pos[b_idx].item()
        lo = max(0, wc - refine_radius)
        hi = min(K, wc + refine_radius + 1)
        weights[b_idx, lo:hi] = refine_boost

    weights = weights / weights.sum(dim=1, keepdim=True)
    return (clip_scores * weights).sum(dim=1)


@register_tta_pooling('coarse_to_fine_softmin')
def _pool_coarse_to_fine_softmin(clip_scores: torch.Tensor,
                                   k_coarse: int = 5,
                                   refine_boost: float = 2.0,
                                   refine_radius: int = 1,
                                   alpha: float = 0.3,
                                   **kwargs) -> torch.Tensor:
    """
    MD Scheme 4 + Scheme 2 combination: Coarse-to-Fine weighting + Mean/Softmin fusion.
    """
    B, K = clip_scores.shape
    if K == 1:
        return clip_scores.squeeze(1)

    # Coarse-to-Fine weighted mean
    k_c = min(k_coarse, K)
    coarse_indices = torch.linspace(0, K - 1, k_c).long()
    coarse_scores = clip_scores[:, coarse_indices]
    worst_coarse_pos = coarse_indices[coarse_scores.argmin(dim=1)]

    weights = torch.ones_like(clip_scores)
    for b_idx in range(B):
        wc = worst_coarse_pos[b_idx].item()
        lo = max(0, wc - refine_radius)
        hi = min(K, wc + refine_radius + 1)
        weights[b_idx, lo:hi] = refine_boost
    weights = weights / weights.sum(dim=1, keepdim=True)
    s_ctf = (clip_scores * weights).sum(dim=1)

    # Softmin
    s_range = (clip_scores.max(dim=1).values - clip_scores.min(dim=1).values).clamp(min=1e-8)
    tau_val = 0.1 * s_range
    neg_s = -clip_scores / tau_val.unsqueeze(1).clamp(min=1e-8)
    log_me = torch.logsumexp(neg_s, dim=1) - float(np.log(K))
    s_softmin = -tau_val * log_me

    return (1 - alpha) * s_ctf + alpha * s_softmin


# ---------------------------------------------------------------------------
# TTA Scheme definitions (name → config dict)
# ---------------------------------------------------------------------------
# Each scheme defines:
#   num_clips : int            — number of temporal clips to sample
#   tta_hflip : bool           — whether to also do horizontal flip TTA
#   pooling   : str            — pooling strategy name (from registry)
#   pool_kwargs : dict          — extra kwargs for the pooling function
#   description : str           — human-readable description
# ---------------------------------------------------------------------------

TTA_SCHEMES = {
    'S0_baseline': {
        'num_clips': 2,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'Baseline: 2 clips, mean (current default)',
    },
    'S1_clip4_mean': {
        'num_clips': 4,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': '4 clips, mean pooling',
    },
    'S2_clip8_mean': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': '8 clips, mean pooling',
    },
    'S3_clip16_mean': {
        'num_clips': 16,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': '16 clips, mean pooling',
    },
    'S4_clip4_hflip': {
        'num_clips': 4,
        'tta_hflip': True,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': '4 clips, mean + horizontal flip TTA',
    },
    'S5_clip8_recency': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'recency',
        'pool_kwargs': {'beta': 1.5},
        'description': '8 clips, recency weighting (β=1.5)',
    },
    'S6_clip8_forgetting': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'forgetting',
        'pool_kwargs': {'tau_f': 6.0, 'lambda2': 1.0},
        'description': '8 clips, exponential forgetting memory (τ_f=6, λ2=1.0)',
    },
    'S7_clip8_softmin': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'softmin',
        'pool_kwargs': {'kappa': 10.0},
        'description': '8 clips, softmin worst-clip emphasis (κ=10)',
    },
    'S8_clip8_adaptive_simple': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'adaptive_simple',
        'pool_kwargs': {'tau_f': 6.0, 'beta': 1.5, 'lambda1': 0.5, 'lambda2': 1.0},
        'description': '8 clips, simplified adaptive pooling (recency + forgetting)',
    },
    'S9_clip8_adaptive_full': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'adaptive',
        'pool_kwargs': {'tau_f': 6.0, 'beta': 1.5, 'kappa': 10.0,
                        'lambda1': 0.5, 'lambda2': 1.0, 'window': 5},
        'description': '8 clips, full Recency-Hysteresis Adaptive Pooling',
    },
    'S10_clip8_recency_softmin': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'recency_softmin',
        'pool_kwargs': {'beta': 1.5, 'kappa': 10.0},
        'description': '8 clips, recency + softmin combined',
    },
    'S11_clip16_adaptive_simple': {
        'num_clips': 16,
        'tta_hflip': False,
        'pooling': 'adaptive_simple',
        'pool_kwargs': {'tau_f': 8.0, 'beta': 1.5, 'lambda1': 0.5, 'lambda2': 1.0},
        'description': '16 clips, simplified adaptive pooling (τ_f=8)',
    },
    'S12_clip8_hflip_adaptive': {
        'num_clips': 8,
        'tta_hflip': True,
        'pooling': 'adaptive_simple',
        'pool_kwargs': {'tau_f': 6.0, 'beta': 1.5, 'lambda1': 0.5, 'lambda2': 1.0},
        'description': '8 clips, adaptive + horizontal flip TTA',
    },
    # ======================================================================
    # M-series: Multi-strategy ablation schemes (test_time_vqa_multi_strategy.md)
    # Strictly corresponds to Schemes 0-7, 10 in the markdown doc
    # All schemes use 8 adjacent-frame clips, matching the training distribution
    # ======================================================================
    # ── Scheme 0: Baseline ──
    'M0_baseline_center': {
        'num_clips': 1,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 0: single center clip, strictest baseline',
    },
    # ── Scheme 1: multi-clip mean ──
    'M1a_clip5_mean': {
        'num_clips': 5,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 5 clips, mean pooling',
    },
    'M1b_clip7_mean': {
        'num_clips': 7,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 7 clips, mean pooling',
    },
    'M1c_clip10_mean': {
        'num_clips': 10,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 10 clips, mean pooling',
    },
    'M1d_clip2_mean': {
        'num_clips': 2,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 2 clips, mean pooling',
    },
    'M1e_clip3_mean': {
        'num_clips': 3,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 3 clips, mean pooling',
    },
    'M1f_clip4_mean': {
        'num_clips': 4,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 4 clips, mean pooling',
    },
    'M1g_clip6_mean': {
        'num_clips': 6,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 6 clips, mean pooling',
    },
    'M1h_clip8_mean': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 8 clips, mean pooling',
    },
    'M1i_clip9_mean': {
        'num_clips': 9,
        'tta_hflip': False,
        'pooling': 'mean',
        'pool_kwargs': {},
        'description': 'MD Scheme 1: 9 clips, mean pooling',
    },
    # ── Scheme 2: Mean + Softmin fusion ──
    'M2a_clip5_mean_softmin': {
        'num_clips': 5,
        'tta_hflip': False,
        'pooling': 'mean_softmin_fused',
        'pool_kwargs': {'alpha': 0.3, 'tau': 0.0},
        'description': 'MD Scheme 2A: 5 clips, mean+softmin fusion (α=0.3, τ=auto)',
    },
    'M2b_clip8_mean_softmin': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'mean_softmin_fused',
        'pool_kwargs': {'alpha': 0.3, 'tau': 0.0},
        'description': 'MD Scheme 2A: 8 clips, mean+softmin fusion (α=0.3, τ=auto)',
    },
    'M2c_clip5_mean_softmin_adaptive': {
        'num_clips': 5,
        'tta_hflip': False,
        'pooling': 'mean_softmin_adaptive',
        'pool_kwargs': {'alpha_max': 0.5, 'c_alpha': 5.0, 'tau': 0.0},
        'description': 'MD Scheme 2B: 5 clips, mean+softmin adaptive α (α_max=0.5)',
    },
    'M2d_clip8_mean_softmin_adaptive': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'mean_softmin_adaptive',
        'pool_kwargs': {'alpha_max': 0.5, 'c_alpha': 5.0, 'tau': 0.0},
        'description': 'MD Scheme 2B: 8 clips, mean+softmin adaptive α (α_max=0.5)',
    },
    # ── Scheme 3: Badness-aware weighted mean ──
    'M3a_clip5_weighted_badness': {
        'num_clips': 5,
        'tta_hflip': False,
        'pooling': 'weighted_badness',
        'pool_kwargs': {'beta_w': 0.7},
        'description': 'MD Scheme 3: 5 clips, badness-aware weighting (β=0.7)',
    },
    'M3b_clip8_weighted_badness': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'weighted_badness',
        'pool_kwargs': {'beta_w': 0.7},
        'description': 'MD Scheme 3: 8 clips, badness-aware weighting (β=0.7)',
    },
    'M3c_clip8_weighted_badness_strong': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'weighted_badness',
        'pool_kwargs': {'beta_w': 1.0},
        'description': 'MD Scheme 3: 8 clips, badness-aware weighting (β=1.0, stronger)',
    },
    # ── Scheme 5: Badness Memory weighting ──
    'M5a_clip5_memory_v2': {
        'num_clips': 5,
        'tta_hflip': False,
        'pooling': 'memory_v2',
        'pool_kwargs': {'beta_mem': 0.9, 'lambda_m': 1.0, 'lambda_b': 0.5},
        'description': 'MD Scheme 5B: 5 clips, memory+badness joint (β=0.9)',
    },
    'M5b_clip8_memory_v2': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'memory_v2',
        'pool_kwargs': {'beta_mem': 0.9, 'lambda_m': 1.0, 'lambda_b': 0.5},
        'description': 'MD Scheme 5B: 8 clips, memory+badness joint (β=0.9)',
    },
    'M5c_clip8_memory_v2_strong': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'memory_v2',
        'pool_kwargs': {'beta_mem': 0.95, 'lambda_m': 1.5, 'lambda_b': 0.5},
        'description': 'MD Scheme 5B: 8 clips, memory+badness (β=0.95, λ_m=1.5, longer memory)',
    },
    # ── Scheme 6: Local Softmin Smoothing ──
    'M6a_clip5_local_softmin': {
        'num_clips': 5,
        'tta_hflip': False,
        'pooling': 'local_softmin_mean',
        'pool_kwargs': {'delta': 0.3, 'tau_local': 0.0},
        'description': 'MD Scheme 6: 5 clips, local softmin smoothing→mean (δ=0.3)',
    },
    'M6b_clip8_local_softmin': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'local_softmin_mean',
        'pool_kwargs': {'delta': 0.3, 'tau_local': 0.0},
        'description': 'MD Scheme 6: 8 clips, local softmin smoothing→mean (δ=0.3)',
    },
    # ── Scheme 5+7: Memory + Recency combination ──
    'M7a_clip8_memory_recency': {
        'num_clips': 8,
        'tta_hflip': False,
        'pooling': 'memory_recency',
        'pool_kwargs': {'beta_mem': 0.9, 'lambda_m': 1.0, 'lambda_b': 0.5, 'gamma': 0.2},
        'description': 'MD Scheme 5+7: 8 clips, memory+recency (β=0.9, γ=0.2)',
    },
    # ── Scheme 4: Coarse-to-Fine two-stage ──
    'M4a_clip10_coarse_to_fine': {
        'num_clips': 10,
        'tta_hflip': False,
        'pooling': 'coarse_to_fine',
        'pool_kwargs': {'k_coarse': 5, 'refine_boost': 2.0, 'refine_radius': 1},
        'description': 'MD Scheme 4A: 10 clips, coarse-to-fine (coarse 5 + refine worst neighborhood, boost=2x)',
    },
    'M4b_clip10_coarse_to_fine_strong': {
        'num_clips': 10,
        'tta_hflip': False,
        'pooling': 'coarse_to_fine',
        'pool_kwargs': {'k_coarse': 5, 'refine_boost': 3.0, 'refine_radius': 2},
        'description': 'MD Scheme 4B: 10 clips, coarse-to-fine (boost=3x, larger neighborhood)',
    },
    'M4c_clip10_ctf_softmin': {
        'num_clips': 10,
        'tta_hflip': False,
        'pooling': 'coarse_to_fine_softmin',
        'pool_kwargs': {'k_coarse': 5, 'refine_boost': 2.0, 'refine_radius': 1, 'alpha': 0.3},
        'description': 'MD Scheme 4+2: 10 clips, coarse-to-fine + softmin fusion (α=0.3)',
    },
    # ── Scheme 10 (simplified): center + multi-clip dual-path fusion ──
    # Note: true multi-path fusion requires post-processing; here we approximate with the existing framework using 16 clips
    'M10a_clip16_mean_softmin': {
        'num_clips': 16,
        'tta_hflip': False,
        'pooling': 'mean_softmin_fused',
        'pool_kwargs': {'alpha': 0.3, 'tau': 0.0},
        'description': 'MD Scheme 10 approx.: 16 clips, mean+softmin fusion (high coverage)',
    },
    'M10b_clip16_memory_v2': {
        'num_clips': 16,
        'tta_hflip': False,
        'pooling': 'memory_v2',
        'pool_kwargs': {'beta_mem': 0.9, 'lambda_m': 1.0, 'lambda_b': 0.5},
        'description': 'MD Scheme 10 approx.: 16 clips, memory+badness joint (high coverage)',
    },
}


def get_tta_scheme(name: str) -> dict:
    """Get a TTA scheme definition by name."""
    if name not in TTA_SCHEMES:
        available = list(TTA_SCHEMES.keys())
        raise ValueError(f"Unknown TTA scheme: '{name}'. Available: {available}")
    return TTA_SCHEMES[name]


def list_tta_schemes() -> list:
    """Return list of available TTA scheme names with descriptions."""
    return [(name, cfg['description']) for name, cfg in TTA_SCHEMES.items()]


_CLIP_DIM_KEYS_6D = {'resize_dis', 'resize_ref'}
_CLIP_DIM_KEYS_7D = {'gms_dis', 'gms_ref', 'fupic_dis', 'fupic_ref', 'resize_dis', 'resize_ref'}


def _logistic5(x, b1, b2, b3, b4, b5):
    part = 0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))
    return b1 * part + b4 * x + b5


def _rescale_logistic5(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return _rescale_zscore(pred, target)

    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 5:
        return _rescale_zscore(x, y)

    try:
        p0 = [float(np.max(y)), 1.0, float(np.mean(x)), 0.0, 0.0]
        popt, _ = curve_fit(_logistic5, x, y, p0=p0, maxfev=10000)
        fit = _logistic5(x, *popt)
        return np.nan_to_num(fit, nan=float(np.mean(y)))
    except Exception:
        return _rescale_zscore(x, y)


def _rescale_zscore(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rescale predictions to match target distribution (z-score rescale)."""
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    std_pr = np.std(pred)
    std_tg = np.std(target)
    if std_pr < 1e-8 or std_tg < 1e-8:
        # Degenerate case: constant predictions or targets — just center on target mean
        return np.full_like(pred, np.mean(target))
    rescaled = ((pred - np.mean(pred)) / std_pr) * std_tg + np.mean(target)
    return np.nan_to_num(rescaled, nan=np.mean(target))


def _rescale(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Rescale predictions before metric computation.

    Modes (via env HMF_EVAL_RESCALE):
      - zscore (default)
      - logistic5
      - none  (no rescaling — return raw predictions as-is)
    """
    mode = str(os.getenv('HMF_EVAL_RESCALE', 'zscore')).strip().lower()
    if mode == 'logistic5':
        return _rescale_logistic5(pred, target)
    if mode == 'none':
        return np.array(pred, dtype=np.float64)
    return _rescale_zscore(pred, target)


def _get_rescale_mode() -> str:
    """Return current rescale mode string."""
    return str(os.getenv('HMF_EVAL_RESCALE', 'zscore')).strip().lower()


def _infer_num_clips(
    dis_yuv: Optional[torch.Tensor],
    spatial_info: Dict[str, torch.Tensor],
    batch: Optional[Dict[str, Any]] = None,
) -> int:
    """Infer clip count K from batched tensors. Returns 1 when no clip dimension exists."""
    if batch is not None:
        nc = batch.get('num_clips', None)
        if isinstance(nc, torch.Tensor) and nc.numel() > 0:
            try:
                k = int(nc.view(-1)[0].item())
                if k > 0:
                    return k
            except Exception:
                pass
    if dis_yuv is not None and isinstance(dis_yuv, torch.Tensor) and dis_yuv.dim() == 6:
        return int(dis_yuv.shape[1])
    for k, t in spatial_info.items():
        if not isinstance(t, torch.Tensor):
            continue
        if k in _CLIP_DIM_KEYS_7D and t.dim() == 7:
            return int(t.shape[1])
        if k in _CLIP_DIM_KEYS_6D and t.dim() == 6:
            return int(t.shape[1])
    return 1


def _select_clip_tensor(name: str, tensor: torch.Tensor, clip_idx: int) -> torch.Tensor:
    """Slice the clip dimension when present for known multiclip tensor layouts."""
    if name in _CLIP_DIM_KEYS_7D and tensor.dim() == 7:
        return tensor[:, clip_idx]
    if name in _CLIP_DIM_KEYS_6D and tensor.dim() == 6:
        return tensor[:, clip_idx]
    return tensor


def _filter_valid_batch(batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Drop invalid/dummy samples from an eval batch."""
    v = batch.get('data_valid', None)
    if v is None or not isinstance(v, torch.Tensor):
        return batch
    mask = v.detach().bool().view(-1)
    if mask.numel() == 0 or not torch.any(mask):
        return None
    if torch.all(mask):
        return batch

    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    keep = idx.tolist()
    n = int(mask.shape[0])
    filtered = {}
    for k, val in batch.items():
        if isinstance(val, torch.Tensor) and val.shape[:1] == (n,):
            filtered[k] = val.index_select(0, idx)
        elif isinstance(val, list) and len(val) == n:
            filtered[k] = [val[i] for i in keep]
        else:
            filtered[k] = val
    return filtered


def save_inference_csv(
    results: List[Dict],
    output_dir: str,
    filename: str = 'inference_results.csv',
    metrics: Optional[Dict] = None,
    phase_metrics: Optional[Dict] = None,
):
    """Save per-video inference results as CSV, with optional metrics summary at bottom."""
    if not is_main_process():
        return
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fieldnames = ['video_id', 'dataset_name', 'mos', 'pred_score', 'pred_raw', 'pred_rescaled',
                  'height', 'width']
    if results and 'orig_height' in results[0]:
        fieldnames.extend(['orig_height', 'orig_width'])
    if results and 'stage' in results[0]:
        fieldnames.insert(3, 'stage')
    # Add timing columns if present
    has_timing = results and 'data_load_time' in results[0]
    if has_timing:
        fieldnames.extend(['num_frames', 'data_load_time', 'model_infer_time', 'total_time'])
        if 'io_decode_time' in results[0]:
            fieldnames.extend(['io_decode_time', 'preprocess_time', 'sample_total_time'])
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Write metrics summary at the TOP for easy visibility
        if metrics:
            srcc = metrics.get('SRCC', 0.0)
            plcc = metrics.get('PLCC', 0.0)
            krcc = metrics.get('KRCC', 0.0)
            rmse = metrics.get('RMSE', 0.0)
            f.write(f'# Overall: SRCC={srcc:.4f}  PLCC={plcc:.4f}  KRCC={krcc:.4f}  RMSE={rmse:.4f}\n')
        if phase_metrics:
            for pname, pm in phase_metrics.items():
                if pname == 'all':
                    continue
                s = pm.get('SRCC', 0.0)
                p = pm.get('PLCC', 0.0)
                k = pm.get('KRCC', 0.0)
                r = pm.get('RMSE', 0.0)
                f.write(f'# {pname}: SRCC={s:.4f}  PLCC={p:.4f}  KRCC={k:.4f}  RMSE={r:.4f}\n')

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    logger.info(f"Inference CSV saved: {filepath} ({len(results)} rows)")


def save_clip_scores_csv(
    per_video: List[Dict],
    output_dir: str,
    filename: str,
    num_clips: int,
    weight_name: str = '',
    ckpt_path: str = '',
):
    """Save per-video clip-level scores as CSV for offline fusion.

    Each row represents one video with columns:
      video_id, dataset_name, mos, stage, height, width, clip_0_score, ..., clip_{K-1}_score

    The file header contains metadata comments for traceability.

    Args:
        per_video: List of per-video result dicts (must contain 'clip_scores' field).
        output_dir: Directory to save the CSV file.
        filename: CSV filename (e.g. 'clip_scores_L3_K8.csv').
        num_clips: Number of clips (K) — used for column headers.
        weight_name: Human-readable weight name for metadata.
        ckpt_path: Checkpoint path for metadata.
    """
    if not is_main_process():
        return
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Build clip column names
    clip_cols = [f'clip_{i}_score' for i in range(num_clips)]
    fieldnames = ['video_id', 'dataset_name', 'mos', 'stage', 'height', 'width'] + clip_cols

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Meta-information comment rows
        from datetime import datetime as _dt
        f.write(f'# weight_name: {weight_name}\n')
        f.write(f'# num_clips: {num_clips}\n')
        f.write(f'# ckpt_path: {ckpt_path}\n')
        f.write(f'# timestamp: {_dt.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# num_videos: {len(per_video)}\n')

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for r in per_video:
            row = {
                'video_id': r.get('video_id', ''),
                'dataset_name': r.get('dataset_name', ''),
                'mos': r.get('mos', ''),
                'stage': r.get('stage', ''),
                'height': r.get('height', 0),
                'width': r.get('width', 0),
            }
            clip_scores = r.get('clip_scores', [])
            for i in range(num_clips):
                if i < len(clip_scores) and clip_scores[i] is not None:
                    row[f'clip_{i}_score'] = float(clip_scores[i])
                else:
                    row[f'clip_{i}_score'] = 'NaN'
            writer.writerow(row)

    logger.info(f"Clip scores CSV saved: {filepath} ({len(per_video)} videos, {num_clips} clips)")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
    dry_run: bool = False,
    dataset_name: str = '',
    use_only_vif_branch: bool = False,
    tta_hflip: bool = False,
    enable_timing: bool = False,
    tta_pooling: str = 'mean',
    tta_pool_kwargs: Optional[Dict] = None,
    return_clip_scores: bool = False,
    timing_warmup_batches: int = 0,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Standard evaluation: single pass over dataloader.

    Args:
        enable_timing: If True, collect per-video timing profiling data.
            Adds 'timing_stats' key to the returned metrics dict and
            'data_load_time', 'model_infer_time', 'total_time' to per_video rows.
        tta_pooling: Name of the TTA pooling strategy for multi-clip aggregation.
            Default 'mean'. See list_tta_pooling_strategies() for options.
        tta_pool_kwargs: Extra keyword arguments for the pooling function.
        return_clip_scores: If True, each per_video row will include a
            'clip_scores' field containing a list of per-clip raw scores.
            This is used by the offline fusion pipeline to cache clip-level
            predictions and try different pooling strategies without re-inference.
        timing_warmup_batches: Number of initial batches whose timing records
            are EXCLUDED from aggregation (CUDA JIT, NAS prefetch, cold cache
            cause the first few batches to be artificially slow).  Default 0.
            Note: inference is still performed on these batches so the output
            predictions remain complete — only their timing numbers are dropped.

    Returns:
        (metrics_dict, per_video_results_list)
        When enable_timing=True, metrics_dict also contains 'timing_stats'.
        When return_clip_scores=True, each per_video dict has 'clip_scores': List[float].
    """
    model.eval()

    # Resolve TTA pooling function
    _pool_kwargs = dict(tta_pool_kwargs or {})
    _pool_fn = get_tta_pooling_fn(tta_pooling)
    if tta_pooling != 'mean' and is_main_process():
        logger.info(f'TTA pooling: {tta_pooling} (kwargs={_pool_kwargs})')

    all_preds = []
    all_targets = []
    all_video_ids = []
    all_dataset_names = []
    all_stages = []
    all_heights = []
    all_widths = []
    all_orig_heights = []
    all_orig_widths = []

    # Clip-level scores accumulator (only when return_clip_scores=True)
    all_clip_scores = []  # List[List[float]], per-sample clip scores

    # Timing accumulators (batch-level, later mapped to per-video)
    all_data_load_times = []   # per-sample data load time (s) -- whole DataLoader next() (IO + preprocess + collate)
    all_model_infer_times = [] # per-sample model inference time (s) -- H2D transfer + forward (whole model block)
    all_pure_forward_times = [] # per-sample pure model forward time (s) -- only model(...) call, no H2D / slice / empty_cache
    all_total_times = []       # per-sample total time (s)
    all_num_frames = []        # per-sample num_frames (actual T sampled into the model)

    # Fine-grained per-sample timing from dataset __getitem__ (new).
    #   all_io_decode_times     : disk IO + YUV decode + sampled-clip cache load
    #   all_preprocess_times    : Resize + GMS patch sampling + fupic + semantic branch
    #   all_sample_total_times  : io + preprocess (should be <= data_load_time)
    #   all_gpu_preprocess_times: GPU-side resize_dis rebuild (opt-in; 0.0 when off)
    all_io_decode_times = []
    all_preprocess_times = []
    all_sample_total_times = []
    all_gpu_preprocess_times = []

    # GPU-preprocess application accounting (single-clip path only). Lets us
    # verify at the end whether the GPU rebuild actually ran on every batch
    # (applied) or silently fell back to the CPU placeholder (fallback).
    _gpu_prep_applied_batches = 0
    _gpu_prep_fallback_batches = 0

    has_cuda = torch.cuda.is_available() and device.type == 'cuda'

    # Detect whether the model has a cross-clip temporal interaction module (detect ahead of time to avoid repeated per-batch checks)
    _has_cross_clip_temporal = False
    _core = model.module if hasattr(model, 'module') else model
    if hasattr(_core, 'semantic_branch') and _core.semantic_branch is not None:
        _has_cross_clip_temporal = (
            getattr(_core.semantic_branch, 'cross_clip_temporal_module', None) is not None
        )

    num_batches = len(dataloader)

    _use_tqdm = is_main_process() and tqdm is not None
    # Use explicit iterator for timing
    raw_iter = iter(dataloader)
    if _use_tqdm:
        pbar = tqdm(
            total=num_batches, desc=f'Eval [{dataset_name}]', unit='batch',
            dynamic_ncols=True, leave=False,
        )
    else:
        pbar = None

    eval_wall_start = time.perf_counter()

    for batch_idx in range(num_batches):
        if dry_run and batch_idx >= 1:
            break

        # --- Phase 1: Data loading (DataLoader __next__) ---
        if enable_timing:
            if has_cuda:
                torch.cuda.synchronize(device)
            t_data_start = time.perf_counter()

        try:
            batch = next(raw_iter)
        except StopIteration:
            break

        if enable_timing:
            t_data_end = time.perf_counter()
            batch_data_time = t_data_end - t_data_start
            # Warn on slow data loading (likely NAS IO or 4K decode bottleneck)
            if batch_data_time > 30.0:
                vids = batch.get('video_id', []) if isinstance(batch, dict) else []
                if isinstance(vids, torch.Tensor):
                    vids = vids.tolist()
                vid_str = ', '.join(str(v) for v in vids[:4])
                if len(vids) > 4:
                    vid_str += f', ... ({len(vids)} total)'
                logger.warning(
                    "[SLOW DATA] Batch %d/%d took %.1fs to load. Videos: [%s]",
                    batch_idx + 1, num_batches, batch_data_time, vid_str,
                )

        batch = _filter_valid_batch(batch)
        if batch is None:
            if pbar is not None:
                pbar.update(1)
            continue

        # --- Phase 2: Device transfer + model forward ---
        # Default GPU preprocess time (overwritten by the single-clip path
        # when raw YUV planes are present in the batch).
        batch_gpu_prep_time = 0.0

        # Determine full-res YUV requirement
        ref_yuv = batch.get('ref_yuv')
        has_vif_precomputed = ('vif_precomputed_frame_features' in batch and
                               batch['vif_precomputed_frame_features'] is not None)
        has_dense_vif_in_batch = ('vif_dis_y_full' in batch and
                                  batch.get('vif_dis_y_full') is not None and
                                  'vif_ref_y_full' in batch and
                                  batch.get('vif_ref_y_full') is not None)
        core_model = model.module if hasattr(model, 'module') else model
        model_has_vif = (hasattr(core_model, 'vif_branch') and
                         core_model.vif_branch is not None)
        need_fullres = (ref_yuv is not None and
                        model_has_vif and
                        not has_vif_precomputed and
                        not has_dense_vif_in_batch)
        if not (need_fullres and 'dis_yuv' in batch):
            dis_yuv = None
            ref_yuv = None
        else:
            dis_yuv = batch['dis_yuv']   # keep on CPU for now
            # ref_yuv already set from batch

        target = batch['mos'].to(device, non_blocking=True)

        # Infer num_clips from **CPU** tensors (before any .to(device))
        _spatial_keys = ['gms_dis', 'gms_ref', 'resize_dis', 'resize_ref',
                         'fupic_dis', 'fupic_ref',
                         'vif_dis_y_full', 'vif_ref_y_full', 'vif_frame_mask',
                         'vif_precomputed_frame_features', 'vif_precomputed_mask',
                         'vif_precomputed_ti_video_feat', 'vif_precomputed_vmaf_frame_score']
        _spatial_cpu = {}
        for k in _spatial_keys:
            if k in batch and batch[k] is not None:
                _spatial_cpu[k] = batch[k]  # keep on CPU

        meta = {}
        for mk in ['height', 'width']:
            if mk in batch and isinstance(batch[mk], torch.Tensor):
                meta[mk] = batch[mk].to(device, non_blocking=True)

        num_clips = _infer_num_clips(dis_yuv, _spatial_cpu, batch=batch)

        # ── H2D transfer phase (excluded from method timing) ──
        # Move all tensors to GPU BEFORE starting the method timer.
        # H2D transfer is data movement (like IO), not part of our method.
        if num_clips > 1 and _has_cross_clip_temporal:
            if dis_yuv is not None:
                dis_yuv = dis_yuv.to(device, non_blocking=True)
            if ref_yuv is not None:
                ref_yuv = ref_yuv.to(device, non_blocking=True)
            spatial_info = {}
            for k, v in _spatial_cpu.items():
                spatial_info[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        elif num_clips <= 1:
            # Single-clip path: H2D before timing
            if dis_yuv is not None:
                dis_yuv = dis_yuv.to(device, non_blocking=True)
            if ref_yuv is not None:
                ref_yuv = ref_yuv.to(device, non_blocking=True)
            spatial_info = {}
            for k, v in _spatial_cpu.items():
                spatial_info[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            # Also move raw YUV planes to GPU here (before timing starts),
            # so that _apply_gpu_preprocess_legacy_pe doesn't need to do
            # H2D + sync inside the timed region.
            for _raw_key in ('raw_y_dis', 'raw_u_half_dis', 'raw_v_half_dis',
                             'raw_y_ref', 'raw_u_half_ref', 'raw_v_half_ref'):
                _rv = batch.get(_raw_key)
                if isinstance(_rv, torch.Tensor) and _rv.device != device:
                    batch[_raw_key] = _rv.to(device, non_blocking=True)

        # ── Method timer starts HERE (after H2D, pure GPU compute only) ──
        # Two timers run in parallel:
        #   • batch_model_time  — wraps the ENTIRE model block: per-clip H2D,
        #                          CPU tensor slicing, empty_cache and forward.
        #                          Kept for continuity with the reference pipeline.
        #   • batch_forward_time — only the model(...) call itself. Isolates
        #                          the pure GPU forward pass from surrounding
        #                          data-movement overhead.
        if enable_timing:
            if has_cuda:
                torch.cuda.synchronize(device)
            t_model_start = time.perf_counter()
        batch_forward_time = 0.0

        def _timed_forward_run(_call):
            """Run ``_call()`` with tight CUDA sync fences and return its output.
            Adds the wall time of the pure model call (only what happens inside
            ``_call``) to ``batch_forward_time``.
            """
            nonlocal batch_forward_time
            if enable_timing and has_cuda:
                torch.cuda.synchronize(device)
            _t0 = time.perf_counter()
            _res = _call()
            if enable_timing and has_cuda:
                torch.cuda.synchronize(device)
            batch_forward_time += time.perf_counter() - _t0
            return _res

        if num_clips > 1 and _has_cross_clip_temporal:
            # ---------------------------------------------------------------
            # CROSS-CLIP TEMPORAL inference:
            # ---------------------------------------------------------------
            # GPU preprocess (mode=2) is NOT implemented for this path; a zero
            # placeholder would never be rebuilt. Fail-fast instead of scoring 0.
            _pph_cct = batch.get('gpu_prep_placeholder', None)
            _cct_needs = (
                (isinstance(_pph_cct, torch.Tensor) and bool(_pph_cct.numel())
                 and bool(_pph_cct.reshape(-1)[0].item() != 0))
                or (not isinstance(_pph_cct, torch.Tensor) and bool(_pph_cct))
            )
            if _cct_needs:
                raise RuntimeError(
                    "[gpu_preprocess] GPU preprocess (mode=2) is not supported on "
                    "the cross-clip temporal path; a zero placeholder was emitted "
                    "but cannot be rebuilt. Use HMF_VQA_GPU_PREPROCESS=0/1 for this "
                    "model, or disable the cross-clip temporal module."
                )

            def _do_cct():
                with autocast(enabled=use_amp):
                    _outputs = model(
                        dis_yuv, ref_yuv, spatial_info, meta,
                        use_only_vif_branch=use_only_vif_branch,
                    )
                return _outputs
            outputs = _timed_forward_run(_do_cct)
            pred = outputs['score']

            if return_clip_scores:
                # After temporal fusion there is only one score, wrap as a single-element list
                _pred_cpu = pred.detach().cpu()
                for b_i in range(_pred_cpu.shape[0]):
                    all_clip_scores.append([_pred_cpu[b_i].item()])

        elif num_clips > 1:
            # ---------------------------------------------------------------
            # MEMORY-EFFICIENT multi-clip inference:
            # Keep all data on CPU; move only one clip at a time to GPU.
            # This avoids CUDA OOM for high clip counts (8/16) on 24GB GPUs.
            # ---------------------------------------------------------------
            # Does the batch carry a zero placeholder that REQUIRES a GPU rebuild?
            _pph = batch.get('gpu_prep_placeholder', None)
            _mc_needs_rebuild = False
            if isinstance(_pph, torch.Tensor):
                _mc_needs_rebuild = bool(_pph.numel()) and bool(_pph.reshape(-1)[0].item() != 0)
            elif _pph:
                _mc_needs_rebuild = True
            _mc_gpu_applied = False

            clip_preds = []
            for clip_idx in range(num_clips):
                # Slice clip from CPU tensors, then move to device
                dis_clip = dis_yuv[:, clip_idx].to(device, non_blocking=True) if (dis_yuv is not None and dis_yuv.dim() == 6) else (dis_yuv.to(device, non_blocking=True) if dis_yuv is not None else None)
                ref_clip = ref_yuv[:, clip_idx].to(device, non_blocking=True) if (ref_yuv is not None and ref_yuv.dim() == 6) else (ref_yuv.to(device, non_blocking=True) if ref_yuv is not None else None)
                spatial_clip = {}
                for k, v in _spatial_cpu.items():
                    sliced = _select_clip_tensor(k, v, clip_idx)
                    spatial_clip[k] = sliced.to(device, non_blocking=True) if isinstance(sliced, torch.Tensor) else sliced

                # ── GPU preprocess for THIS clip (opt-in; rebuilds resize_dis) ──
                # The per-clip raw-plane H2D inside the helper is excluded from
                # the CUDA-event window, so only the GPU rebuild compute is timed.
                _clip_gpu_prep = _apply_gpu_preprocess_legacy_pe_clip(
                    batch, spatial_clip, clip_idx, device, has_cuda=has_cuda,
                )
                if _clip_gpu_prep is not None:
                    batch_gpu_prep_time += _clip_gpu_prep
                    _mc_gpu_applied = True
                elif _mc_needs_rebuild:
                    # FAIL-FAST: placeholder emitted but GPU rebuild did NOT run →
                    # resize_dis is still all-zeros → silently WRONG results.
                    raise RuntimeError(
                        "[gpu_preprocess] multi-clip (K={}) zero placeholder was "
                        "emitted (HMF_VQA_GPU_PREPROCESS=2) but the GPU rebuild did "
                        "NOT run on clip {} — resize_dis is still all-zeros. Aborting "
                        "instead of scoring zeros. Check raw YUV planes / layout.".format(
                            num_clips, clip_idx)
                    )

                def _do_clip():
                    with autocast(enabled=use_amp):
                        _outputs = model(
                            dis_clip, ref_clip, spatial_clip, meta,
                            use_only_vif_branch=use_only_vif_branch,
                        )
                    return _outputs
                outputs = _timed_forward_run(_do_clip)
                clip_preds.append(outputs['score'].detach())

                # Free GPU memory for this clip immediately
                del dis_clip, ref_clip, spatial_clip, outputs
                if has_cuda:
                    torch.cuda.empty_cache()

            if _mc_gpu_applied:
                _gpu_prep_applied_batches += 1

            pred_stack = torch.stack(clip_preds, dim=1)  # [B, K]
            # Use TTA pooling strategy (default: mean)
            pred = _pool_fn(pred_stack, **_pool_kwargs)
            # Save clip-level raw scores (for offline fusion)
            if return_clip_scores:
                _cs_cpu = pred_stack.cpu()  # [B, K]
                for b_i in range(_cs_cpu.shape[0]):
                    all_clip_scores.append(_cs_cpu[b_i].tolist())
            del clip_preds, pred_stack
        else:
            # Single-clip path: H2D already done above; just GPU preprocess + forward.

            # ── GPU preprocess (optional, opt-in via dataset env var) ──
            # Replaces resize_dis produced on CPU by the worker with one
            # rebuilt on GPU from raw Y/U_half/V_half planes.  Wall-clock
            # time of this rebuild is measured with CUDA events.
            _gpu_prep_time = _apply_gpu_preprocess_legacy_pe(
                batch, spatial_info, device, has_cuda=has_cuda,
            )
            # Did the worker emit a zero placeholder that REQUIRES a GPU rebuild?
            _pph = batch.get('gpu_prep_placeholder', None)
            _needs_rebuild = False
            if isinstance(_pph, torch.Tensor):
                _needs_rebuild = bool(_pph.numel()) and bool(_pph.reshape(-1)[0].item() != 0)
            elif _pph:
                _needs_rebuild = True

            if _gpu_prep_time is not None:
                batch_gpu_prep_time = _gpu_prep_time
                _gpu_prep_applied_batches += 1
            else:
                batch_gpu_prep_time = 0.0
                # ── FAIL-FAST: placeholder emitted but GPU rebuild did NOT run ──
                # resize_dis is still all-zeros → the model would be fed zeros and
                # produce silently WRONG results. Abort loudly instead.
                if _needs_rebuild:
                    raise RuntimeError(
                        "[gpu_preprocess] HMF_VQA_GPU_PREPROCESS=2 emitted a zero "
                        "placeholder for resize_dis, but the GPU rebuild did NOT run "
                        "(resize_dis is still all-zeros). This would silently produce "
                        "WRONG results. Likely causes: raw YUV planes missing from the "
                        "batch (non-.yuv input / per-clip capture failed / pack "
                        "exception), unexpected resize_dis layout, or the gpu_preprocess "
                        "module failed to import. Aborting instead of scoring zeros."
                    )
                _gpu_prep_fallback_batches += 1

            def _do_single():
                with autocast(enabled=use_amp):
                    _outputs = model(
                        dis_yuv, ref_yuv, spatial_info, meta,
                        use_only_vif_branch=use_only_vif_branch,
                    )
                return _outputs
            outputs = _timed_forward_run(_do_single)
            pred = outputs['score']

            # TTA: horizontal flip inference
            if tta_hflip:
                spatial_flip = {}
                for k, v in spatial_info.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 3:
                        spatial_flip[k] = v.flip(-1)
                    else:
                        spatial_flip[k] = v
                def _do_flip():
                    with autocast(enabled=use_amp):
                        _outputs = model(
                            dis_yuv.flip(-1) if dis_yuv is not None else None,
                            ref_yuv.flip(-1) if ref_yuv is not None else None,
                            spatial_flip, meta,
                            use_only_vif_branch=use_only_vif_branch,
                        )
                    return _outputs
                outputs_flip = _timed_forward_run(_do_flip)
                pred = (pred + outputs_flip['score']) * 0.5

        # Single-clip path: save clip-level score (single-element list)
        if return_clip_scores and num_clips <= 1:
            _pred_cpu = pred.detach().cpu()
            for b_i in range(_pred_cpu.shape[0]):
                all_clip_scores.append([float(_pred_cpu[b_i])])

        if enable_timing:
            if has_cuda:
                torch.cuda.synchronize(device)
            t_model_end = time.perf_counter()
            batch_model_time = t_model_end - t_model_start
            batch_total_time = t_model_end - t_data_start

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

        batch_size = pred.shape[0]

        # Collect IDs
        vids = batch.get('video_id', [''] * batch_size)
        if isinstance(vids, torch.Tensor):
            vids = vids.tolist()
        all_video_ids.extend(vids)

        dsnames = batch.get('dataset_name', [dataset_name] * batch_size)
        if isinstance(dsnames, torch.Tensor):
            dsnames = dsnames.tolist()
        all_dataset_names.extend(dsnames)

        stages = batch.get('stage', [None] * batch_size)
        if isinstance(stages, torch.Tensor):
            stages = stages.tolist()
        all_stages.extend(stages)

        # Collect resolution info
        heights = batch.get('height', [0] * batch_size)
        if isinstance(heights, torch.Tensor):
            heights = heights.tolist()
        all_heights.extend(heights)
        widths = batch.get('width', [0] * batch_size)
        if isinstance(widths, torch.Tensor):
            widths = widths.tolist()
        all_widths.extend(widths)

        # Original (annotation-declared) resolution for by-resolution timing
        # stats.  Falls back to (heights/widths) when not present (older datasets).
        orig_heights = batch.get('orig_height', None)
        if orig_heights is None:
            orig_heights = heights
        elif isinstance(orig_heights, torch.Tensor):
            orig_heights = orig_heights.tolist()
        all_orig_heights.extend(orig_heights)
        orig_widths = batch.get('orig_width', None)
        if orig_widths is None:
            orig_widths = widths
        elif isinstance(orig_widths, torch.Tensor):
            orig_widths = orig_widths.tolist()
        all_orig_widths.extend(orig_widths)

        # Collect timing (distribute batch time evenly to per-sample)
        if enable_timing:
            per_sample_data = batch_data_time / max(batch_size, 1)
            per_sample_model = batch_model_time / max(batch_size, 1)
            per_sample_forward = batch_forward_time / max(batch_size, 1)
            per_sample_total = batch_total_time / max(batch_size, 1)

            # Fine-grained per-sample timing from dataset __getitem__
            io_list = batch.get('io_decode_time', None)
            pre_list = batch.get('preprocess_time', None)
            samp_total_list = batch.get('sample_total_time', None)
            if isinstance(io_list, torch.Tensor):
                io_list = io_list.detach().cpu().tolist()
            if isinstance(pre_list, torch.Tensor):
                pre_list = pre_list.detach().cpu().tolist()
            if isinstance(samp_total_list, torch.Tensor):
                samp_total_list = samp_total_list.detach().cpu().tolist()
            # Infer num_frames from batch tensors.
            # resize_dis shapes after collate:
            #   avg/random1 mode: [B, 3, T, H, W]  (5D) → T at dim 2
            #   stack mode:       [B, P, 3, T, H, W] (6D) → T at dim 3
            # gms_dis shapes:     [B, P, 3, T, ph, pw] (6D) → T at dim 3
            # dis_yuv shapes:     [B, 3, T, H, W] (5D) → T at dim 2
            nf_val = 0
            for _tkey in ('resize_dis', 'gms_dis', 'fupic_dis', 'dis_yuv'):
                _t = batch.get(_tkey, None)
                if _t is None or not isinstance(_t, torch.Tensor):
                    continue
                d = _t.dim()
                if d == 6:
                    # [B, P, 3, T, H, W] → T at dim 3
                    nf_val = int(_t.shape[3])
                    break
                elif d == 5:
                    # [B, 3, T, H, W] → T at dim 2
                    nf_val = int(_t.shape[2])
                    break
                elif d == 4:
                    # [B, 3, T*H, W] or similar — less reliable, skip
                    continue
            nf_batch = [nf_val] * batch_size

            # Prefer per-sample num_frames reported by dataset when available
            ds_nf = batch.get('num_frames_sampled', None)
            if isinstance(ds_nf, torch.Tensor):
                ds_nf = ds_nf.detach().cpu().tolist()
            if isinstance(ds_nf, (list, tuple)) and len(ds_nf) == batch_size:
                try:
                    ds_nf = [int(v) for v in ds_nf]
                    if any(v > 0 for v in ds_nf):
                        nf_batch = ds_nf
                except Exception:
                    pass

            # Warmup: the first `timing_warmup_batches` batches' timings are
            # dropped from aggregation (CUDA JIT, NAS prefetch, cold Python
            # cache make them artificially slow).  We still append sentinel
            # entries (NaN) so that per-video indices stay aligned; the
            # `_compute_timing_stats` call filters them out.
            _is_warmup = (
                enable_timing and timing_warmup_batches > 0
                and batch_idx < timing_warmup_batches
            )
            if _is_warmup and is_main_process() and batch_idx == 0:
                logger.info(
                    f'[Timing] Skipping first {timing_warmup_batches} batch(es) '
                    f'for warmup (CUDA JIT / NAS prefetch / cold cache).'
                )

            for b_i in range(batch_size):
                if _is_warmup:
                    # Sentinel NaNs keep per-video index alignment but are
                    # filtered out by _compute_timing_stats().
                    all_data_load_times.append(float('nan'))
                    all_model_infer_times.append(float('nan'))
                    all_pure_forward_times.append(float('nan'))
                    all_total_times.append(float('nan'))
                    all_io_decode_times.append(float('nan'))
                    all_preprocess_times.append(float('nan'))
                    all_sample_total_times.append(float('nan'))
                    all_gpu_preprocess_times.append(float('nan'))
                    continue
                all_data_load_times.append(per_sample_data)
                all_model_infer_times.append(per_sample_model)
                all_pure_forward_times.append(per_sample_forward)
                all_total_times.append(per_sample_total)
                # Fine-grained timings (0 when dataset didn't inject them)
                io_v = 0.0
                pre_v = 0.0
                samp_total_v = 0.0
                try:
                    if io_list is not None and b_i < len(io_list):
                        io_v = float(io_list[b_i])
                    if pre_list is not None and b_i < len(pre_list):
                        pre_v = float(pre_list[b_i])
                    if samp_total_list is not None and b_i < len(samp_total_list):
                        samp_total_v = float(samp_total_list[b_i])
                except Exception:
                    pass
                all_io_decode_times.append(io_v)
                all_preprocess_times.append(pre_v)
                all_sample_total_times.append(samp_total_v)
                # GPU preprocess time: batch-level wall-clock divided by
                # batch_size so per-sample amortized cost lands in the
                # same units as all the other per-sample fields.
                all_gpu_preprocess_times.append(
                    float(batch_gpu_prep_time) / max(batch_size, 1)
                )
            all_num_frames.extend(nf_batch[:batch_size])

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # ── GPU-preprocess accounting summary ──
    # Confirms the GPU rebuild actually ran. Printed only when it fired at least
    # once (applied>0). If you requested HMF_VQA_GPU_PREPROCESS=2 but see no such
    # line, the run used CPU preprocessing entirely (e.g. multi-clip TTA, which
    # is single-clip-only for GPU rebuild).
    if enable_timing and is_main_process() and _gpu_prep_applied_batches > 0:
        logger.info(
            "[gpu_preprocess] GPU resize_dis rebuild applied on %d batch(es); "
            "CPU-fallback on %d batch(es).",
            _gpu_prep_applied_batches, _gpu_prep_fallback_batches,
        )

    eval_wall_end = time.perf_counter()
    eval_wall_elapsed = eval_wall_end - eval_wall_start

    if not all_preds:
        metrics = {'SRCC': 0.0, 'PLCC': 0.0, 'KRCC': 0.0, 'RMSE': 0.0}
        model.train()
        return metrics, []

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # All-gather across ranks if DDP
    if dist.is_initialized():
        all_preds, all_targets, all_video_ids, all_dataset_names, all_stages, all_heights, all_widths = \
            _gather_all(
                all_preds, all_targets, all_video_ids, all_dataset_names, all_stages,
                all_heights, all_widths,
            )

    # Rescale predictions
    rescale_mode = _get_rescale_mode()
    preds_rescaled = _rescale(all_preds, all_targets)

    # Compute metrics on rescaled predictions
    metrics = compute_all_metrics(preds_rescaled, all_targets)

    # Always compute raw (no-rescale) metrics for comparison
    raw_metrics = compute_all_metrics(all_preds, all_targets)
    metrics['raw_SRCC'] = raw_metrics['SRCC']
    metrics['raw_PLCC'] = raw_metrics['PLCC']
    metrics['raw_KRCC'] = raw_metrics['KRCC']
    metrics['raw_RMSE'] = raw_metrics['RMSE']

    # Build per-video results
    per_video = []
    timing_records = []
    for i in range(len(all_preds)):
        row = {
            'video_id': str(all_video_ids[i]) if i < len(all_video_ids) else '',
            'dataset_name': str(all_dataset_names[i]) if i < len(all_dataset_names) else dataset_name,
            'mos': float(all_targets[i]),
            'pred_score': float(preds_rescaled[i]),
            'pred_raw': float(all_preds[i]),
            'pred_rescaled': float(preds_rescaled[i]),
            'height': int(all_heights[i]) if i < len(all_heights) else 0,
            'width': int(all_widths[i]) if i < len(all_widths) else 0,
            'orig_height': int(all_orig_heights[i]) if i < len(all_orig_heights) else 0,
            'orig_width': int(all_orig_widths[i]) if i < len(all_orig_widths) else 0,
        }
        if i < len(all_stages) and all_stages[i] is not None:
            row['stage'] = all_stages[i]
        # Add clip-level scores (for offline fusion)
        if return_clip_scores and i < len(all_clip_scores):
            row['clip_scores'] = all_clip_scores[i]
        # Add timing fields
        if enable_timing and i < len(all_data_load_times):
            _dl = all_data_load_times[i]
            # Skip warmup sentinel (NaN) — still emit the per-video prediction
            # but don't contribute to timing aggregation.
            import math as _math
            if not (isinstance(_dl, float) and _math.isnan(_dl)):
                row['data_load_time'] = round(_dl, 6)
                row['model_infer_time'] = round(all_model_infer_times[i], 6)
                if i < len(all_pure_forward_times):
                    _pft = all_pure_forward_times[i]
                    if isinstance(_pft, (int, float)) and _pft == _pft:  # not NaN
                        row['pure_forward_time'] = round(float(_pft), 6)
                row['total_time'] = round(all_total_times[i], 6)
                row['num_frames'] = int(all_num_frames[i]) if i < len(all_num_frames) else 0
                # Fine-grained (dataset-side) timings
                if i < len(all_io_decode_times):
                    row['io_decode_time'] = round(all_io_decode_times[i], 6)
                    row['preprocess_time'] = round(all_preprocess_times[i], 6)
                    row['sample_total_time'] = round(all_sample_total_times[i], 6)
                # GPU preprocess time (resize_dis rebuild on device).
                if i < len(all_gpu_preprocess_times):
                    _gpt = all_gpu_preprocess_times[i]
                    import math as _math_g
                    if not (isinstance(_gpt, float) and _math_g.isnan(_gpt)):
                        row['gpu_preprocess_time'] = round(float(_gpt), 6)
                timing_records.append(row)
        per_video.append(row)

    # Compute and attach timing stats
    if enable_timing and timing_records and is_main_process():
        timing_stats = _compute_timing_stats(timing_records, dataset_name)
        timing_stats['eval_wall_clock_sec'] = round(eval_wall_elapsed, 3)
        metrics['timing_stats'] = timing_stats
        print_timing_report(timing_stats)

    if is_main_process():
        logger.info(f'Eval [{dataset_name}]: '
                     f'SRCC={metrics["SRCC"]:.4f}  PLCC={metrics["PLCC"]:.4f}  '
                     f'KRCC={metrics["KRCC"]:.4f}  RMSE={metrics["RMSE"]:.4f}')

    model.train()
    return metrics, per_video


@torch.no_grad()
def evaluate_cvqm_by_phase(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
    dry_run: bool = False,
    use_only_vif_branch: bool = False,
    tta_hflip: bool = False,
    enable_timing: bool = False,
    tta_pooling: str = 'mean',
    tta_pool_kwargs: Optional[Dict] = None,
    return_clip_scores: bool = False,
    timing_warmup_batches: int = 0,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict]]:
    """
    CVQM phase-based evaluation: evaluates the full CVQM test set,
    then splits results by stage (Phase1 / Phase2) for separate metrics.

    Returns:
        (results_dict, per_video_results)
        results_dict keys: 'all', 'phase1', 'phase2' each with SRCC/PLCC/KRCC/RMSE
    """
    # Run full evaluation first
    metrics_all, per_video = evaluate(
        model,
        dataloader,
        device,
        use_amp,
        dry_run,
        dataset_name='CVQM',
        use_only_vif_branch=use_only_vif_branch,
        tta_hflip=tta_hflip,
        enable_timing=enable_timing,
        tta_pooling=tta_pooling,
        tta_pool_kwargs=tta_pool_kwargs,
        return_clip_scores=return_clip_scores,
        timing_warmup_batches=timing_warmup_batches,
    )

    # ── Per-phase 5PL fitting (Phase1 5PL alone + Phase2 5PL alone) ──
    # Consistent with collect_sd3_results.py / generate_clean.py:
    #   Each phase independently performs 5PL fitting internally; the total stage computes metrics on the concatenation of fitted scores.
    #   This adapts to the score-distribution differences across phases better than "global 5PL → slice".
    stages = np.array([r.get('stage', None) for r in per_video])
    targets = np.array([r['mos'] for r in per_video])
    preds_raw = np.array([r['pred_raw'] for r in per_video])

    # Detect rescale mode (logistic5 / zscore / none)
    rescale_mode = _get_rescale_mode()

    # Rescale each phase independently (5PL fitting)
    preds_phase_rescaled = np.copy(preds_raw)  # Initialize to raw; overwritten per phase later
    results = {}

    for phase_label, phase_name in [(1, 'phase1'), (2, 'phase2')]:
        mask = stages == phase_label
        if mask.sum() > 4:
            # Per-phase independent rescale (5PL fitting)
            phase_rescaled = _rescale(preds_raw[mask], targets[mask])
            preds_phase_rescaled[mask] = phase_rescaled

            phase_metrics = compute_all_metrics(phase_rescaled, targets[mask])
            raw_phase = compute_all_metrics(preds_raw[mask], targets[mask])
            phase_metrics['raw_SRCC'] = raw_phase['SRCC']
            phase_metrics['raw_PLCC'] = raw_phase['PLCC']
            phase_metrics['raw_KRCC'] = raw_phase['KRCC']
            phase_metrics['raw_RMSE'] = raw_phase['RMSE']
            results[phase_name] = phase_metrics
            if is_main_process():
                logger.info(
                    f'  CVQM {phase_name}: SRCC={phase_metrics["SRCC"]:.4f} '
                    f'PLCC={phase_metrics["PLCC"]:.4f} '
                    f'KRCC={phase_metrics["KRCC"]:.4f}'
                )
        else:
            results[phase_name] = {'SRCC': 0.0, 'PLCC': 0.0, 'KRCC': 0.0, 'RMSE': 0.0}

    # Total stage (all): compute metrics on the concatenation of per-phase-fitted scores
    metrics_all_phase = compute_all_metrics(preds_phase_rescaled, targets)
    raw_all = compute_all_metrics(preds_raw, targets)
    metrics_all_phase['raw_SRCC'] = raw_all['SRCC']
    metrics_all_phase['raw_PLCC'] = raw_all['PLCC']
    metrics_all_phase['raw_KRCC'] = raw_all['KRCC']
    metrics_all_phase['raw_RMSE'] = raw_all['RMSE']
    # Preserve timing_stats returned by evaluate() (if present)
    if 'timing_stats' in metrics_all:
        metrics_all_phase['timing_stats'] = metrics_all['timing_stats']
    results['all'] = metrics_all_phase

    if is_main_process():
        logger.info(
            f'  CVQM all: SRCC={metrics_all_phase["SRCC"]:.4f} '
            f'PLCC={metrics_all_phase["PLCC"]:.4f} '
            f'KRCC={metrics_all_phase["KRCC"]:.4f}'
        )

    # Update per_video pred_rescaled / pred_score to per-phase-fitted values
    for i in range(len(per_video)):
        per_video[i]['pred_rescaled'] = float(preds_phase_rescaled[i])
        per_video[i]['pred_score'] = float(preds_phase_rescaled[i])

    model.train()
    return results, per_video


def _gather_all(preds, targets, video_ids, dataset_names, stages, heights, widths):
    """All-gather predictions + metadata across DDP ranks."""
    if not dist.is_initialized():
        return preds, targets, video_ids, dataset_names, stages, heights, widths

    # Gather numeric arrays
    preds, targets = _gather_predictions(preds, targets)

    # Gather string lists via all_gather_object
    world_size = dist.get_world_size()
    all_vids = [None] * world_size
    all_dsnames = [None] * world_size
    all_stages_list = [None] * world_size
    all_heights = [None] * world_size
    all_widths = [None] * world_size
    dist.all_gather_object(all_vids, video_ids)
    dist.all_gather_object(all_dsnames, dataset_names)
    dist.all_gather_object(all_stages_list, stages)
    dist.all_gather_object(all_heights, heights)
    dist.all_gather_object(all_widths, widths)

    gathered_vids = [v for sublist in all_vids for v in sublist]
    gathered_dsnames = [v for sublist in all_dsnames for v in sublist]
    gathered_stages = [v for sublist in all_stages_list for v in sublist]
    gathered_heights = [v for sublist in all_heights for v in sublist]
    gathered_widths = [v for sublist in all_widths for v in sublist]

    return preds, targets, gathered_vids, gathered_dsnames, gathered_stages, gathered_heights, gathered_widths


def _gather_predictions(preds: np.ndarray, targets: np.ndarray):
    """All-gather predictions across DDP ranks."""
    if not dist.is_initialized():
        return preds, targets

    world_size = dist.get_world_size()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    preds_t = torch.from_numpy(preds).to(device)
    targets_t = torch.from_numpy(targets).to(device)

    # Gather sizes first (each rank may have different num samples)
    local_size = torch.tensor([preds_t.shape[0]], device=preds_t.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)

    # Pad to max_size
    if preds_t.shape[0] < max_size:
        pad = max_size - preds_t.shape[0]
        preds_t = torch.cat([preds_t, torch.zeros(pad, device=preds_t.device)])
        targets_t = torch.cat([targets_t, torch.zeros(pad, device=targets_t.device)])

    # Gather
    all_preds = [torch.zeros_like(preds_t) for _ in range(world_size)]
    all_targets = [torch.zeros_like(targets_t) for _ in range(world_size)]
    dist.all_gather(all_preds, preds_t)
    dist.all_gather(all_targets, targets_t)

    # Unpad
    gathered_preds = []
    gathered_targets = []
    for i in range(world_size):
        gathered_preds.append(all_preds[i][:all_sizes[i]].cpu().numpy())
        gathered_targets.append(all_targets[i][:all_sizes[i]].cpu().numpy())

    return np.concatenate(gathered_preds), np.concatenate(gathered_targets)
