"""
DataModule: builds datasets, DataLoaders, and samplers for training and evaluation.
Supports multi-dataset joint training and CVQM two-stage evaluation.
Mock dataset for dry-run mode.

Key design:
  - VQADataset returns per-sample: dis_yuv [3,T,H,W], gms_dis [P,3,T,ph,pw],
    resize_dis [3,T,ts,ts], height, width, mos, video_id, etc.
  - Single clip per sample (random for train, first for test) — no K dimension.
  - Spatial sampling (GMS patches + resize) done in dataset for efficiency.
  - skip_val: skip building val datasets (train==val for many setups).
  - test_eval_mode: 'fast' = training-style sampling, 'full' = full evaluation.
"""

import os
import random
import logging
import hashlib
import re
import sys
import time
import json
import subprocess
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler

from .datasets import (
    SampleMeta, get_parser,
    parse_cvqm, split_cvqm_by_stage,
    get_dataset_config,
    get_dataset_path_runtime_info,
    resolve_sampled_clip_cache_root,
    validate_dataset_layout,
)
from .io.video_reader import VideoReaderFactory, is_container_video
from .sampling.frame_sampler import FrameSampler
from .sampling.spatial_sampler import GMSSampler, ResizeSampler, FuPiCSampler, FragmentSampler, build_spatial_sampler
from .yuv.colorspace import rgb_to_yuv_bt709, rgb_to_yuv_bt601
from .cache.frame_cache import (
    load_cached_clip as _load_cached_clip_raw,
    get_cache_clip_path,
    cache_clip_exists,
    init_frame_cache_buffer,
)

logger = logging.getLogger('hmf_vqa.datamodule')


def _get_shm_size() -> Optional[int]:
    """Return /dev/shm available size in bytes, or None if unavailable."""
    try:
        stat = os.statvfs('/dev/shm')
        return stat.f_bavail * stat.f_frsize
    except (OSError, AttributeError):
        return None


class VQADataset(Dataset):
    """
    Unified VQA dataset.

    Outputs per sample:
      - dis_yuv: [3, T, H, W] float32 YUV in [0, 1]
      - gms_dis: [P, 3, T, ph, pw]  GMS patches for detail branch
      - resize_dis: [3, T, ts, ts]  resized frames for semantic branch
      - ref versions of above (when FR)
      - height, width: resolution scalars for RAPE/ScaleToken
      - mos, video_id, dataset_name, stage

    Single clip per sample: randomly selected during training, first clip during test.
    Spatial sampling (GMS patches + resize) is done here for efficient CPU-side processing.
    """

    def __init__(
        self,
        samples: List[SampleMeta],
        reader: VideoReaderFactory,
        frame_sampler: FrameSampler,
        is_fr: bool = True,
        is_train: bool = True,
        gms_sampler: Optional[GMSSampler] = None,
        fupic_sampler: Optional[FuPiCSampler] = None,
        detail_resize_sampler: Optional[ResizeSampler] = None,
        resize_sampler: Optional[ResizeSampler] = None,
        fragment_sampler: Optional[FragmentSampler] = None,
        detail_gms_reduce: str = 'none',
        semantic_gms_sampler: Optional[GMSSampler] = None,
        semantic_gms_mode: str = 'avg',
        semantic_gms_legacy_pe: bool = False,
        semantic_gms_adaptive_crop: bool = False,
        adaptive_crop_max_scale: float = 0.0,
        semantic_fragment_sampler: Optional[FragmentSampler] = None,
        semantic_target_size: Optional[int] = None,
        mss_gms_sampler: Optional[GMSSampler] = None,
        vif_branch_cfg: Optional[dict] = None,
        output_dir: Optional[str] = None,
        vif_only_mode: bool = False,
        enable_detail_branch: bool = True,
        enable_semantic_branch: bool = True,
        data_error_fail_fast: bool = True,
        data_error_max_count: int = 8,
        data_error_max_ratio: float = 0.05,
        fr_align_mode: str = 'auto',
        fr_align_ratio_threshold: float = 1.2,
        use_cvqm_npy: bool = False,
        cvqm_npy_segments: int = 4,
        cvqm_npy_colorspace: str = 'bt709',
        resize_antialias: bool = True,
        frame_cache_root: Optional[str] = None,
        sampled_clip_cache_mode: str = 'off',
        sampled_clip_cache_root: Optional[str] = None,
        cache_only: bool = False,
        ref_clip_cache_size: int = 1,
        ref_clip_cache_max_mb: float = 256.0,
        # Phase2/4K augmentation
        aug_hflip: bool = False,
        aug_tflip: bool = False,
        aug_brightness: float = 0.0,
        # Resolution-adaptive GMS: scale patch count based on video resolution
        sem_adaptive_patches: bool = False,
        # Gradient-weighted Top-K GMS cell selection
        gradient_topk_sampling: bool = False,
        gradient_topk_mode: str = 'weighted',
    ):
        self.samples = samples
        self.reader = reader
        self.frame_sampler = frame_sampler
        self.is_fr = is_fr
        self.is_train = is_train
        self.gms_sampler = gms_sampler
        self.fupic_sampler = fupic_sampler
        self.detail_resize_sampler = detail_resize_sampler
        self.resize_sampler = resize_sampler
        self.fragment_sampler = fragment_sampler
        self.detail_gms_reduce = str(detail_gms_reduce or 'none').lower().strip()
        if self.detail_gms_reduce not in ('none', 'avg', 'random1'):
            self.detail_gms_reduce = 'none'
        self.semantic_gms_sampler = semantic_gms_sampler
        self.semantic_gms_mode = str(semantic_gms_mode or 'avg').lower().strip()
        if self.semantic_gms_mode not in ('avg', 'random1', 'stack'):
            self.semantic_gms_mode = 'avg'
        self.semantic_gms_legacy_pe = bool(semantic_gms_legacy_pe)
        self.semantic_gms_adaptive_crop = bool(semantic_gms_adaptive_crop)
        self.adaptive_crop_max_scale = float(adaptive_crop_max_scale)
        self.sem_adaptive_patches = bool(sem_adaptive_patches)
        self.gradient_topk_sampling = bool(gradient_topk_sampling)
        self.gradient_topk_mode = str(gradient_topk_mode or 'weighted').lower().strip()
        self.semantic_fragment_sampler = semantic_fragment_sampler
        self.semantic_target_size = int(semantic_target_size) if semantic_target_size else None
        self.mss_gms_sampler = mss_gms_sampler
        self.vif_branch_cfg = vif_branch_cfg or {}
        self.output_dir = output_dir
        self.vif_mode = str(self.vif_branch_cfg.get('mode', 'aligned')).lower()
        # When VIF branch is disabled, there is NO consumer for the full-resolution
        # dis_yuv / ref_yuv tensors in the model forward pass.  Dropping them from
        # the result dict saves huge amounts of worker→main shared-memory transfer
        # (e.g. 759 MB per 4K sample × 2 for FR mode).
        _vif_enabled = bool(self.vif_branch_cfg.get('enable', True))
        self.skip_fullres_yuv = (not _vif_enabled)
        self.vif_align_with_other_branches = bool(
            self.vif_branch_cfg.get('align_with_other_branches', True)
        )
        # ── GPU preprocess opt-in (env-controlled; safe default: off) ──
        # When enabled, the dataset additionally emits raw YUV420 planes
        # (Y + U_half + V_half) so that downstream (evaluator) can rebuild
        # `resize_dis` on GPU, avoiding the costly CPU-side UV bicubic
        # upsample + stack + patch gather.
        #
        # Modes (via HMF_VQA_GPU_PREPROCESS):
        #   ''      | '0' | 'off'     → legacy CPU path only (default)
        #   '1'     | 'safe'          → emit raw planes IN ADDITION to
        #                               resize_dis (worker still does CPU
        #                               preprocess; GPU rebuild is done in
        #                               evaluator and OVERWRITES resize_dis).
        #                               Primarily for equivalence testing.
        #   '2'     | 'fast'          → emit raw planes and SKIP the CPU
        #                               resize_dis preprocess entirely.
        #                               Maximum speedup; requires evaluator
        #                               GPU rebuild to be live.
        _gpu_prep_env = str(os.environ.get('HMF_VQA_GPU_PREPROCESS', '') or '').strip().lower()
        if _gpu_prep_env in ('', '0', 'off', 'false'):
            self.gpu_preprocess_mode = 0
        elif _gpu_prep_env in ('1', 'safe'):
            self.gpu_preprocess_mode = 1
        elif _gpu_prep_env in ('2', 'fast'):
            self.gpu_preprocess_mode = 2
        else:
            self.gpu_preprocess_mode = 0
        self.vif_max_dense_frames = int(self.vif_branch_cfg.get('max_dense_frames', 32))
        self.vif_cache_enable = bool(self.vif_branch_cfg.get('cache_features', False))
        self.vif_cache_force_rebuild = bool(self.vif_branch_cfg.get('cache_force_rebuild', False))
        self.vif_cache_dir = self.vif_branch_cfg.get('cache_dir', None)
        self.vif_cache_key_mode = str(self.vif_branch_cfg.get('cache_key_mode', 'portable_v2')).lower()
        self.vif_cache_partition_total = int(max(1, self.vif_branch_cfg.get('cache_partition_total', 1)))
        self.vif_cache_partition_index = int(self.vif_branch_cfg.get('cache_partition_index', 0))
        self.vif_cache_partition_on_remaining = bool(
            self.vif_branch_cfg.get('cache_partition_on_remaining', False)
        )
        self.vif_cache_device = str(self.vif_branch_cfg.get('cache_device', 'cpu')).lower()
        # Feature source is always libvmaf (pytorch backend removed).
        _fs = str(self.vif_branch_cfg.get('feature_source', 'libvmaf')).lower()
        if _fs != 'libvmaf':
            logger.warning("feature_source '%s' is deprecated; using libvmaf.", _fs)
        self.vif_feature_source = 'libvmaf'
        self.vif_cache_require_vmaf_score = bool(
            self.vif_branch_cfg.get('cache_require_vmaf_score', False)
        )
        self.vif_ffmpeg_bin = str(self.vif_branch_cfg.get('ffmpeg_bin', 'ffmpeg'))
        # CUDA path is intentionally disabled for libvmaf backend.
        # Keep the old flag for backward compatibility, but ignore it.
        _libvmaf_use_cuda_req = bool(self.vif_branch_cfg.get('libvmaf_use_cuda', False))
        self.vif_libvmaf_use_cuda = False
        if _libvmaf_use_cuda_req:
            logger.warning(
                "vif_branch.libvmaf_use_cuda is deprecated and ignored; libvmaf backend runs on CPU."
            )
        self.vif_libvmaf_n_threads = int(self.vif_branch_cfg.get('libvmaf_n_threads', 0))
        self.vif_libvmaf_model = str(self.vif_branch_cfg.get('libvmaf_model', '') or '')
        self._ffmpeg_filters_cached = None
        self._libvmaf_checked = False
        self._has_libvmaf = False
        if self.vif_cache_partition_index < 0:
            self.vif_cache_partition_index = 0
        if self.vif_cache_partition_index >= self.vif_cache_partition_total:
            self.vif_cache_partition_index = self.vif_cache_partition_index % self.vif_cache_partition_total
        self._vif_extractor = None
        self._vif_extractor_device = torch.device('cpu')
        self._vmaf_svr_scorer = None
        self._vif_feature_version = 'libvmaf_core_v1'
        self._vif_cache_checked = False
        self._vif_cache_prefix_index = None
        self.vif_only_mode = bool(vif_only_mode)
        self.vif_cache_prebuild = bool(self.vif_branch_cfg.get('prebuild_cache', False))
        self._vif_cache_status_logged = False
        self.enable_detail_branch = bool(enable_detail_branch)
        self.enable_semantic_branch = bool(enable_semantic_branch)
        self.data_error_fail_fast = bool(data_error_fail_fast)
        self.data_error_max_count = int(max(1, data_error_max_count))
        self.data_error_max_ratio = float(max(0.0, data_error_max_ratio))
        self.fr_align_mode = str(fr_align_mode).lower().strip()
        self.fr_align_ratio_threshold = float(max(1.0, fr_align_ratio_threshold))
        self.use_cvqm_npy = bool(use_cvqm_npy)
        self.cvqm_npy_segments = int(max(1, cvqm_npy_segments))
        self.cvqm_npy_colorspace = str(cvqm_npy_colorspace or 'bt709').lower().strip()
        if self.cvqm_npy_colorspace not in ('bt709', 'bt601'):
            self.cvqm_npy_colorspace = 'bt709'
        self.resize_antialias = bool(resize_antialias)
        # Frame cache: raw binary cache on NVMe for CVQM evaluation acceleration.
        self.frame_cache_root = str(frame_cache_root).strip() if frame_cache_root else None
        if self.frame_cache_root and not os.path.isdir(self.frame_cache_root):
            _msg = f"[FrameCache] ★ cache_root does NOT exist, DISABLED: {self.frame_cache_root}"
            logger.warning(_msg)
            print(_msg, flush=True)
            self.frame_cache_root = None
        elif self.frame_cache_root:
            _msg = f"[FrameCache] ★ cache_root EXISTS, ENABLED: {self.frame_cache_root}"
            logger.info(_msg)
            print(_msg, flush=True)
        else:
            print("[FrameCache] ★ No frame_cache_root configured (cache OFF)", flush=True)
        self._frame_cache_logged = False
        self._frame_cache_miss_logged = False  # Whether the first miss has been logged
        self.sampled_clip_cache_mode = str(sampled_clip_cache_mode or 'off').lower().strip()
        if self.sampled_clip_cache_mode not in ('off', 'read', 'write', 'readwrite'):
            logger.warning(
                "Unknown sampled_clip_cache_mode '%s', fallback to off",
                self.sampled_clip_cache_mode,
            )
            self.sampled_clip_cache_mode = 'off'
        self.sampled_clip_cache_root = (
            str(sampled_clip_cache_root).strip() if sampled_clip_cache_root else None
        )
        self.cache_only = bool(cache_only)
        self._sampled_clip_cache_logged = False
        self._sampled_clip_cache_write_logged = False
        self._sampled_clip_cache_mkdir_done = False
        self.ref_clip_cache_size = int(max(0, ref_clip_cache_size))
        self.ref_clip_cache_max_bytes = int(max(0.0, float(ref_clip_cache_max_mb)) * 1024 * 1024)
        self._ref_clip_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._ref_clip_cache_bytes = 0
        self._ref_clip_cache_logged = False
        self._ref_clip_cache_skip_logged = False
        if self.fr_align_mode not in ('auto', 'normalized', 'index'):
            logger.warning("Unknown fr_align_mode '%s', fallback to auto", self.fr_align_mode)
            self.fr_align_mode = 'auto'
        self._fetch_count = 0
        self._error_count = 0
        # Phase2/4K augmentation flags
        self.aug_hflip = bool(aug_hflip) and self.is_train
        self.aug_tflip = bool(aug_tflip) and self.is_train
        self.aug_brightness = float(aug_brightness) if self.is_train else 0.0

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _to_int_or_none(v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _dis_spec(self, sample) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
        w = self._to_int_or_none(sample.width)
        h = self._to_int_or_none(sample.height)
        bd = self._to_int_or_none(sample.bitdepth)
        pf = str(sample.pix_fmt) if sample.pix_fmt is not None else None
        # Auto-detect pre-resized files by checking if the file path contains
        # "resize" (e.g., "2160p_resize1080p/", "Resize1080p").
        # This only triggers for raw YUV files where annotation says >1920 width
        # but file was pre-resized to 1080p for faster training/eval.
        if w and h and bd and w > 1920 and hasattr(sample, 'dis_path') and sample.dis_path:
            if self._path_has_resize_hint(sample.dis_path):
                w, h = self._infer_actual_resolution(sample.dis_path, w, h, bd)
        return w, h, bd, pf

    def _ref_spec(self, sample) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
        dw, dh, dbd, dpf = self._dis_spec(sample)
        rw = self._to_int_or_none(getattr(sample, 'ref_width', None))
        rh = self._to_int_or_none(getattr(sample, 'ref_height', None))
        rbd = self._to_int_or_none(getattr(sample, 'ref_bitdepth', None))
        rpf = str(getattr(sample, 'ref_pix_fmt', None)) if getattr(sample, 'ref_pix_fmt', None) is not None else None
        w = rw or dw
        h = rh or dh
        bd = rbd or dbd
        pf_out = rpf or dpf
        # Same resize-hint check for ref path
        ref_path = str(getattr(sample, 'ref_path', '') or '')
        if w and h and bd and w > 1920 and ref_path:
            if self._path_has_resize_hint(ref_path):
                w, h = self._infer_actual_resolution(ref_path, w, h, bd)
        return w, h, bd, pf_out

    _resolution_cache: dict = {}  # path → (w, h) cache to avoid repeated os.stat

    @staticmethod
    def _path_has_resize_hint(path: str) -> bool:
        """Check if path contains 'resize' (case-insensitive), indicating a pre-resized file."""
        return 'resize' in path.lower()

    def _infer_actual_resolution(self, path: str, ann_w: int, ann_h: int, bd: int) -> Tuple[int, int]:
        """Infer actual resolution for a pre-resized file (path contains 'resize').

        Since this is only called when the path has a 'resize' hint, the file
        is expected to be pre-resized to 1080p. Check 1080p FIRST.
        Fall back to annotation resolution only if 1080p doesn't match.
        """
        if path in self.__class__._resolution_cache:
            return self.__class__._resolution_cache[path]
        try:
            if not os.path.isfile(path):
                return ann_w, ann_h
            file_size = os.path.getsize(path)
            if file_size == 0:
                return ann_w, ann_h
            bps = 2 if bd > 8 else 1

            # Step 1: Path has 'resize' hint → check 1080p first (expected actual resolution)
            for test_w, test_h in [(1920, 1080), (1080, 1920)]:
                test_frame_bytes = int(test_h * test_w * 1.5) * bps
                if test_frame_bytes > 0 and file_size % test_frame_bytes == 0:
                    implied_frames = file_size // test_frame_bytes
                    if 0 < implied_frames < 100000:
                        if not getattr(self.__class__, '_resize_detect_logged', False):
                            self.__class__._resize_detect_logged = True
                            logger.info(
                                "[ResizeDetect] File %s: annotation=%dx%d, actual=%dx%d "
                                "(pre-resized path, %d frames)",
                                os.path.basename(path), ann_w, ann_h,
                                test_w, test_h, implied_frames,
                            )
                        self.__class__._resolution_cache[path] = (test_w, test_h)
                        return test_w, test_h

            # Step 2: 1080p doesn't match → try annotation resolution
            ann_frame_bytes = int(ann_h * ann_w * 1.5) * bps
            if ann_frame_bytes > 0 and file_size % ann_frame_bytes == 0:
                self.__class__._resolution_cache[path] = (ann_w, ann_h)
                return ann_w, ann_h
        except Exception:
            pass
        self.__class__._resolution_cache[path] = (ann_w, ann_h)
        return ann_w, ann_h

    def _resolve_fr_align_mode(self, sample, total_ref: int, total_dis: int) -> str:
        mode = self.fr_align_mode
        if mode in ('normalized', 'index'):
            return mode
        if total_ref <= 0 or total_dis <= 0:
            return 'normalized'
        if total_ref == total_dis:
            return 'normalized'

        ds = str(getattr(sample, 'dataset_name', '')).upper()
        if ds == 'CVQM':
            return 'index'
        if ds == 'MCML4K':
            return 'index'

        ratio = max(total_ref, total_dis) / max(1.0, float(min(total_ref, total_dis)))
        if ratio >= self.fr_align_ratio_threshold:
            return 'index'
        return 'normalized'

    @staticmethod
    def _sanitize_cache_token(text: str, max_len: int = 64) -> str:
        if text is None:
            text = 'na'
        text = re.sub(r'[^0-9a-zA-Z._-]+', '_', str(text)).strip('._-')
        if not text:
            text = 'na'
        return text[:max_len]

    @staticmethod
    def _file_sig(path: Optional[str]) -> str:
        if not path:
            return 'none'
        try:
            st = os.stat(path)
            return f'{os.path.abspath(path)}|sz={st.st_size}|mt={st.st_mtime_ns}'
        except Exception:
            return os.path.abspath(path)

    @staticmethod
    def _portable_file_sig(path: Optional[str]) -> str:
        """
        Cross-machine stable file signature:
        - no absolute path
        - no mtime
        Uses parent/stem token + file size for practical uniqueness.
        """
        if not path:
            return 'none'
        parent = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        token = f'{parent}__{stem}'
        try:
            st = os.stat(path)
            return f'{token}|sz={st.st_size}'
        except Exception:
            return token

    def _build_vif_cache_key_src(self, sample) -> str:
        if self.vif_cache_key_mode in ('strict_v1', 'path_mtime_v1'):
            ref_sig = self._file_sig(sample.ref_path)
            dis_sig = self._file_sig(sample.dis_path)
        else:
            ref_sig = self._portable_file_sig(sample.ref_path)
            dis_sig = self._portable_file_sig(sample.dis_path)
        dis_w, dis_h, dis_bd, dis_pf = self._dis_spec(sample)
        ref_w, ref_h, ref_bd, ref_pf = self._ref_spec(sample)
        return (
            f"dataset={sample.dataset_name}|video={sample.video_id}|"
            f"ref={ref_sig}|dis={dis_sig}|"
            f"dw={dis_w}|dh={dis_h}|dbd={dis_bd}|dpf={dis_pf}|"
            f"rw={ref_w}|rh={ref_h}|rbd={ref_bd}|rpf={ref_pf}|"
            f"mode=dense|max={self.vif_max_dense_frames}|policy=linspace_round_v1|"
            f"keymode={self.vif_cache_key_mode}|"
            f"src={self.vif_feature_source}|"
            f"ver={getattr(self, '_vif_feature_version', 'unknown')}|"
            f"ns={self.vif_branch_cfg.get('num_scales', 4)}|"
            f"ws={self.vif_branch_cfg.get('window_size', 5)}|"
            f"qp={self.vif_branch_cfg.get('quantile_pool_size', 64)}|"
            f"adm={self.vif_branch_cfg.get('use_adm', True)}|"
            f"mot={self.vif_branch_cfg.get('use_motion', True)}|"
            f"gr={self.vif_branch_cfg.get('use_grad_ratio', False)}|"
            f"lap={self.vif_branch_cfg.get('use_lap_ratio', False)}|"
            f"ti={self.vif_branch_cfg.get('use_ti', False)}|"
            f"vmaf_cuda={int(self.vif_libvmaf_use_cuda)}|"
            f"vmaf_nt={int(max(0, self.vif_libvmaf_n_threads))}|"
            f"vmaf_model={self.vif_libvmaf_model}"
        )

    @staticmethod
    def _stable_partition_hash(text: str) -> int:
        return int(hashlib.sha1(text.encode('utf-8')).hexdigest()[:16], 16)

    def _cache_partition_bucket(self, sample) -> int:
        src = f'{sample.dataset_name}|{sample.video_id}'
        return self._stable_partition_hash(src) % max(1, self.vif_cache_partition_total)

    def _effective_prebuild_partition(self) -> Tuple[int, int]:
        """
        Compose user-defined machine partition with DDP rank sharding.
        This prevents rank0-only prebuild bottlenecks and NCCL timeouts.
        """
        total = int(max(1, self.vif_cache_partition_total))
        index = int(self.vif_cache_partition_index % total)
        if dist.is_available() and dist.is_initialized():
            world = int(dist.get_world_size())
            rank = int(dist.get_rank())
            total = total * max(1, world)
            index = index * max(1, world) + rank
        return total, index

    def _in_prebuild_partition(self, sample) -> bool:
        # Legacy/default mode: hash over all videos.
        total, index = self._effective_prebuild_partition()
        src = f'{sample.dataset_name}|{sample.video_id}'
        bucket = self._stable_partition_hash(src) % max(1, total)
        return bucket == index

    def _resolve_vif_cache_device(self) -> torch.device:
        # libvmaf backend extracts features through ffmpeg/libvmaf CPU path.
        return torch.device('cpu')

    def _cache_display_token(self, sample) -> str:
        """Human-readable token for cache filename (keeps QP and source bucket)."""
        if sample.dis_path:
            parent = os.path.basename(os.path.dirname(sample.dis_path))
            stem = os.path.splitext(os.path.basename(sample.dis_path))[0]
            token = f'{parent}__{stem}'
        else:
            token = str(sample.video_id)
        return self._sanitize_cache_token(token, max_len=112)

    def _vif_cache_bucket_name(self) -> str:
        """
        Cache subdirectory bucket for the current VIF dense sampling/feature recipe.
        Different sampling settings go to different subfolders to avoid mixing.
        """
        mode = str(self.vif_mode).lower()
        align = 1 if self.vif_align_with_other_branches else 0
        # Include resolved extraction device so CPU/GPU caches are isolated.
        dev = self._resolve_vif_cache_device().type
        base = (
            f"mode={mode}__align={align}__max={int(self.vif_max_dense_frames)}__"
            f"src={self._sanitize_cache_token(self.vif_feature_source, 16)}__"
            f"ns={int(self.vif_branch_cfg.get('num_scales', 4))}__"
            f"ws={int(self.vif_branch_cfg.get('window_size', 5))}__"
            f"qp={int(self.vif_branch_cfg.get('quantile_pool_size', 64))}__"
            f"adm={int(bool(self.vif_branch_cfg.get('use_adm', True)))}__"
            f"mot={int(bool(self.vif_branch_cfg.get('use_motion', True)))}__"
            f"gr={int(bool(self.vif_branch_cfg.get('use_grad_ratio', False)))}__"
            f"lap={int(bool(self.vif_branch_cfg.get('use_lap_ratio', False)))}__"
            f"ti={int(bool(self.vif_branch_cfg.get('use_ti', False)))}__"
            f"dev={dev}__"
            f"ver={self._sanitize_cache_token(getattr(self, '_vif_feature_version', 'unknown'), 24)}"
        )
        bucket_hash = hashlib.sha1(base.encode('utf-8')).hexdigest()[:10]
        return f"{base}__h={bucket_hash}"

    def _sample_dense_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0] * max(1, self.vif_max_dense_frames)
        n = min(total_frames, max(1, self.vif_max_dense_frames))
        if n >= total_frames:
            return list(range(total_frames))
        idx = torch.linspace(0, total_frames - 1, steps=n).round().long()
        return idx.tolist()

    def _sample_dense_aligned_indices(
        self,
        total_ref: int,
        total_dis: int,
        align_mode: str = 'normalized',
    ) -> Tuple[List[int], List[int]]:
        mode = str(align_mode).lower().strip()
        if mode not in ('normalized', 'index'):
            mode = 'normalized'
        if mode == 'index':
            ridx = self._sample_dense_indices(total_ref)
            if total_dis <= 0:
                didx = [0] * len(ridx)
            else:
                didx = [min(max(int(i), 0), total_dis - 1) for i in ridx]
            return ridx, didx

        n = max(1, self.vif_max_dense_frames)
        max_total = max(int(total_ref), int(total_dis), 1)
        n = min(n, max_total)
        if n <= 1:
            ridx = [0 if total_ref > 0 else 0]
            didx = [0 if total_dis > 0 else 0]
            return ridx, didx
        t = torch.linspace(0.0, 1.0, steps=n)
        if total_ref > 1:
            ridx = (t * float(total_ref - 1)).round().long().tolist()
        else:
            ridx = [0] * n
        if total_dis > 1:
            didx = (t * float(total_dis - 1)).round().long().tolist()
        else:
            didx = [0] * n
        return ridx, didx

    def _resolve_vif_cache_dir(self) -> Optional[str]:
        if self.vif_cache_dir:
            root = os.path.abspath(self.vif_cache_dir)
            return os.path.join(root, self._vif_cache_bucket_name())
        if self.output_dir:
            return os.path.join(self.output_dir, 'vif_feature_cache')
        return None

    def _resolve_vif_cache_root(self) -> Optional[str]:
        if self.vif_cache_dir:
            return os.path.abspath(self.vif_cache_dir)
        if self.output_dir:
            return os.path.join(self.output_dir, 'vif_feature_cache')
        return None

    def _ensure_vif_extractor(self):
        """Initialize libvmaf feature extraction backend."""
        self._vif_feature_version = 'libvmaf_core_v1'
        self._vif_extractor = None
        self._vif_extractor_device = self._resolve_vif_cache_device()
        self._ensure_vmaf_svr_scorer()

    def _ensure_vmaf_svr_scorer(self):
        """Lazily initialize VMAFSVRScorer for computing vmaf_frame_score at cache time."""
        if getattr(self, '_vmaf_svr_scorer', None) is not None:
            return
        try:
            from src.models.branches.vmaf_like_branch import VMAFSVRScorer
            vmaf_model_dir = self.vif_branch_cfg.get('vmaf_model_dir', None)
            if isinstance(vmaf_model_dir, str) and not vmaf_model_dir.strip():
                vmaf_model_dir = None
            self._vmaf_svr_scorer = VMAFSVRScorer(model_dir=vmaf_model_dir)
        except Exception:
            self._vmaf_svr_scorer = None

    @staticmethod
    def _escape_filter_value(v: str) -> str:
        return str(v).replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")

    def _ffmpeg_filter_set(self) -> set:
        if self._ffmpeg_filters_cached is not None:
            return self._ffmpeg_filters_cached
        filters = set()
        try:
            proc = subprocess.run(
                [self.vif_ffmpeg_bin, '-hide_banner', '-filters'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            txt = proc.stdout or ''
            for line in txt.splitlines():
                if ' libvmaf ' in f' {line} ':
                    filters.add('libvmaf')
        except Exception:
            pass
        self._ffmpeg_filters_cached = filters
        return filters

    def _check_libvmaf_once(self):
        if self._libvmaf_checked:
            return
        self._libvmaf_checked = True
        fset = self._ffmpeg_filter_set()
        self._has_libvmaf = 'libvmaf' in fset
        logger.info(
            "libvmaf backend check: ffmpeg_bin=%s, libvmaf=%s, mode=cpu",
            self.vif_ffmpeg_bin,
            self._has_libvmaf,
        )

    @staticmethod
    def _is_raw_path(path: str) -> bool:
        ext = os.path.splitext(str(path))[1].lower()
        return ext in {'.yuv', '.y4m', '.nv12', '.raw'}

    def _ffmpeg_input_args(self, path: str, sample, is_ref: bool = False) -> List[str]:
        if self._is_raw_path(path):
            if is_ref:
                w, h, _bd, pf = self._ref_spec(sample)
            else:
                w, h, _bd, pf = self._dis_spec(sample)
            if w is None or h is None:
                w = int(sample.width or 1920)
                h = int(sample.height or 1080)
            pf = str(pf or 'yuv420p10le')
            return ['-f', 'rawvideo', '-pix_fmt', pf, '-s:v', f'{w}x{h}', '-i', path]
        return ['-i', path]

    @staticmethod
    def _metric_pick(metrics: dict, candidates: List[str]) -> Optional[float]:
        for k in candidates:
            if k in metrics:
                try:
                    return float(metrics[k])
                except Exception:
                    continue
        return None

    def _build_ti_feat_from_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.numel() == 0:
            return torch.zeros(4, dtype=torch.float32)
        if not bool(self.vif_branch_cfg.get('use_motion', True)):
            return torch.zeros(4, dtype=torch.float32)
        m2 = frame_features[:, -1].float()
        vif_s0 = frame_features[:, 0].float()
        m2_mean = m2.mean()
        m2_p90 = torch.quantile(m2, 0.9)
        m2_std = m2.std(unbiased=False)
        vif_mean = vif_s0.mean()
        interaction = m2_mean * (1.0 - vif_mean)
        return torch.stack([m2_mean, m2_p90, m2_std, interaction], dim=0).float()

    def _parse_libvmaf_json_to_pack(self, json_path: str) -> Optional[dict]:
        try:
            with open(json_path, 'r') as f:
                obj = json.load(f)
        except Exception:
            return None
        frames = obj.get('frames', [])
        if not isinstance(frames, list) or not frames:
            return None
        frames = sorted(frames, key=lambda x: int(x.get('frameNum', 0)))

        num_scales = int(self.vif_branch_cfg.get('num_scales', 4))
        if num_scales != 4:
            logger.warning("libvmaf backend currently supports num_scales=4 only, got %d", num_scales)
            return None
        use_adm = bool(self.vif_branch_cfg.get('use_adm', True))
        use_motion = bool(self.vif_branch_cfg.get('use_motion', True))
        if bool(self.vif_branch_cfg.get('use_grad_ratio', False)) or bool(self.vif_branch_cfg.get('use_lap_ratio', False)):
            logger.warning("libvmaf backend does not support grad/lap extra features.")
            return None
        if bool(self.vif_branch_cfg.get('use_ti', False)):
            logger.warning("libvmaf backend does not support extra TI feature.")
            return None

        feats = []
        vmaf_scores = []
        has_vmaf = True
        for fr in frames:
            m = fr.get('metrics', {})
            if not isinstance(m, dict):
                return None
            v0 = self._metric_pick(m, ['integer_vif_scale0', 'float_vif_scale0', 'vif_scale0'])
            v1 = self._metric_pick(m, ['integer_vif_scale1', 'float_vif_scale1', 'vif_scale1'])
            v2 = self._metric_pick(m, ['integer_vif_scale2', 'float_vif_scale2', 'vif_scale2'])
            v3 = self._metric_pick(m, ['integer_vif_scale3', 'float_vif_scale3', 'vif_scale3'])
            if None in (v0, v1, v2, v3):
                return None
            row = [v0, v1, v2, v3]
            if use_adm:
                adm2 = self._metric_pick(m, ['integer_adm2', 'float_adm2', 'adm2'])
                if adm2 is None:
                    return None
                row.append(adm2)
            if use_motion:
                motion2 = self._metric_pick(m, ['integer_motion2', 'float_motion2', 'motion2', 'motion'])
                if motion2 is None:
                    return None
                row.append(motion2)
            feats.append(row)
            if has_vmaf:
                vmaf = self._metric_pick(m, ['integer_vmaf', 'float_vmaf', 'vmaf', 'vmaf_score'])
                if vmaf is None:
                    has_vmaf = False
                else:
                    vmaf_scores.append(vmaf)

        frame_features = torch.tensor(feats, dtype=torch.float32)
        keep_idx = None
        if frame_features.shape[0] > int(self.vif_max_dense_frames):
            keep_idx = torch.linspace(
                0, frame_features.shape[0] - 1, steps=int(self.vif_max_dense_frames)
            ).round().long()
            frame_features = frame_features.index_select(0, keep_idx)
        frame_mask = torch.ones(frame_features.shape[0], dtype=torch.float32)
        ti_feat = self._build_ti_feat_from_frame_features(frame_features)
        out = {
            'frame_features': frame_features,
            'frame_mask': frame_mask,
            'ti_video_feat': ti_feat,
        }
        if has_vmaf and len(vmaf_scores) == len(feats):
            vmaf_frame = torch.tensor(vmaf_scores, dtype=torch.float32)
            if keep_idx is not None:
                vmaf_frame = vmaf_frame.index_select(0, keep_idx)
            out['vmaf_frame_score'] = vmaf_frame
        return out

    @staticmethod
    def _raw_yuv_frame_count(path: str, w: Optional[int], h: Optional[int],
                             bd: Optional[int], pf: Optional[str]) -> Optional[int]:
        """Estimate frame count from raw YUV file size (bytes / bytes_per_frame)."""
        try:
            sz = os.path.getsize(path)
        except OSError:
            return None
        if not w or not h or w <= 0 or h <= 0:
            return None
        bps = 2 if int(bd or 8) > 8 else 1  # bytes per sample
        pf_s = str(pf or '').lower()
        if '444' in pf_s:
            chroma = 3.0
        elif '422' in pf_s:
            chroma = 2.0
        else:
            chroma = 1.5  # yuv420 default
        bytes_per_frame = int(w * h * chroma * bps)
        if bytes_per_frame <= 0:
            return None
        return sz // bytes_per_frame

    def _run_libvmaf_ffmpeg(self, sample) -> Optional[dict]:
        self._check_libvmaf_once()
        flt = 'libvmaf'
        if not self._has_libvmaf:
            return None

        with tempfile.TemporaryDirectory(prefix='hmf_vmaf_') as td:
            json_path = os.path.join(td, 'vmaf_features.json')
            log_path = self._escape_filter_value(json_path)
            opts = [f"log_fmt=json", f"log_path={log_path}"]
            if self.vif_libvmaf_n_threads > 0:
                opts.append(f"n_threads={int(self.vif_libvmaf_n_threads)}")
            if self.vif_libvmaf_model:
                opts.append(f"model={self.vif_libvmaf_model}")
            kv = ':'.join(opts)

            graph = (
                f"[0:v]setpts=PTS-STARTPTS[dis];"
                f"[1:v]setpts=PTS-STARTPTS[ref];"
                f"[dis][ref]{flt}={kv}"
            )

            cmd = [self.vif_ffmpeg_bin, '-hide_banner', '-nostats', '-loglevel', 'error']
            cmd += self._ffmpeg_input_args(sample.dis_path, sample, is_ref=False)
            cmd += self._ffmpeg_input_args(sample.ref_path, sample, is_ref=True)

            # For raw YUV inputs, libvmaf processes until the LONGER stream ends.
            # If DIS has fewer frames than REF (e.g. 10-bit encoded clip vs long REF),
            # frames beyond DIS EOF are garbage → vif_s0 ≈ 0.
            # Fix: add -frames:v N to cap output at min(dis_n, ref_n).
            frame_limit = None
            if self._is_raw_path(sample.dis_path):
                dis_w, dis_h, dis_bd, dis_pf = self._dis_spec(sample)
                dis_n = self._raw_yuv_frame_count(sample.dis_path, dis_w, dis_h, dis_bd, dis_pf)
                if dis_n is not None and dis_n > 0:
                    frame_limit = dis_n
            if self._is_raw_path(sample.ref_path):
                ref_w, ref_h, ref_bd, ref_pf = self._ref_spec(sample)
                ref_n = self._raw_yuv_frame_count(sample.ref_path, ref_w, ref_h, ref_bd, ref_pf)
                if ref_n is not None and ref_n > 0:
                    frame_limit = ref_n if frame_limit is None else min(frame_limit, ref_n)

            extra_out = []
            if frame_limit is not None:
                extra_out = ['-frames:v', str(int(frame_limit))]

            cmd += ['-lavfi', graph] + extra_out + ['-f', 'null', '-']

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or '').strip().splitlines()
                if err:
                    logger.warning("libvmaf(cpu) failed for %s: %s", sample.video_id, err[-1])
                else:
                    logger.warning("libvmaf(cpu) failed for %s (empty stderr)", sample.video_id)
                return None
            return self._parse_libvmaf_json_to_pack(json_path)

    def _extract_libvmaf_pack(self, sample) -> Optional[dict]:
        return self._run_libvmaf_ffmpeg(sample)

    def _vif_cache_path(self, sample) -> Optional[Tuple[str, str]]:
        if not self.vif_cache_enable:
            return None
        cache_root = self._resolve_vif_cache_dir()
        if cache_root is None:
            return None
        if not self._vif_cache_checked:
            os.makedirs(cache_root, exist_ok=True)
            self._vif_cache_checked = True

        key_src = self._build_vif_cache_key_src(sample)
        key = hashlib.sha1(key_src.encode('utf-8')).hexdigest()
        ds = self._sanitize_cache_token(sample.dataset_name, max_len=24)
        vid = self._cache_display_token(sample)
        cache_name = f'{ds}__{vid}__{key[:16]}.pt'
        return os.path.join(cache_root, cache_name), key

    def _cache_prefix(self, sample) -> str:
        ds = self._sanitize_cache_token(sample.dataset_name, max_len=24)
        vid = self._cache_display_token(sample)
        return f'{ds}__{vid}'

    def _ensure_cache_prefix_index(self):
        if self._vif_cache_prefix_index is not None:
            return
        self._vif_cache_prefix_index = {}
        cache_root = self._resolve_vif_cache_dir()
        if cache_root is None or not os.path.isdir(cache_root):
            return
        try:
            for ent in os.scandir(cache_root):
                if not ent.is_file() or not ent.name.endswith('.pt'):
                    continue
                # Name pattern: <prefix>__<hash16>.pt ; prefix may contain "__".
                base = ent.name[:-3]
                if '__' not in base:
                    continue
                prefix, _suffix = base.rsplit('__', 1)
                self._vif_cache_prefix_index.setdefault(prefix, []).append(ent.path)
        except Exception:
            pass

    def _vif_prefix_compat_cache_paths(self, sample) -> List[str]:
        """
        Candidate cache files for cross-machine reuse:
        same dataset/video prefix, different hash suffix.
        """
        self._ensure_cache_prefix_index()
        if self._vif_cache_prefix_index is None:
            return []
        prefix = self._cache_prefix(sample)
        paths = self._vif_cache_prefix_index.get(prefix, [])
        if not paths:
            return []
        return sorted(paths)

    def _vif_legacy_cache_path(self, sample) -> Optional[str]:
        cache_root_bucket = self._resolve_vif_cache_dir()
        cache_root_plain = self._resolve_vif_cache_root()
        if cache_root_bucket is None and cache_root_plain is None:
            return None
        key_src = self._build_vif_cache_key_src(sample)
        key = hashlib.sha1(key_src.encode('utf-8')).hexdigest()
        candidates = []
        if cache_root_bucket is not None:
            candidates.append(os.path.join(cache_root_bucket, f'{key}.pt'))
        if cache_root_plain is not None:
            candidates.append(os.path.join(cache_root_plain, f'{key}.pt'))
        for legacy in candidates:
            if os.path.exists(legacy):
                return legacy
        return None

    def _has_vif_cache_file(self, sample) -> bool:
        ret = self._vif_cache_path(sample)
        if ret is None:
            return False
        cache_path, _ = ret
        if os.path.exists(cache_path):
            return True
        if self._vif_legacy_cache_path(sample) is not None:
            return True
        compat_paths = self._vif_prefix_compat_cache_paths(sample)
        return bool(compat_paths)

    def _collect_prebuild_samples(self) -> Tuple[List, int, int, int, int, int, int]:
        """
        Returns:
          local_samples: samples assigned to current partition
          part_total: effective partition total (with DDP rank composition)
          part_index: effective partition index (0-based)
          shard_skip: non-local build-target count for reporting
          remaining_total: number of cache-miss videos across all samples
          global_hit: cache-hit count across all samples
          global_miss: cache-miss count across all samples
        """
        part_total, part_index = self._effective_prebuild_partition()
        if not self.vif_cache_partition_on_remaining:
            local = [s for s in self.samples if self._in_prebuild_partition(s)]
            # Fast path for status/progress planning: avoid loading all cache files.
            # Actual per-sample validity is still checked in _try_load_vif_cached_features
            # during prebuild/read.
            global_hit = sum(1 for s in self.samples if self._has_vif_cache_file(s))
            global_miss = max(0, len(self.samples) - global_hit)
            shard_skip = len(self.samples) - len(local)
            remaining_total = global_miss
            return local, part_total, part_index, shard_skip, remaining_total, global_hit, global_miss

        # Remaining-only partitioning: split only unresolved (cache-miss) videos.
        # Here we must use strict cache validity (not just file existence),
        # otherwise stale/incomplete cache files can be treated as "hit" and
        # later fall back to costly online dense extraction during training.
        ordered = sorted(
            self.samples,
            key=lambda s: (str(getattr(s, 'dataset_name', '')), str(getattr(s, 'video_id', ''))),
        )
        remaining = []
        global_hit = 0
        for s in ordered:
            if self._try_load_vif_cached_features(s) is None:
                remaining.append(s)
            else:
                global_hit += 1
        global_miss = len(remaining)
        local = [s for idx, s in enumerate(remaining) if (idx % max(1, part_total)) == part_index]
        shard_skip = max(0, global_miss - len(local))
        remaining_total = global_miss
        return local, part_total, part_index, shard_skip, remaining_total, global_hit, global_miss

    def _is_valid_cached_obj(self, obj: dict, expected_key: str, sample=None) -> bool:
        if not isinstance(obj, dict):
            return False
        for req in ('frame_features', 'frame_mask', 'ti_video_feat'):
            if req not in obj:
                return False
        if self.vif_cache_require_vmaf_score:
            if 'vmaf_frame_score' not in obj or not isinstance(obj.get('vmaf_frame_score'), torch.Tensor):
                return False
        meta = obj.get('_cache_meta', None)
        if meta is None:
            # Backward-compatible for old cache files.
            return True
        if not isinstance(meta, dict):
            return False
        if str(meta.get('cache_key', '')) == expected_key:
            return True

        # Cross-machine compatibility fallback: accept old-key cache files
        # if they match dataset/video and feature recipe.
        if sample is None:
            return False
        if str(meta.get('dataset_name', '')) != str(sample.dataset_name):
            return False
        if str(meta.get('video_id', '')) != str(sample.video_id):
            return False
        if bool(meta.get('dense_mode', True)) is not True:
            return False
        if int(meta.get('max_dense_frames', -1)) != int(self.vif_max_dense_frames):
            return False
        if str(meta.get('feature_version', 'unknown')) != str(getattr(self, '_vif_feature_version', 'unknown')):
            return False
        # Optional stricter spec check for newer cache metadata.
        dis_w, dis_h, dis_bd, dis_pf = self._dis_spec(sample)
        ref_w, ref_h, ref_bd, ref_pf = self._ref_spec(sample)
        for key, exp in (
            ('dis_width', dis_w), ('dis_height', dis_h), ('dis_bitdepth', dis_bd), ('dis_pix_fmt', dis_pf),
            ('ref_width', ref_w), ('ref_height', ref_h), ('ref_bitdepth', ref_bd), ('ref_pix_fmt', ref_pf),
        ):
            if key in meta and str(meta.get(key)) != str(exp):
                return False
        return True

    def log_vif_cache_status(self):
        """Log dense-VIF cache hit/miss once at dataset build time."""
        if self._vif_cache_status_logged:
            return
        self._vif_cache_status_logged = True
        if not self.vif_cache_enable:
            return
        use_vif_dense = (
            self.is_fr and
            (self.vif_mode == 'dense') and
            (not self.vif_align_with_other_branches)
        )
        if not use_vif_dense:
            return
        if not self.samples:
            return

        cache_root = self._resolve_vif_cache_dir()
        if cache_root is None:
            return

        local_samples, part_total, part_index, shard_skip, remaining_total, global_hit, global_miss = self._collect_prebuild_samples()
        effective_total = len(local_samples)
        if self.vif_cache_partition_on_remaining:
            local_hit = 0
            local_miss = effective_total
        else:
            local_hit = sum(1 for s in local_samples if self._try_load_vif_cached_features(s) is not None)
            local_miss = max(0, effective_total - local_hit)
        basis = 'remaining' if self.vif_cache_partition_on_remaining else 'all'
        resolved_dev = self._resolve_vif_cache_device().type
        logger.info(
            f"VIF cache status ({'train' if self.is_train else 'eval'}): "
            f"hit={global_hit}/{len(self.samples)}, miss={global_miss}, "
            f"local_hit={local_hit}, local_build_target={local_miss}, shard_skip={shard_skip}, "
            f"remaining_total={remaining_total}, partition_basis={basis}, "
            f"partition={part_index}/{part_total}, "
            f"dir={cache_root}, force_rebuild={self.vif_cache_force_rebuild}, "
            f"key_mode={self.vif_cache_key_mode}, cache_device={self.vif_cache_device}, "
            f"cache_device_resolved={resolved_dev}, feature_source={self.vif_feature_source}"
        )

    def prebuild_vif_cache(self):
        """
        Prebuild VIF dense cache with progress display.
        Runs on rank0 in DDP; other ranks wait at barrier.
        """
        use_vif_dense = (
            self.is_fr and
            (self.vif_mode == 'dense') and
            (not self.vif_align_with_other_branches) and
            self.vif_only_mode and
            self.vif_cache_enable and
            self.vif_cache_prebuild
        )
        if not use_vif_dense:
            return

        ddp = dist.is_available() and dist.is_initialized()
        rank = int(dist.get_rank()) if ddp else 0
        local_samples, part_total, part_index, shard_skip, remaining_total, global_hit, global_miss = self._collect_prebuild_samples()
        total = len(local_samples)
        basis = 'remaining' if self.vif_cache_partition_on_remaining else 'all'
        resolved_dev = self._resolve_vif_cache_device().type
        if self.vif_cache_partition_on_remaining:
            local_hit_plan = 0
            local_build_target = total
        else:
            local_hit_plan = sum(1 for s in local_samples if self._try_load_vif_cached_features(s) is not None)
            local_build_target = max(0, total - local_hit_plan)

        logger.info(
            f'Prebuilding VIF cache ({ "train" if self.is_train else "eval" }): '
            f'local={total}, total={len(self.samples)}, shard_skip={shard_skip}, '
            f'hit={global_hit}/{len(self.samples)}, miss={global_miss}, '
            f'local_hit={local_hit_plan}, local_build_target={local_build_target}, '
            f'remaining_total={remaining_total}, partition_basis={basis}, '
            f'partition={part_index}/{part_total}, rank={rank}, '
            f'dir={self._resolve_vif_cache_dir()}, cache_device={self.vif_cache_device}, '
            f'cache_device_resolved={resolved_dev}, feature_source={self.vif_feature_source}'
        )

        if total > 0:
            already = built = failed = 0
            start_t = time.time()
            iterator = local_samples
            _use_tqdm = False
            pbar = None
            gpu_mem_mb = 0
            # In non-interactive logging (nohup/file), tqdm carriage-return output
            # is hard to read. Fall back to periodic logger progress.
            if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
                try:
                    from tqdm import tqdm as _tqdm
                    iterator = _tqdm(
                        local_samples,
                        desc=f'Cache {"train" if self.is_train else "eval"} r{rank}',
                        unit='video',
                        dynamic_ncols=True,
                        leave=False,
                    )
                    pbar = iterator
                    _use_tqdm = True
                except Exception:
                    _use_tqdm = False

            for idx, sample in enumerate(iterator):
                try:
                    cached = self._try_load_vif_cached_features(sample)
                    if cached is not None:
                        already += 1
                    else:
                        out = self._build_and_store_vif_cached_features(
                            sample=sample, ref_y=None, dis_y=None,
                        )
                        if out is not None:
                            built += 1
                        else:
                            failed += 1
                except Exception:
                    failed += 1

                done = idx + 1
                elapsed = max(1e-6, time.time() - start_t)
                rate = done / elapsed
                eta_s = int(max(0.0, (total - done) / max(rate, 1e-9)))
                eta_mm = eta_s // 60
                eta_ss = eta_s % 60
                pct = 100.0 * done / max(1, total)
                if torch.cuda.is_available() and self._vif_extractor_device.type == 'cuda':
                    gpu_mem_mb = int(
                        torch.cuda.memory_allocated(self._vif_extractor_device) / (1024 * 1024)
                    )

                if _use_tqdm and pbar is not None:
                    pbar.set_postfix({
                        'hit': global_hit,
                        'already': already,
                        'build': built,
                        'fail': failed,
                        'gpu_mb': gpu_mem_mb,
                        'eta': f'{eta_mm:02d}:{eta_ss:02d}',
                    })
                else:
                    logger.info(
                        f'Prebuild progress ({ "train" if self.is_train else "eval" }): '
                        f'{done}/{total} ({pct:.1f}%), '
                        f'hit={global_hit}, already={already}, build={built}, fail={failed}, '
                        f'rate={rate:.2f} video/s, gpu_mem={gpu_mem_mb}MB, '
                        f'eta={eta_mm:02d}:{eta_ss:02d}, '
                        f'partition={part_index}/{part_total}, rank={rank}, basis={basis}'
                    )

            if _use_tqdm and pbar is not None:
                pbar.close()
            logger.info(
                f'Prebuild done ({ "train" if self.is_train else "eval" }): '
                f'hit={global_hit}, already={already}, build={built}, fail={failed}, local_total={total}, '
                f'partition={part_index}/{part_total}, rank={rank}, basis={basis}'
            )
        if ddp:
            dist.barrier()

    def _try_load_vif_cached_features(self, sample):
        ret = self._vif_cache_path(sample)
        if ret is None:
            return None
        cache_path, expected_key = ret
        if not self.vif_cache_force_rebuild and os.path.exists(cache_path):
            try:
                obj = torch.load(cache_path, map_location='cpu')
                if self._is_valid_cached_obj(obj, expected_key, sample=sample):
                    out = {
                        'vif_precomputed_frame_features': obj['frame_features'].float(),
                        'vif_precomputed_mask': obj['frame_mask'].float(),
                        'vif_precomputed_ti_video_feat': obj['ti_video_feat'].float(),
                    }
                    if 'vmaf_frame_score' in obj and isinstance(obj['vmaf_frame_score'], torch.Tensor):
                        out['vif_precomputed_vmaf_frame_score'] = obj['vmaf_frame_score'].float()
                    else:
                        out['vif_precomputed_vmaf_frame_score'] = torch.full_like(
                            out['vif_precomputed_mask'].float(), float('nan')
                        )
                    return out
            except Exception:
                pass
        if not self.vif_cache_force_rebuild:
            legacy = self._vif_legacy_cache_path(sample)
            if legacy is not None:
                try:
                    obj = torch.load(legacy, map_location='cpu')
                    if self._is_valid_cached_obj(obj, expected_key, sample=sample):
                        out = {
                            'vif_precomputed_frame_features': obj['frame_features'].float(),
                            'vif_precomputed_mask': obj['frame_mask'].float(),
                            'vif_precomputed_ti_video_feat': obj['ti_video_feat'].float(),
                        }
                        if 'vmaf_frame_score' in obj and isinstance(obj['vmaf_frame_score'], torch.Tensor):
                            out['vif_precomputed_vmaf_frame_score'] = obj['vmaf_frame_score'].float()
                        else:
                            out['vif_precomputed_vmaf_frame_score'] = torch.full_like(
                                out['vif_precomputed_mask'].float(), float('nan')
                            )
                        return out
                except Exception:
                    pass
            # Portable fallback: scan same dataset/video prefix files and validate metadata.
            for compat_path in self._vif_prefix_compat_cache_paths(sample):
                if compat_path == cache_path:
                    continue
                try:
                    obj = torch.load(compat_path, map_location='cpu')
                    if self._is_valid_cached_obj(obj, expected_key, sample=sample):
                        out = {
                            'vif_precomputed_frame_features': obj['frame_features'].float(),
                            'vif_precomputed_mask': obj['frame_mask'].float(),
                            'vif_precomputed_ti_video_feat': obj['ti_video_feat'].float(),
                        }
                        if 'vmaf_frame_score' in obj and isinstance(obj['vmaf_frame_score'], torch.Tensor):
                            out['vif_precomputed_vmaf_frame_score'] = obj['vmaf_frame_score'].float()
                        else:
                            out['vif_precomputed_vmaf_frame_score'] = torch.full_like(
                                out['vif_precomputed_mask'].float(), float('nan')
                            )
                        return out
                except Exception:
                    continue
        return None

    def _build_and_store_vif_cached_features(
        self,
        sample,
        ref_y: Optional[torch.Tensor] = None,
        dis_y: Optional[torch.Tensor] = None,
    ):
        ret = self._vif_cache_path(sample)
        if ret is None:
            return None
        cache_path, cache_key = ret
        self._ensure_vif_extractor()
        device = self._vif_extractor_device
        pack = self._extract_libvmaf_pack(sample)
        if pack is None:
            return None

        ff = pack['frame_features']
        fm = pack['frame_mask']
        ti = pack['ti_video_feat']
        dis_w, dis_h, dis_bd, dis_pf = self._dis_spec(sample)
        ref_w, ref_h, ref_bd, ref_pf = self._ref_spec(sample)
        obj = {
            'frame_features': ff.squeeze(0).cpu().half() if ff.dim() == 3 else ff.cpu().half(),
            'frame_mask': fm.squeeze(0).cpu().float() if fm.dim() == 2 else fm.cpu().float(),
            'ti_video_feat': ti.squeeze(0).cpu().float() if ti.dim() == 2 else ti.cpu().float(),
            '_cache_meta': {
                'cache_key': cache_key,
                'dataset_name': sample.dataset_name,
                'video_id': sample.video_id,
                'ref_path': sample.ref_path,
                'dis_path': sample.dis_path,
                'dis_width': dis_w,
                'dis_height': dis_h,
                'dis_bitdepth': dis_bd,
                'dis_pix_fmt': dis_pf,
                'ref_width': ref_w,
                'ref_height': ref_h,
                'ref_bitdepth': ref_bd,
                'ref_pix_fmt': ref_pf,
                'dense_mode': True,
                'max_dense_frames': int(self.vif_max_dense_frames),
                'feature_version': getattr(self, '_vif_feature_version', 'unknown'),
                'cache_key_mode': self.vif_cache_key_mode,
                'cache_device': str(self.vif_cache_device),
                'cache_device_resolved': str(device.type),
                'feature_source': str(self.vif_feature_source),
                'ffmpeg_bin': str(self.vif_ffmpeg_bin),
                'libvmaf_use_cuda': bool(self.vif_libvmaf_use_cuda),
            },
        }
        ext_vmaf = pack.get('vmaf_frame_score', None)
        if isinstance(ext_vmaf, torch.Tensor):
            obj['vmaf_frame_score'] = ext_vmaf.squeeze(0).cpu().float() if ext_vmaf.dim() == 2 else ext_vmaf.cpu().float()

        # ── Compute vmaf_frame_score via SVR if not already present ────
        if 'vmaf_frame_score' not in obj or not isinstance(obj.get('vmaf_frame_score'), torch.Tensor):
            vmaf_scorer = getattr(self, '_vmaf_svr_scorer', None)
            if vmaf_scorer is not None:
                try:
                    w = obj['_cache_meta'].get('ref_width', -1)
                    h = obj['_cache_meta'].get('ref_height', -1)
                    if w > 0 and h > 0:
                        ff_cpu = obj['frame_features'].float()
                        fm_cpu = obj['frame_mask'].bool() if 'frame_mask' in obj else None
                        svr_scores = vmaf_scorer.score(ff_cpu, int(w), int(h), fm_cpu)
                        obj['vmaf_frame_score'] = svr_scores.float()
                except Exception:
                    pass  # graceful fallback: vmaf_frame_score remains absent

        torch.save(obj, cache_path)
        out = {
            'vif_precomputed_frame_features': obj['frame_features'].float(),
            'vif_precomputed_mask': obj['frame_mask'],
            'vif_precomputed_ti_video_feat': obj['ti_video_feat'],
        }
        if 'vmaf_frame_score' in obj and isinstance(obj['vmaf_frame_score'], torch.Tensor):
            out['vif_precomputed_vmaf_frame_score'] = obj['vmaf_frame_score'].float()
        else:
            out['vif_precomputed_vmaf_frame_score'] = torch.full_like(
                out['vif_precomputed_mask'].float(), float('nan')
            )
        return out

    def _build_base_result(self, sample, H: int, W: int, num_clips: int = 1) -> dict:
        # Fine-grained timing: original resolution (before any resize),
        # used by evaluator for resolution-aware FPS stats.
        # 'height'/'width' above may be overwritten with *tensor* shape after
        # resize fallback paths; 'orig_height'/'orig_width' always reflect the
        # annotation-declared raw resolution.
        orig_h = int(getattr(sample, 'height', 0) or H or 0)
        orig_w = int(getattr(sample, 'width', 0) or W or 0)
        result = {
            'mos': torch.tensor(sample.mos, dtype=torch.float32),
            'video_id': sample.video_id,
            'dataset_name': sample.dataset_name,
            'height': torch.tensor(float(H)),
            'width': torch.tensor(float(W)),
            'orig_height': torch.tensor(float(orig_h)),
            'orig_width': torch.tensor(float(orig_w)),
            'num_clips': torch.tensor(int(max(1, num_clips)), dtype=torch.int64),
            'data_valid': torch.tensor(True),
        }
        if sample.stage is not None:
            result['stage'] = sample.stage
        extra = sample.extra or {}
        if 'content_id' in extra:
            result['content_id'] = str(extra['content_id'])
        # Keep this key always present for stable collate; NaN means unavailable.
        vmaf_target = float('nan')
        if 'vmaf_target' in extra:
            try:
                vmaf_target = float(extra['vmaf_target'])
            except Exception:
                vmaf_target = float('nan')
        result['vmaf_target'] = vmaf_target
        return result

    def _gpu_prep_read_frames_capture(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Unified reader wrapper used when ``self.gpu_preprocess_mode > 0``.

        Returns
        -------
        yuv_aligned : torch.Tensor [3, T, H, W] float32  — content depends on mode:
            * safe mode (mode=1): bit-exact to ``self.reader.read_frames(...)``
            * fast mode (mode=2): placeholder; uses a CHEAP nearest-neighbour UV
              upsample instead of the CPU reader's bilinear/bicubic. The numeric
              content is NEVER consumed downstream in fast mode (sem_fn is
              skipped and ``skip_fullres_yuv`` pops dis_yuv/ref_yuv) — we only
              need correct (3, T, H, W) shape + float32 dtype so that the rest
              of ``_finalize_loaded_clips`` runs without shape errors.
              This saves ~15-30 ms per 4K frame (UV bicubic upsample cost).
        raw_half    : Optional tuple ``(Y, U_half, V_half)`` with shapes
            ``(T,H,W)``, ``(T,H/2,W/2)`` and ``(T,H/2,W/2)`` respectively.
            ``None`` when the input path is not a raw YUV file served by the
            ``native`` backend (in that case GPU preprocess falls back to the
            legacy CPU path for this sample).
        """
        # Only raw .yuv files via native backend can cheaply provide raw halves.
        path_ok = False
        try:
            ext = os.path.splitext(str(path))[1].lower()
            path_ok = (ext == '.yuv') and (getattr(self.reader, 'raw_yuv_backend', 'native') == 'native')
        except Exception:
            path_ok = False

        if not path_ok:
            yuv_aligned = self.reader.read_frames(
                path, frame_indices, width, height, bitdepth, pix_fmt,
            )
            return yuv_aligned, None

        # Fetch raw YUV420 planes once, then locally align for CPU-path
        # compatibility.  This avoids a second IO/decode pass.
        Y, U_half, V_half = self.reader.read_frames_raw_yuv420(
            path, frame_indices, width, height, bitdepth, pix_fmt,
        )
        from .yuv.yuv_align import align_yuv
        # In fast mode, dis_yuv numeric content is not consumed (sem_fn skipped,
        # skip_fullres_yuv pops it), so use the cheapest UV upsample method.
        # At 4K this alone saves ~15-30 ms/frame over bilinear/bicubic.
        if self.gpu_preprocess_mode == 2:
            uv_mode = 'nearest'
        else:
            uv_mode = getattr(self.reader, 'uv_upsample', 'bicubic')
        yuv_aligned = align_yuv(Y, U_half, V_half, uv_mode)  # [3, T, H, W]
        return yuv_aligned, (Y.contiguous(), U_half.contiguous(), V_half.contiguous())

    def _inject_timing(
        self,
        result: dict,
        t_all_start: float,
        t_io_end: Optional[float],
        t_all_end: float,
        dis_clips_tensor,
    ) -> None:
        """
        Inject fine-grained per-sample timing into the dataset output dict.

        Fields:
          io_decode_time  - seconds spent on disk IO + YUV decode + sampled-clip
                            cache load (before any preprocessing happens).
          preprocess_time - seconds spent on Resize / GMS patch sampling /
                            fupic / semantic branch preparation, etc.
          sample_total_time - io_decode_time + preprocess_time (excludes
                              CPU→GPU transfer and model forward; those live
                              in evaluator.model_infer_time).
          num_frames_sampled - frames actually sampled per clip (T).  Used by
                              evaluator to compute per-frame FPS.
        """
        try:
            if not isinstance(result, dict):
                return
            if t_io_end is None:
                t_io_end = t_all_end
            io_t = max(0.0, float(t_io_end - t_all_start))
            pre_t = max(0.0, float(t_all_end - t_io_end))
            total_t = max(0.0, float(t_all_end - t_all_start))
            # Derive T (frames per clip) from loaded tensor list.
            nf = 0
            if dis_clips_tensor:
                first = dis_clips_tensor[0] if isinstance(dis_clips_tensor, list) else dis_clips_tensor
                if isinstance(first, torch.Tensor):
                    # Typical shapes: [3, T, H, W] or [T, 3, H, W]
                    if first.dim() == 4:
                        # Heuristic: channel dim is one of {1,3}.
                        if first.shape[0] in (1, 3):
                            nf = int(first.shape[1])
                        else:
                            nf = int(first.shape[0])
            result['io_decode_time'] = torch.tensor(io_t, dtype=torch.float32)
            result['preprocess_time'] = torch.tensor(pre_t, dtype=torch.float32)
            result['sample_total_time'] = torch.tensor(total_t, dtype=torch.float32)
            result['num_frames_sampled'] = torch.tensor(int(nf), dtype=torch.int64)
        except Exception:
            # Timing is purely diagnostic; never fail a batch because of it.
            pass

    def _resolve_cvqm_npy_path(self, sample: SampleMeta, seg_idx: int, is_ref: bool = False) -> Optional[str]:
        """
        Resolve legacy CVQM npy path:
          <npy_root>/<PhaseX|Ref>/<rel_no_ext>/seg_<idx>.npy
        """
        if str(getattr(sample, 'dataset_name', '')).upper() != 'CVQM':
            return None
        try:
            cfg = get_dataset_config('CVQM')
        except Exception:
            return None

        npy_root = str(cfg.get('npy_root', '') or '').strip()
        if not npy_root:
            return None
        phase_roots = cfg.get('phase_roots', {}) if isinstance(cfg.get('phase_roots', {}), dict) else {}
        ref_root = str(cfg.get('ref_root', '') or '').strip()

        if is_ref:
            base = sample.ref_path
            subfolder = 'Ref'
            rel_path = os.path.basename(base or '')
            if base and ref_root and os.path.abspath(base).startswith(os.path.abspath(ref_root)):
                rel_path = os.path.relpath(base, ref_root)
            rel_path_no_ext = os.path.splitext(rel_path)[0]
            # Ref style can be: Video_1920x1080_50fps_10bit.yuv -> Video/seg_*.npy
            m = re.match(r'^(.+?)_\d+x\d+.*$', rel_path_no_ext)
            if m:
                rel_path_no_ext = m.group(1)
        else:
            base = sample.dis_path
            phase = int(sample.stage) if getattr(sample, 'stage', None) in (1, 2) else 1
            subfolder = f'Phase{phase}'
            rel_path = os.path.basename(base or '')
            phase_root = phase_roots.get(phase, '')
            if base and phase_root and os.path.abspath(base).startswith(os.path.abspath(phase_root)):
                rel_path = os.path.relpath(base, phase_root)
            rel_path_no_ext = os.path.splitext(rel_path)[0]

        npy_path = os.path.join(npy_root, subfolder, rel_path_no_ext, f'seg_{int(seg_idx)}.npy')
        if os.path.exists(npy_path):
            return npy_path
        # Fallback: some NAS layouts use Phase1_npy / Phase2_npy / Ref_npy
        alt_subfolder = f'{subfolder}_npy'
        alt_path = os.path.join(npy_root, alt_subfolder, rel_path_no_ext, f'seg_{int(seg_idx)}.npy')
        return alt_path if os.path.exists(alt_path) else None

    def _fit_temporal_len(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Fit first dimension length of [T,...] tensor to target t."""
        cur = int(x.shape[0])
        if cur == t:
            return x
        if cur <= 0:
            return x.new_zeros((t, *x.shape[1:]))
        if cur > t:
            idx = torch.linspace(0, cur - 1, steps=t).round().long()
            return x.index_select(0, idx)
        pad_count = t - cur
        pad = x[-1:].expand(pad_count, *x.shape[1:])
        return torch.cat([x, pad], dim=0)

    def _load_cvqm_npy_clip(
        self,
        sample: SampleMeta,
        seg_idx: int,
        is_ref: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Load one CVQM clip from pre-extracted npy and convert to YUV [3,T,H,W].
        Supports common layouts:
          - [T,H,W,3] RGB
          - [T,3,H,W] RGB
          - [3,T,H,W] RGB
        """
        npy_path = self._resolve_cvqm_npy_path(sample, seg_idx=seg_idx, is_ref=is_ref)
        if npy_path is None:
            return None
        try:
            arr = np.load(npy_path)
        except Exception as e:
            logger.warning("Failed loading CVQM npy %s: %s", npy_path, e)
            return None

        t_target = int(self.frame_sampler.num_frames)
        ten = torch.from_numpy(arr)
        if ten.dim() != 4:
            return None

        # Normalize to [T,3,H,W] RGB float in [0,1]
        if ten.shape[-1] == 3:
            rgb = ten.float().permute(0, 3, 1, 2).contiguous()
        elif ten.shape[1] == 3:
            rgb = ten.float().contiguous()
        elif ten.shape[0] == 3:
            rgb = ten.float().permute(1, 0, 2, 3).contiguous()
        else:
            return None

        if rgb.max().item() > 1.0:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
        rgb = self._fit_temporal_len(rgb, t_target)

        # [T,3,H,W] -> [3,T,H,W] YUV
        if self.cvqm_npy_colorspace == 'bt601':
            yuv_t = rgb_to_yuv_bt601(rgb).permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)
        else:
            yuv_t = rgb_to_yuv_bt709(rgb).permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)
        return yuv_t

    # -- Frame cache loading (raw binary on NVMe) ---------------------------
    _frame_cache_hit_count: int = 0
    _frame_cache_miss_count: int = 0
    _frame_cache_log_interval: int = 200
    _ref_clip_cache_hit_count: int = 0
    _ref_clip_cache_miss_count: int = 0
    _ref_clip_cache_log_interval: int = 100
    _sampled_clip_cache_hit_count: int = 0
    _sampled_clip_cache_miss_count: int = 0
    _sampled_clip_cache_write_count: int = 0
    _sampled_clip_cache_log_interval: int = 50

    def _load_frame_cache_clip(
        self,
        sample: SampleMeta,
        clip_idx: int,
        is_ref: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Load a CVQM clip from the raw binary frame cache on NVMe.

        Returns YUV float32 [3, T, H, W] matching the output of reader.read_frames(),
        or None if the cache file does not exist.

        The cache stores raw uint16 YUV420 data with a 16-byte header.
        Normalization uses the same signal_range/tenbit_mode as the YUV reader
        to ensure bit-exact consistency.
        """
        if self.frame_cache_root is None:
            return None
        ds_name = str(getattr(sample, 'dataset_name', '')).upper()
        if ds_name not in ('CVQM', 'AVT', 'WATERLOO4K', 'BVICC', 'BVIHD'):
            return None

        # Resolve video_id for cache path lookup
        if is_ref:
            if not sample.ref_path:
                return None
            if ds_name in ('AVT', 'WATERLOO4K', 'BVICC', 'BVIHD'):
                # AVT/Waterloo4K ref: container file, use the filename without extension as video_id
                vid = os.path.splitext(os.path.basename(sample.ref_path))[0]
            else:
                vid = os.path.basename(sample.ref_path)
        else:
            vid = sample.video_id

        cache_path = get_cache_clip_path(self.frame_cache_root, vid, clip_idx, is_ref=is_ref)
        if not os.path.isfile(cache_path):
            if not self._frame_cache_miss_logged:
                self._frame_cache_miss_logged = True
                _msg = (f"[FrameCache] ★ First cache MISS: file not found → fallback to live decode\n"
                        f"  cache_root={self.frame_cache_root}\n"
                        f"  expected_path={cache_path}\n"
                        f"  dataset={ds_name} vid={vid} clip={clip_idx} is_ref={is_ref}")
                logger.warning(_msg)
                print(_msg, flush=True)
            self.__class__._frame_cache_miss_count += 1
            return None

        try:
            # Use the SAME reader parameters as VideoReaderFactory._read_raw()
            # to ensure bit-exact consistency between cache and YUV paths.
            sr = getattr(self.reader, 'default_signal_range', 'auto')
            tbm = getattr(self.reader, 'default_tenbit_mode', 'shift8')
            # Match _read_raw: pf = pix_fmt or self.default_pix_fmt
            pf = str(sample.pix_fmt) if sample.pix_fmt is not None else None
            pf = pf or getattr(self.reader, 'default_pix_fmt', 'yuv420p10le')

            Y, U, V = _load_cached_clip_raw(cache_path, signal_range=sr, tenbit_mode=tbm, pix_fmt=pf)

            # align_yuv: UV upsample + stack → [3, T, H, W]
            # Same uv_upsample mode as VideoReaderFactory._read_raw()
            from .yuv.yuv_align import align_yuv
            uv_mode = getattr(self.reader, 'uv_upsample', 'bicubic')
            yuv = align_yuv(Y, U, V, uv_mode)

            self.__class__._frame_cache_hit_count += 1
            total = self.__class__._frame_cache_hit_count + self.__class__._frame_cache_miss_count
            if not self._frame_cache_logged:
                self._frame_cache_logged = True
                _msg = (f"[FrameCache] ★ First cache HIT: reading from .bin cache\n"
                        f"  path={cache_path}\n"
                        f"  dataset={ds_name} vid={vid} clip={clip_idx} is_ref={is_ref}")
                logger.info(_msg)
                print(_msg, flush=True)
            if total % self.__class__._frame_cache_log_interval == 0:
                hit = self.__class__._frame_cache_hit_count
                miss = self.__class__._frame_cache_miss_count
                hit_pct = 100.0 * hit / total if total > 0 else 0
                logger.info(
                    "[FrameCache] After %d loads: hit=%d (%.1f%%), miss=%d",
                    total, hit, hit_pct, miss,
                )
            return yuv

        except Exception as e:
            logger.warning("[FrameCache] Failed loading %s: %s", cache_path, e)
            self.__class__._frame_cache_miss_count += 1
            return None

    def _make_ref_clip_cache_key(
        self,
        path: str,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
    ) -> tuple:
        return (
            os.path.abspath(path),
            tuple(int(i) for i in frame_indices),
            int(width) if width is not None else -1,
            int(height) if height is not None else -1,
            int(bitdepth) if bitdepth is not None else -1,
            str(pix_fmt or ''),
            str(getattr(self.reader, 'container_decoder', 'unknown')),
            str(getattr(self.reader, 'container_yuv_matrix', 'bt709')),
            bool(getattr(self.reader, 'container_yuv_direct', False)),
        )

    def _load_ref_clip_from_memory_cache(
        self,
        sample: SampleMeta,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
    ) -> Optional[torch.Tensor]:
        path = str(getattr(sample, 'ref_path', '') or '').strip()
        if self.ref_clip_cache_size <= 0 or not path or not is_container_video(path):
            return None

        key = self._make_ref_clip_cache_key(path, frame_indices, width, height, bitdepth, pix_fmt)
        cached = self._ref_clip_cache.get(key)
        if cached is None:
            self.__class__._ref_clip_cache_miss_count += 1
            return None

        self._ref_clip_cache.move_to_end(key)
        self.__class__._ref_clip_cache_hit_count += 1
        total = self.__class__._ref_clip_cache_hit_count + self.__class__._ref_clip_cache_miss_count
        if not self._ref_clip_cache_logged:
            self._ref_clip_cache_logged = True
            logger.info(
                "[RefClipCache] First hit: dataset=%s vid=%s ref=%s entries=%d mem=%.1fMB",
                str(getattr(sample, 'dataset_name', '?')),
                str(getattr(sample, 'video_id', '?')),
                path,
                len(self._ref_clip_cache),
                self._ref_clip_cache_bytes / (1024 * 1024),
            )
        if total % self.__class__._ref_clip_cache_log_interval == 0:
            hit = self.__class__._ref_clip_cache_hit_count
            miss = self.__class__._ref_clip_cache_miss_count
            hit_pct = 100.0 * hit / max(1, total)
            logger.info(
                "[RefClipCache] After %d lookups: hit=%d (%.1f%%), miss=%d, entries=%d, mem=%.1fMB",
                total, hit, hit_pct, miss, len(self._ref_clip_cache),
                self._ref_clip_cache_bytes / (1024 * 1024),
            )
        return cached

    def _store_ref_clip_in_memory_cache(
        self,
        sample: SampleMeta,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
        clip: torch.Tensor,
    ) -> None:
        path = str(getattr(sample, 'ref_path', '') or '').strip()
        if self.ref_clip_cache_size <= 0 or not path or not is_container_video(path):
            return
        if clip is None or not isinstance(clip, torch.Tensor):
            return

        clip_bytes = int(clip.numel() * clip.element_size())
        if self.ref_clip_cache_max_bytes > 0 and clip_bytes > self.ref_clip_cache_max_bytes:
            if not self._ref_clip_cache_skip_logged:
                self._ref_clip_cache_skip_logged = True
                logger.info(
                    "[RefClipCache] Skip oversized clip: dataset=%s vid=%s size=%.1fMB limit=%.1fMB ref=%s",
                    str(getattr(sample, 'dataset_name', '?')),
                    str(getattr(sample, 'video_id', '?')),
                    clip_bytes / (1024 * 1024),
                    self.ref_clip_cache_max_bytes / (1024 * 1024),
                    path,
                )
            return

        key = self._make_ref_clip_cache_key(path, frame_indices, width, height, bitdepth, pix_fmt)
        if key in self._ref_clip_cache:
            old = self._ref_clip_cache.pop(key)
            self._ref_clip_cache_bytes -= int(old.numel() * old.element_size())

        while self._ref_clip_cache and (
            len(self._ref_clip_cache) >= self.ref_clip_cache_size or
            (
                self.ref_clip_cache_max_bytes > 0 and
                self._ref_clip_cache_bytes + clip_bytes > self.ref_clip_cache_max_bytes
            )
        ):
            _, old = self._ref_clip_cache.popitem(last=False)
            self._ref_clip_cache_bytes -= int(old.numel() * old.element_size())

        if self.ref_clip_cache_max_bytes > 0 and self._ref_clip_cache_bytes + clip_bytes > self.ref_clip_cache_max_bytes:
            return

        cached = clip.detach().contiguous()
        self._ref_clip_cache[key] = cached
        self._ref_clip_cache_bytes += clip_bytes

    def _sampled_clip_cache_enabled(self, path: Optional[str]) -> bool:
        if self.is_train:
            return False
        if self.sampled_clip_cache_mode == 'off':
            return False
        if not self.sampled_clip_cache_root:
            return False
        return bool(path and is_container_video(path))

    def _build_sampled_clip_cache_key_src(
        self,
        sample: SampleMeta,
        path: str,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
        is_ref: bool,
    ) -> str:
        file_sig = self._portable_file_sig(path)
        if is_ref:
            sample_token = f"refsrc={file_sig}"
        else:
            sample_token = f"video={sample.video_id}|file={file_sig}"
        cache_ver = 'yuv_f32_ref_v2' if is_ref else 'yuv_f32_v1'
        return (
            f"dataset={sample.dataset_name}|is_ref={int(is_ref)}|{sample_token}|"
            f"idx={','.join(str(int(i)) for i in frame_indices)}|"
            f"w={int(width) if width is not None else -1}|"
            f"h={int(height) if height is not None else -1}|"
            f"bd={int(bitdepth) if bitdepth is not None else -1}|"
            f"pf={str(pix_fmt or '')}|"
            f"decoder={str(getattr(self.reader, 'container_decoder', 'unknown'))}|"
            f"matrix={str(getattr(self.reader, 'container_yuv_matrix', 'bt709'))}|"
            f"yuv_direct={bool(getattr(self.reader, 'container_yuv_direct', False))}|"
            f"cache={cache_ver}"
        )

    def _sampled_clip_cache_display_token(
        self,
        sample: SampleMeta,
        path: str,
        is_ref: bool,
    ) -> str:
        if path:
            parent = os.path.basename(os.path.dirname(path))
            stem = os.path.splitext(os.path.basename(path))[0]
            token = f'{parent}__{stem}'
            return self._sanitize_cache_token(token, max_len=112)
        if is_ref and getattr(sample, 'ref_path', None):
            return self._sanitize_cache_token(os.path.basename(str(sample.ref_path)), max_len=112)
        return self._cache_display_token(sample)

    def _sampled_clip_cache_path(
        self,
        sample: SampleMeta,
        path: str,
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
        is_ref: bool,
    ) -> Optional[str]:
        if not self._sampled_clip_cache_enabled(path):
            return None
        key_src = self._build_sampled_clip_cache_key_src(
            sample, path, frame_indices, width, height, bitdepth, pix_fmt, is_ref,
        )
        key = hashlib.sha1(key_src.encode('utf-8')).hexdigest()
        ds = self._sanitize_cache_token(sample.dataset_name, max_len=24)
        kind = 'ref' if is_ref else 'dis'
        prefix = self._sampled_clip_cache_display_token(sample, path, is_ref)
        root = str(self.sampled_clip_cache_root or '').rstrip(os.sep)
        base_name = os.path.basename(root).lower()
        use_flat_layout = (
            base_name.endswith('_frame') or
            (os.path.isdir(os.path.join(root, 'ref')) or os.path.isdir(os.path.join(root, 'dis')))
        ) and not os.path.isdir(os.path.join(root, ds))
        if use_flat_layout:
            return os.path.join(root, kind, f'{prefix}__{key[:16]}.npy')
        return os.path.join(root, ds, kind, f'{prefix}__{key[:16]}.npy')

    def _load_sampled_clip_from_disk_cache(
        self,
        sample: SampleMeta,
        path: Optional[str],
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
        is_ref: bool,
    ) -> Optional[torch.Tensor]:
        cache_path = self._sampled_clip_cache_path(
            sample, str(path or ''), frame_indices, width, height, bitdepth, pix_fmt, is_ref,
        )
        if cache_path is None or not os.path.isfile(cache_path):
            self.__class__._sampled_clip_cache_miss_count += 1
            return None
        if self.sampled_clip_cache_mode not in ('read', 'readwrite'):
            self.__class__._sampled_clip_cache_miss_count += 1
            return None
        try:
            arr = np.load(cache_path, allow_pickle=False)
            if arr.ndim != 4:
                raise ValueError(f'expected 4D tensor, got shape={arr.shape}')
            ten = torch.from_numpy(np.ascontiguousarray(arr)).float()
            self.__class__._sampled_clip_cache_hit_count += 1
            total = (
                self.__class__._sampled_clip_cache_hit_count +
                self.__class__._sampled_clip_cache_miss_count
            )
            if not self._sampled_clip_cache_logged:
                self._sampled_clip_cache_logged = True
                logger.info(
                    "[SampledClipCache] First hit: dataset=%s kind=%s vid=%s path=%s",
                    str(getattr(sample, 'dataset_name', '?')),
                    'ref' if is_ref else 'dis',
                    str(getattr(sample, 'video_id', '?')),
                    cache_path,
                )
            if total % self.__class__._sampled_clip_cache_log_interval == 0:
                hit = self.__class__._sampled_clip_cache_hit_count
                miss = self.__class__._sampled_clip_cache_miss_count
                hit_pct = 100.0 * hit / max(1, total)
                logger.info(
                    "[SampledClipCache] After %d lookups: hit=%d (%.1f%%), miss=%d, writes=%d",
                    total, hit, hit_pct, miss, self.__class__._sampled_clip_cache_write_count,
                )
            return ten
        except Exception as e:
            logger.warning("[SampledClipCache] Failed loading %s: %s", cache_path, e)
            self.__class__._sampled_clip_cache_miss_count += 1
            return None

    def _store_sampled_clip_to_disk_cache(
        self,
        sample: SampleMeta,
        path: Optional[str],
        frame_indices: List[int],
        width: Optional[int],
        height: Optional[int],
        bitdepth: Optional[int],
        pix_fmt: Optional[str],
        is_ref: bool,
        clip: Optional[torch.Tensor],
    ) -> None:
        cache_path = self._sampled_clip_cache_path(
            sample, str(path or ''), frame_indices, width, height, bitdepth, pix_fmt, is_ref,
        )
        if cache_path is None:
            return
        if self.sampled_clip_cache_mode not in ('write', 'readwrite'):
            return
        if clip is None or not isinstance(clip, torch.Tensor):
            return
        if os.path.isfile(cache_path):
            return
        cache_dir = os.path.dirname(cache_path)
        try:
            if not self._sampled_clip_cache_mkdir_done:
                os.makedirs(cache_dir, exist_ok=True)
                self._sampled_clip_cache_mkdir_done = True
            else:
                os.makedirs(cache_dir, exist_ok=True)
            arr = clip.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix='.tmp_sampled_clip_',
                suffix='.npy',
                dir=cache_dir,
            )
            os.close(tmp_fd)
            try:
                with open(tmp_path, 'wb') as f:
                    np.save(f, arr, allow_pickle=False)
                os.replace(tmp_path, cache_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            self.__class__._sampled_clip_cache_write_count += 1
            if not self._sampled_clip_cache_write_logged:
                self._sampled_clip_cache_write_logged = True
                logger.info(
                    "[SampledClipCache] First write: dataset=%s kind=%s vid=%s path=%s size=%.1fMB",
                    str(getattr(sample, 'dataset_name', '?')),
                    'ref' if is_ref else 'dis',
                    str(getattr(sample, 'video_id', '?')),
                    cache_path,
                    arr.nbytes / (1024 * 1024),
                )
        except Exception as e:
            logger.warning("[SampledClipCache] Failed writing %s: %s", cache_path, e)

    def _load_offline_sampled_clip_file(self, cache_path: Optional[str]) -> Optional[torch.Tensor]:
        path = str(cache_path or '').strip()
        if not path:
            return None
        try:
            arr = np.load(path, allow_pickle=False)
            if arr.ndim != 4:
                raise ValueError(f'expected 4D tensor, got shape={arr.shape}')
            return torch.from_numpy(np.ascontiguousarray(arr)).float()
        except Exception as e:
            logger.warning("[OfflineSampledClip] Failed loading %s: %s", path, e)
            return None

    def _load_offline_sampled_clip_lists(
        self, sample: SampleMeta
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        extra = dict(getattr(sample, 'extra', {}) or {})
        if not bool(extra.get('offline_cache_eval', False)):
            return None

        dis_paths = [str(p).strip() for p in list(extra.get('offline_dis_cache_paths', []) or []) if str(p).strip()]
        ref_paths = [str(p).strip() for p in list(extra.get('offline_ref_cache_paths', []) or []) if str(p).strip()]
        if not dis_paths:
            return None

        dis_clips_tensor: List[torch.Tensor] = []
        ref_clips_tensor: List[torch.Tensor] = []
        for p in dis_paths:
            ten = self._load_offline_sampled_clip_file(p)
            if ten is None:
                raise FileNotFoundError(f"offline dis cache missing or invalid: {p}")
            dis_clips_tensor.append(ten)
        for p in ref_paths:
            ten = self._load_offline_sampled_clip_file(p)
            if ten is None:
                raise FileNotFoundError(f"offline ref cache missing or invalid: {p}")
            ref_clips_tensor.append(ten)
        return dis_clips_tensor, ref_clips_tensor

    def _finalize_loaded_clips(
        self,
        sample: SampleMeta,
        dis_clips_tensor: List[torch.Tensor],
        ref_clips_tensor: List[torch.Tensor],
        dis_w: Optional[int],
        dis_h: Optional[int],
        clip_count: int,
        use_vif_dense: bool = False,
        total_ref: int = 0,
        total_dis: int = 0,
        align_mode: str = 'normalized',
        dis_bd: Optional[int] = None,
        dis_pf: Optional[str] = None,
        ref_bd: Optional[int] = None,
        ref_pf: Optional[str] = None,
    ) -> dict:
        if self.cache_only:
            first = dis_clips_tensor[0] if dis_clips_tensor else None
            if isinstance(first, torch.Tensor):
                if first.dim() == 4:
                    H = int(first.shape[2])
                    W = int(first.shape[3])
                elif first.dim() == 5:
                    H = int(first.shape[3])
                    W = int(first.shape[4])
                else:
                    H = dis_h or sample.height or 1080
                    W = dis_w or sample.width or 1920
            else:
                H = dis_h or sample.height or 1080
                W = dis_w or sample.width or 1920
            return self._build_base_result(sample, H, W, num_clips=clip_count)

        dis_yuv = dis_clips_tensor[0] if len(dis_clips_tensor) == 1 else torch.stack(dis_clips_tensor, dim=0)
        ref_yuv = None
        if ref_clips_tensor:
            ref_yuv = ref_clips_tensor[0] if len(ref_clips_tensor) == 1 else torch.stack(ref_clips_tensor, dim=0)

        if self.is_train:
            dis_yuv, ref_yuv = self._apply_augmentation(dis_yuv, ref_yuv)

        # Resolution for meta (height/width): prefer annotation's original resolution
        # over tensor shape.  This is critical when frame cache stores 4K frames
        # pre-resized to 1080p — tensor shape would be 1080p but resolution_token
        # needs the original 4K dimensions to work correctly.
        if sample.height and sample.width:
            H, W = int(sample.height), int(sample.width)
        elif dis_yuv.dim() == 4:
            H, W = dis_yuv.shape[2], dis_yuv.shape[3]
        elif dis_yuv.dim() == 5:
            H, W = dis_yuv.shape[3], dis_yuv.shape[4]
        else:
            H = dis_h or sample.height or 1080
            W = dis_w or sample.width or 1920

        result = self._build_base_result(sample, H, W, num_clips=clip_count)
        result['dis_yuv'] = dis_yuv
        if ref_yuv is not None:
            result['ref_yuv'] = ref_yuv

        if use_vif_dense:
            cached = self._try_load_vif_cached_features(sample)
            if cached is not None:
                result.update(cached)
            else:
                cached = self._build_and_store_vif_cached_features(
                    sample=sample, ref_y=None, dis_y=None,
                )
                if cached is not None:
                    result.update(cached)
                else:
                    if self.vif_only_mode:
                        logger.warning(
                            "VIF cache unavailable for %s/%s, mark sample invalid to skip "
                            "(avoid online dense fallback).",
                            str(sample.dataset_name), str(sample.video_id),
                        )
                        return self._make_dummy(sample)
                    dense_ref_idx, dense_dis_idx = self._sample_dense_aligned_indices(
                        total_ref, total_dis, align_mode=align_mode,
                    )
                    dense_dis = self.reader.read_frames_y(
                        sample.dis_path, dense_dis_idx,
                        dis_w, dis_h, dis_bd, dis_pf,
                    )
                    dense_ref = self.reader.read_frames_y(
                        sample.ref_path, dense_ref_idx,
                        None, None, ref_bd, ref_pf,
                    )
                    result['vif_dis_y_full'] = dense_dis
                    result['vif_ref_y_full'] = dense_ref
                    result['vif_frame_mask'] = torch.ones(
                        result['vif_dis_y_full'].shape[1], dtype=torch.float32
                    )

        if self.enable_detail_branch:
            if self.detail_resize_sampler is not None:
                if dis_yuv.dim() == 4:
                    result.update(self._sample_detail_resize(dis_yuv, ref_yuv))
                elif dis_yuv.dim() == 5:
                    d_list, r_list = [], []
                    for k in range(dis_yuv.shape[0]):
                        sub_ref = ref_yuv[k] if ref_yuv is not None else None
                        out = self._sample_detail_resize(dis_yuv[k], sub_ref)
                        d_list.append(out['gms_dis'])
                        if 'gms_ref' in out:
                            r_list.append(out['gms_ref'])
                    result['gms_dis'] = torch.stack(d_list, dim=0)
                    if r_list:
                        result['gms_ref'] = torch.stack(r_list, dim=0)
            elif self.gms_sampler is not None:
                if dis_yuv.dim() == 4:
                    out = self._sample_gms(dis_yuv, ref_yuv)
                    out = self._reduce_detail_gms(out)
                    result.update(out)
                elif dis_yuv.dim() == 5:
                    gms_dis_list, gms_ref_list = [], []
                    for k in range(dis_yuv.shape[0]):
                        sub_ref = ref_yuv[k] if ref_yuv is not None else None
                        out = self._sample_gms(dis_yuv[k], sub_ref)
                        out = self._reduce_detail_gms(out)
                        gms_dis_list.append(out['gms_dis'])
                        if 'gms_ref' in out:
                            gms_ref_list.append(out['gms_ref'])
                    result['gms_dis'] = torch.stack(gms_dis_list, dim=0)
                    if gms_ref_list:
                        result['gms_ref'] = torch.stack(gms_ref_list, dim=0)
            elif self.fragment_sampler is not None:
                if dis_yuv.dim() == 4:
                    result.update(self._sample_fragment(dis_yuv, ref_yuv))
                elif dis_yuv.dim() == 5:
                    frg_dis_list, frg_ref_list = [], []
                    for k in range(dis_yuv.shape[0]):
                        sub_ref = ref_yuv[k] if ref_yuv is not None else None
                        out = self._sample_fragment(dis_yuv[k], sub_ref)
                        frg_dis_list.append(out['gms_dis'])
                        if 'gms_ref' in out:
                            frg_ref_list.append(out['gms_ref'])
                    result['gms_dis'] = torch.stack(frg_dis_list, dim=0)
                    if frg_ref_list:
                        result['gms_ref'] = torch.stack(frg_ref_list, dim=0)
            elif self.fupic_sampler is not None:
                if dis_yuv.dim() == 4:
                    result.update(self._sample_fupic(dis_yuv, ref_yuv))
                elif dis_yuv.dim() == 5:
                    fupic_dis_list, fupic_ref_list = [], []
                    for k in range(dis_yuv.shape[0]):
                        sub_ref = ref_yuv[k] if ref_yuv is not None else None
                        out = self._sample_fupic(dis_yuv[k], sub_ref)
                        fupic_dis_list.append(out['fupic_dis'])
                        if 'fupic_ref' in out:
                            fupic_ref_list.append(out['fupic_ref'])
                    result['fupic_dis'] = torch.stack(fupic_dis_list, dim=0)
                    if fupic_ref_list:
                        result['fupic_ref'] = torch.stack(fupic_ref_list, dim=0)

        if self.enable_semantic_branch:
            sem_fn = None
            if self.mss_gms_sampler is not None:
                sem_fn = self._sample_semantic_mss
            elif self.resize_sampler is not None:
                sem_fn = self._sample_resize
            elif self.semantic_gms_sampler is not None:
                sem_fn = self._sample_semantic_gmsavg
            elif self.semantic_fragment_sampler is not None:
                sem_fn = self._sample_semantic_fragment

            # ── FAST GPU-preprocess: skip CPU semantic sem_fn when safe ──
            # In fast mode (HMF_VQA_GPU_PREPROCESS=2), we can *entirely* skip
            # the CPU-side _sample_semantic_gmsavg as long as:
            #   1. The semantic branch uses the gmsavg + legacy_pe + stack
            #      layout that the GPU path in evaluator knows how to rebuild.
            #   2. All downstream keys the model needs come from resize_dis
            #      (and optionally resize_ref), which the GPU path produces.
            # We emit zero-valued placeholders with the correct shape so that
            # the evaluator's shape-detection code still fires (6D tensor) and
            # the GPU rebuild overwrites them.  This saves ~6-8 ms per sample
            # of CPU time in the worker.
            # GPU-preprocess fast path: skip the CPU semantic sampler and emit a
            # zero placeholder that the evaluator rebuilds on GPU.
            #   • single-clip (dim==4): evaluator single-clip path rebuilds it.
            #   • multi-clip  (dim==5): evaluator multi-clip loop rebuilds it
            #                           per-clip (see _apply_gpu_preprocess_legacy_pe_clip).
            # Both paths are now supported; the `gpu_prep_placeholder` marker lets
            # the evaluator FAIL-FAST if the rebuild is ever missing (raw planes
            # absent / wrong layout), instead of silently scoring all-zeros.
            _skip_cpu_sem_fn = (
                self.gpu_preprocess_mode == 2
                and sem_fn is self._sample_semantic_gmsavg
                and self.semantic_gms_legacy_pe
                and self.semantic_gms_mode == 'stack'
                and not self.gradient_topk_sampling  # stay safe
                and isinstance(dis_yuv, torch.Tensor)
                and dis_yuv.dim() in (4, 5)          # single- or multi-clip
            )

            if sem_fn is not None and _skip_cpu_sem_fn:
                # Zero placeholder matching the CPU-path shape:
                #   legacy_pe forces tgt = 1080p; stack mode + patches_per_frame=P;
                #   semantic_target_size = ts (defaults to ph when None).
                _ph = int(self.semantic_gms_sampler.patch_size)
                _ts = int(self.semantic_target_size or _ph)
                _P = int(self.semantic_gms_sampler.patches_per_frame)
                if dis_yuv.dim() == 4:
                    # [3, T, H, W] → single clip → resize_dis [P, 3, T, ts, ts]
                    _T = int(dis_yuv.shape[1])
                    result['resize_dis'] = torch.zeros(
                        (_P, 3, _T, _ts, _ts), dtype=dis_yuv.dtype)
                    if ref_yuv is not None:
                        result['resize_ref'] = torch.zeros(
                            (_P, 3, _T, _ts, _ts), dtype=dis_yuv.dtype)
                else:
                    # [K, 3, T, H, W] → multi-clip → resize_dis [K, P, 3, T, ts, ts]
                    _K = int(dis_yuv.shape[0])
                    _T = int(dis_yuv.shape[2])
                    result['resize_dis'] = torch.zeros(
                        (_K, _P, 3, _T, _ts, _ts), dtype=dis_yuv.dtype)
                    if ref_yuv is not None:
                        result['resize_ref'] = torch.zeros(
                            (_K, _P, 3, _T, _ts, _ts), dtype=dis_yuv.dtype)
                result['gpu_prep_placeholder'] = torch.tensor(1, dtype=torch.int64)
            elif sem_fn is not None:
                if dis_yuv.dim() == 4:
                    result.update(sem_fn(dis_yuv, ref_yuv))
                elif dis_yuv.dim() == 5:
                    resize_dis_list, resize_ref_list = [], []
                    mss_gms_dis_list, mss_gms_ref_list = [], []
                    for k in range(dis_yuv.shape[0]):
                        sub_ref = ref_yuv[k] if ref_yuv is not None else None
                        out = sem_fn(dis_yuv[k], sub_ref)
                        resize_dis_list.append(out['resize_dis'])
                        if 'resize_ref' in out:
                            resize_ref_list.append(out['resize_ref'])
                        if 'mss_gms_dis' in out:
                            mss_gms_dis_list.append(out['mss_gms_dis'])
                        if 'mss_gms_ref' in out:
                            mss_gms_ref_list.append(out['mss_gms_ref'])
                    result['resize_dis'] = torch.stack(resize_dis_list, dim=0)
                    if resize_ref_list:
                        result['resize_ref'] = torch.stack(resize_ref_list, dim=0)
                    if mss_gms_dis_list:
                        result['mss_gms_dis'] = torch.stack(mss_gms_dis_list, dim=0)
                    if mss_gms_ref_list:
                        result['mss_gms_ref'] = torch.stack(mss_gms_ref_list, dim=0)

        if self.skip_fullres_yuv:
            result.pop('dis_yuv', None)
            result.pop('ref_yuv', None)

        return result

    def build_sampled_clip_cache_index_rows(self) -> List[dict]:
        if self.is_train:
            return []
        rows: List[dict] = []
        for sample in self.samples:
            try:
                dis_w, dis_h, dis_bd, dis_pf = self._dis_spec(sample)
                ref_w, ref_h, ref_bd, ref_pf = self._ref_spec(sample)
                total_dis = self.reader.get_frame_count(
                    sample.dis_path, dis_w, dis_h, dis_bd
                )
                if total_dis == 0:
                    total_dis = self.frame_sampler.num_frames
                total_ref = total_dis
                align_mode = 'normalized'
                if self.is_fr and sample.ref_path:
                    total_ref = self.reader.get_frame_count(
                        sample.ref_path, ref_w, ref_h, ref_bd
                    )
                    if total_ref == 0:
                        total_ref = total_dis
                    align_mode = self._resolve_fr_align_mode(sample, total_ref, total_dis)
                    ref_clips, dis_clips = self.frame_sampler.sample_aligned(
                        total_ref, total_dis, align_mode=align_mode,
                    )
                else:
                    dis_clips = self.frame_sampler.sample(total_dis)
                    ref_clips = None

                dis_cache_paths = []
                ref_cache_paths = []
                for clip_idx in range(len(dis_clips)):
                    cache_path = self._sampled_clip_cache_path(
                        sample,
                        sample.dis_path,
                        dis_clips[clip_idx],
                        dis_w,
                        dis_h,
                        dis_bd,
                        dis_pf,
                        False,
                    )
                    if cache_path:
                        dis_cache_paths.append(cache_path)
                    if self.is_fr and sample.ref_path and ref_clips is not None:
                        ref_cache_path = self._sampled_clip_cache_path(
                            sample,
                            sample.ref_path,
                            ref_clips[clip_idx],
                            ref_w,
                            ref_h,
                            ref_bd,
                            ref_pf,
                            True,
                        )
                        if ref_cache_path:
                            ref_cache_paths.append(ref_cache_path)

                extra = dict(getattr(sample, 'extra', {}) or {})
                rows.append({
                    'dataset_name': str(sample.dataset_name or ''),
                    'source_split': str(extra.get('source_split', sample.split) or ''),
                    'video_id': str(sample.video_id or ''),
                    'sequence': str(extra.get('sequence', '') or ''),
                    'preset': str(extra.get('preset', '') or ''),
                    'codec': str(extra.get('codec', '') or ''),
                    'crf': str(extra.get('crf', '') or ''),
                    'mos': float(getattr(sample, 'mos', 0.0)),
                    'raw_mos': float(extra.get('raw_mos', float(getattr(sample, 'mos', 0.0)) * 10.0)),
                    'dis_path': str(getattr(sample, 'dis_path', '') or ''),
                    'ref_path': str(getattr(sample, 'ref_path', '') or ''),
                    'dis_bitdepth': int(dis_bd or 8),
                    'dis_pix_fmt': str(dis_pf or ''),
                    'ref_actual_bitdepth': int(extra.get('ref_actual_bitdepth', 8) or 8),
                    'ref_actual_pix_fmt': str(extra.get('ref_actual_pix_fmt', '') or ''),
                    'ref_bitdepth': int(ref_bd or dis_bd or 8),
                    'ref_pix_fmt': str(ref_pf or dis_pf or ''),
                    'standard': str(extra.get('standard', '') or ''),
                    'real_bitrate': str(extra.get('real_bitrate', '') or ''),
                    'bitrate': str(extra.get('bitrate', '') or ''),
                    'cache_height': int(dis_h or sample.height or 0),
                    'cache_width': int(dis_w or sample.width or 0),
                    'num_clips': int(len(dis_cache_paths)),
                    'cache_num_frames': int(self.frame_sampler.num_frames),
                    'cache_sampling_strategy': str(self.frame_sampler.strategy),
                    'cache_sampling_rate': int(self.frame_sampler.sampling_rate),
                    'cache_align_mode': str(align_mode),
                    'dis_cache_paths': json.dumps(dis_cache_paths, ensure_ascii=False),
                    'ref_cache_paths': json.dumps(ref_cache_paths, ensure_ascii=False),
                })
            except Exception as e:
                logger.warning(
                    "[CVQADOfflineIndex] Failed building row for %s: %s",
                    str(getattr(sample, 'video_id', '?')),
                    e,
                )
        return rows

    # -- NPY fallback logging (rate-limited to avoid log flooding) -----------
    _npy_fallback_count: int = 0
    _npy_fallback_log_limit: int = 20  # only log first N fallbacks per worker

    # -- Data source tracking (for H2-style timing experiments) ----------------
    _data_source_npy_count: int = 0
    _data_source_yuv_count: int = 0
    _data_source_cache_count: int = 0
    _data_source_log_interval: int = 100  # log summary every N samples
    _data_source_first_logged: bool = False
    _data_source_datasets_logged: set = set()  # track which dataset+fmt combos we've logged

    def _log_data_source(self, fmt: str, path: str, sample: 'SampleMeta'):
        """Track and log data source (cache vs NPY vs YUV) with rate-limited summaries."""
        if fmt == 'cache':
            self.__class__._data_source_cache_count += 1
        elif fmt == 'npy':
            self.__class__._data_source_npy_count += 1
        else:
            self.__class__._data_source_yuv_count += 1
        total = (self.__class__._data_source_cache_count +
                 self.__class__._data_source_npy_count +
                 self.__class__._data_source_yuv_count)

        # Log the very first sample to confirm the pipeline is working
        if not self.__class__._data_source_first_logged:
            self.__class__._data_source_first_logged = True
            vid = getattr(sample, 'video_id', '?')
            logger.info(
                "[DataSource] First sample: fmt=%s vid=%s path=%s",
                fmt, vid, path,
            )

        # Log first encounter of each dataset+fmt combo (shows 4K vs 1080p, cache vs yuv)
        # Limited to avoid log spam from forked workers (class vars are per-process after fork)
        ds_name = str(getattr(sample, 'dataset_name', '?'))
        _ds_logged_count = getattr(self.__class__, '_data_source_ds_log_count', 0)
        if _ds_logged_count < 6:
            ds_key = f"{ds_name}|{fmt}|{'train' if self.is_train else 'eval'}"
            if ds_key not in self.__class__._data_source_datasets_logged:
                self.__class__._data_source_datasets_logged.add(ds_key)
                self.__class__._data_source_ds_log_count = _ds_logged_count + 1
                h = getattr(sample, 'height', 0) or 0
                w = getattr(sample, 'width', 0) or 0
                res_tag = f"{w}x{h}" if w and h else "?"
                dis_path = getattr(sample, 'dis_path', '?')
                logger.info(
                    "[DataSource] First %s %s sample: fmt=%s res=%s dis=%s",
                    ds_name, 'train' if self.is_train else 'eval',
                    fmt, res_tag, dis_path,
                )

        # Periodic summary
        if total % self.__class__._data_source_log_interval == 0:
            cache_c = self.__class__._data_source_cache_count
            npy_c = self.__class__._data_source_npy_count
            yuv_c = self.__class__._data_source_yuv_count
            cache_pct = 100.0 * cache_c / total if total > 0 else 0
            npy_pct = 100.0 * npy_c / total if total > 0 else 0
            yuv_pct = 100.0 * yuv_c / total if total > 0 else 0
            logger.info(
                "[DataSource] After %d samples: Cache=%d (%.1f%%), NPY=%d (%.1f%%), YUV=%d (%.1f%%)",
                total, cache_c, cache_pct, npy_c, npy_pct, yuv_c, yuv_pct,
            )

    def _log_npy_fallback(self, sample: SampleMeta, seg_idx, is_ref: bool = False):
        """Log a warning when NPY loading fails and falls back to raw YUV."""
        self.__class__._npy_fallback_count += 1
        cnt = self.__class__._npy_fallback_count
        if cnt <= self.__class__._npy_fallback_log_limit:
            kind = "ref" if is_ref else "dis"
            path = sample.ref_path if is_ref else sample.dis_path
            vid = getattr(sample, 'video_id', '?')
            stage = getattr(sample, 'stage', '?')
            npy_path = self._resolve_cvqm_npy_path(sample, seg_idx=seg_idx or 0, is_ref=is_ref)
            logger.warning(
                "[NPY fallback #%d] %s vid=%s stage=%s seg=%s -> reading raw YUV. "
                "NPY path resolved: %s",
                cnt, kind, vid, stage, seg_idx, npy_path,
            )
            if cnt == self.__class__._npy_fallback_log_limit:
                logger.warning(
                    "[NPY fallback] Reached %d fallbacks, suppressing further warnings.",
                    cnt,
                )

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        self._fetch_count += 1

        # ── Fine-grained timing (only active in eval mode via evaluator aggregation) ──
        _t_all_start = time.perf_counter()
        _t_io_end = None  # will be set right before _finalize_loaded_clips

        try:
            offline_loaded = self._load_offline_sampled_clip_lists(sample)
            if offline_loaded is not None:
                if self.vif_mode == 'dense':
                    raise RuntimeError(
                        "offline sampled-clip eval does not support vif_mode=dense; "
                        "please use non-VIF model or provide raw videos"
                    )
                dis_clips_tensor, ref_clips_tensor = offline_loaded
                _t_io_end = time.perf_counter()
                result = self._finalize_loaded_clips(
                    sample,
                    dis_clips_tensor,
                    ref_clips_tensor,
                    sample.width,
                    sample.height,
                    clip_count=len(dis_clips_tensor),
                    use_vif_dense=False,
                )
                _t_all_end = time.perf_counter()
                self._inject_timing(result, _t_all_start, _t_io_end, _t_all_end,
                                    dis_clips_tensor)
                return result

            dis_w, dis_h, dis_bd, dis_pf = self._dis_spec(sample)
            ref_w, ref_h, ref_bd, ref_pf = self._ref_spec(sample)
            # Get frame count
            total_dis = self.reader.get_frame_count(
                sample.dis_path, dis_w, dis_h, dis_bd
            )
            if total_dis == 0:
                total_dis = self.frame_sampler.num_frames

            # Frame sampling → list of K clips, each a list of T frame indices
            align_mode = 'normalized'
            if self.is_fr and sample.ref_path:
                total_ref = self.reader.get_frame_count(
                    sample.ref_path, ref_w, ref_h, ref_bd
                )
                if total_ref == 0:
                    total_ref = total_dis
                align_mode = self._resolve_fr_align_mode(sample, total_ref, total_dis)
                ref_clips, dis_clips = self.frame_sampler.sample_aligned(
                    total_ref, total_dis, align_mode=align_mode,
                )
            else:
                dis_clips = self.frame_sampler.sample(total_dis)
                ref_clips = None

            use_vif_dense = (
                self.is_fr and
                (self.vif_mode == 'dense') and
                (not self.vif_align_with_other_branches) and
                (sample.ref_path is not None)
            )
            if use_vif_dense and self.vif_only_mode:
                cached = self._try_load_vif_cached_features(sample)
                H = dis_h or sample.height or 1080
                W = dis_w or sample.width or 1920
                if cached is not None:
                    result = self._build_base_result(sample, H, W, num_clips=1)
                    result.update(cached)
                    return result
                # Cache miss in VIF-only dense mode: precompute directly from dense frames,
                # avoid reading/alignment for other branches (gms/resize) to reduce overhead.
                cached = self._build_and_store_vif_cached_features(
                    sample=sample, ref_y=None, dis_y=None,
                )
                result = self._build_base_result(sample, H, W, num_clips=1)
                if cached is not None:
                    result.update(cached)
                else:
                    # For VIF-only cache training, avoid expensive online dense fallback.
                    # A single cache-miss sample can trigger full-frame FR compute and OOM.
                    logger.warning(
                        "VIF cache unavailable for %s/%s, mark sample invalid to skip "
                        "(avoid online dense fallback).",
                        str(sample.dataset_name), str(sample.video_id),
                    )
                    return self._make_dummy(sample)
                return result

            # Training: one random clip. Evaluation: all clips (for multi-clip aggregation).
            # Exception: multi_clip_4x8 always uses all clips (for cross-clip temporal interaction).
            is_multi_clip_temporal = (self.frame_sampler.strategy == 'multi_clip_4x8')
            if self.is_train and not is_multi_clip_temporal:
                clip_ids = [random.randint(0, len(dis_clips) - 1)]
            else:
                clip_ids = list(range(len(dis_clips)))

            dis_clips_tensor = []
            ref_clips_tensor = []
            # GPU-preprocess path: capture raw YUV420 planes alongside clips.
            # Populated only when the clip was loaded via the *direct* raw YUV
            # path; other loaders (frame_cache / npy / container) leave the
            # corresponding entry as ``None`` and the GPU path falls back to
            # the legacy CPU-side ``resize_dis``.
            dis_raw_halves: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [] if self.gpu_preprocess_mode > 0 else None
            ref_raw_halves: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [] if self.gpu_preprocess_mode > 0 else None
            cache_only_h = None
            cache_only_w = None
            for clip_idx in clip_ids:
                seg_idx = None
                dis_clip = None
                # Priority 1: Frame cache (raw binary on NVMe, eval only)
                if not self.is_train and self.frame_cache_root:
                    dis_clip = self._load_frame_cache_clip(sample, clip_idx, is_ref=False)
                    if dis_clip is not None:
                        self._log_data_source('cache', 'frame_cache', sample)
                # Priority 2: Legacy CVQM npy (RGB segment format)
                _npy_attempted_dis = False
                if dis_clip is None and self.use_cvqm_npy and str(getattr(sample, 'dataset_name', '')).upper() == 'CVQM':
                    _npy_attempted_dis = True
                    if self.is_train:
                        seg_idx = random.randint(0, self.cvqm_npy_segments - 1)
                    else:
                        seg_idx = int(clip_idx) % self.cvqm_npy_segments
                    dis_clip = self._load_cvqm_npy_clip(sample, seg_idx=seg_idx, is_ref=False)
                # Priority 2.5: sampled clip disk cache for container datasets (e.g. CVQAD)
                if dis_clip is None:
                    dis_clip = self._load_sampled_clip_from_disk_cache(
                        sample,
                        sample.dis_path,
                        dis_clips[clip_idx],
                        dis_w,
                        dis_h,
                        dis_bd,
                        dis_pf,
                        False,
                    )
                # Priority 3: Direct YUV read
                _dis_raw_half: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                if dis_clip is None:
                    if _npy_attempted_dis:
                        self._log_npy_fallback(sample, seg_idx, is_ref=False)
                    if self.gpu_preprocess_mode > 0:
                        dis_clip, _dis_raw_half = self._gpu_prep_read_frames_capture(
                            sample.dis_path, dis_clips[clip_idx],
                            dis_w, dis_h, dis_bd, dis_pf,
                        )
                    else:
                        dis_clip = self.reader.read_frames(
                            sample.dis_path, dis_clips[clip_idx],
                            dis_w, dis_h, dis_bd, dis_pf,
                        )
                    self._store_sampled_clip_to_disk_cache(
                        sample,
                        sample.dis_path,
                        dis_clips[clip_idx],
                        dis_w,
                        dis_h,
                        dis_bd,
                        dis_pf,
                        False,
                        dis_clip,
                    )
                    self._log_data_source('yuv', sample.dis_path, sample)
                elif not (not self.is_train and self.frame_cache_root and dis_clip is not None and not _npy_attempted_dis):
                    # Only log npy source if we actually loaded from npy (not cache)
                    if _npy_attempted_dis and dis_clip is not None:
                        npy_path = self._resolve_cvqm_npy_path(sample, seg_idx=seg_idx, is_ref=False)
                        self._log_data_source('npy', npy_path or '?', sample)
                if self.cache_only:
                    if isinstance(dis_clip, torch.Tensor):
                        if dis_clip.dim() == 4:
                            cache_only_h, cache_only_w = int(dis_clip.shape[2]), int(dis_clip.shape[3])
                        elif dis_clip.dim() == 5:
                            cache_only_h, cache_only_w = int(dis_clip.shape[3]), int(dis_clip.shape[4])
                else:
                    dis_clips_tensor.append(dis_clip)
                    if dis_raw_halves is not None:
                        dis_raw_halves.append(_dis_raw_half)
                if self.is_fr and sample.ref_path and ref_clips is not None:
                    ref_clip = None
                    # Priority 1: Frame cache for ref
                    if not self.is_train and self.frame_cache_root:
                        ref_clip = self._load_frame_cache_clip(sample, clip_idx, is_ref=True)
                    # Priority 2: in-memory exact ref clip cache for repeated container refs
                    if ref_clip is None:
                        ref_clip = self._load_ref_clip_from_memory_cache(
                            sample, ref_clips[clip_idx], ref_w, ref_h, ref_bd, ref_pf,
                        )
                    # Priority 2.5: sampled clip disk cache for container refs
                    if ref_clip is None:
                        ref_clip = self._load_sampled_clip_from_disk_cache(
                            sample,
                            sample.ref_path,
                            ref_clips[clip_idx],
                            ref_w,
                            ref_h,
                            ref_bd,
                            ref_pf,
                            True,
                        )
                        if ref_clip is not None:
                            self._store_ref_clip_in_memory_cache(
                                sample, ref_clips[clip_idx], ref_w, ref_h, ref_bd, ref_pf, ref_clip,
                            )
                    # Priority 3: Legacy CVQM npy for ref
                    _npy_attempted_ref = False
                    if ref_clip is None and self.use_cvqm_npy and str(getattr(sample, 'dataset_name', '')).upper() == 'CVQM':
                        _npy_attempted_ref = True
                        seg_idx_ref = seg_idx
                        if seg_idx_ref is None:
                            seg_idx_ref = int(clip_idx) % self.cvqm_npy_segments
                        ref_clip = self._load_cvqm_npy_clip(sample, seg_idx=seg_idx_ref, is_ref=True)
                    # Priority 4: Direct YUV/container read for ref
                    _ref_raw_half: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                    if ref_clip is None:
                        if _npy_attempted_ref:
                            self._log_npy_fallback(sample, seg_idx_ref, is_ref=True)
                        if self.gpu_preprocess_mode > 0:
                            ref_clip, _ref_raw_half = self._gpu_prep_read_frames_capture(
                                sample.ref_path, ref_clips[clip_idx],
                                ref_w, ref_h, ref_bd, ref_pf,
                            )
                        else:
                            ref_clip = self.reader.read_frames(
                                sample.ref_path, ref_clips[clip_idx],
                                ref_w, ref_h, ref_bd, ref_pf,
                            )
                        self._store_ref_clip_in_memory_cache(
                            sample, ref_clips[clip_idx], ref_w, ref_h, ref_bd, ref_pf, ref_clip,
                        )
                        self._store_sampled_clip_to_disk_cache(
                            sample,
                            sample.ref_path,
                            ref_clips[clip_idx],
                            ref_w,
                            ref_h,
                            ref_bd,
                            ref_pf,
                            True,
                            ref_clip,
                        )
                    if not self.cache_only:
                        ref_clips_tensor.append(ref_clip)
                        if ref_raw_halves is not None:
                            ref_raw_halves.append(_ref_raw_half)

            # ── Mark end of IO+decode phase (before preprocessing) ──
            _t_io_end = time.perf_counter()

            result = self._finalize_loaded_clips(
                sample,
                dis_clips_tensor,
                ref_clips_tensor,
                dis_w,
                dis_h,
                clip_count=len(clip_ids),
                use_vif_dense=use_vif_dense,
                total_ref=total_ref if self.is_fr and sample.ref_path else 0,
                total_dis=total_dis,
                align_mode=align_mode,
                dis_bd=dis_bd,
                dis_pf=dis_pf,
                ref_bd=ref_bd,
                ref_pf=ref_pf,
            )
            # ── GPU preprocess: attach raw YUV420 planes to result ──
            # Only attach when *every* loaded clip captured raw halves (avoid
            # mixing clip-level raw/None, which would break the GPU path).
            if self.gpu_preprocess_mode > 0 and dis_raw_halves is not None and dis_raw_halves and \
                    all(rh is not None for rh in dis_raw_halves):
                try:
                    # Stack per-clip [T,H,W] along dim=0 → [K, T, H, W]
                    Y_stk = torch.stack([rh[0] for rh in dis_raw_halves], dim=0)
                    U_stk = torch.stack([rh[1] for rh in dis_raw_halves], dim=0)
                    V_stk = torch.stack([rh[2] for rh in dis_raw_halves], dim=0)
                    # Single-clip evaluation convention: drop K dim.
                    if Y_stk.shape[0] == 1:
                        Y_stk = Y_stk.squeeze(0)
                        U_stk = U_stk.squeeze(0)
                        V_stk = V_stk.squeeze(0)
                    result['raw_y_dis'] = Y_stk
                    result['raw_u_half_dis'] = U_stk
                    result['raw_v_half_dis'] = V_stk
                    # Tag with the uv_upsample mode used by the CPU reader so
                    # that the evaluator can reproduce the exact same UV
                    # upsample on GPU (bit-exact parity with the CPU path).
                    _uv_mode = getattr(self.reader, 'uv_upsample', 'bilinear')
                    # Encode as int (collate-friendly): 0=bilinear, 1=bicubic, 2=nearest
                    _code = {'bilinear': 0, 'bicubic': 1, 'nearest': 2}.get(
                        str(_uv_mode).lower(), 0)
                    result['uv_upsample_code'] = torch.tensor(int(_code), dtype=torch.int64)
                except Exception as _e:
                    logger.debug("[gpu_preprocess] dis raw-halves pack failed: %s", _e)
            if self.gpu_preprocess_mode > 0 and ref_raw_halves is not None and ref_raw_halves and \
                    all(rh is not None for rh in ref_raw_halves):
                try:
                    Y_stk = torch.stack([rh[0] for rh in ref_raw_halves], dim=0)
                    U_stk = torch.stack([rh[1] for rh in ref_raw_halves], dim=0)
                    V_stk = torch.stack([rh[2] for rh in ref_raw_halves], dim=0)
                    if Y_stk.shape[0] == 1:
                        Y_stk = Y_stk.squeeze(0)
                        U_stk = U_stk.squeeze(0)
                        V_stk = V_stk.squeeze(0)
                    result['raw_y_ref'] = Y_stk
                    result['raw_u_half_ref'] = U_stk
                    result['raw_v_half_ref'] = V_stk
                except Exception as _e:
                    logger.debug("[gpu_preprocess] ref raw-halves pack failed: %s", _e)
            _t_all_end = time.perf_counter()
            self._inject_timing(result, _t_all_start, _t_io_end, _t_all_end,
                                dis_clips_tensor)
            return result

        except Exception as e:
            self._error_count += 1
            if self.data_error_fail_fast:
                err_ratio = float(self._error_count) / max(1, self._fetch_count)
                if self._error_count >= self.data_error_max_count and err_ratio >= self.data_error_max_ratio:
                    raise RuntimeError(
                        "Data loading failures exceeded threshold: "
                        f"errors={self._error_count}, fetched={self._fetch_count}, "
                        f"ratio={err_ratio:.2%}, sample={sample.dis_path}"
                    ) from e
            logger.warning(f"Error loading {sample.dis_path}: {e}")
            return self._make_dummy(sample)

    def _apply_augmentation(self, dis_yuv, ref_yuv):
        """Apply data augmentation to YUV tensors.
        
        Augmentations are applied identically to dis and ref to preserve FR alignment.
        Input shape: [3,T,H,W] (train, single clip) or [K,3,T,H,W] (multi-clip).
        """
        if not (self.aug_hflip or self.aug_tflip or self.aug_brightness > 0):
            return dis_yuv, ref_yuv
        
        is_multi_clip = (dis_yuv.dim() == 5)  # [K, 3, T, H, W]
        # Determine the index for each dimension
        t_dim = 2 if is_multi_clip else 1  # T dimension
        
        # Horizontal flip (50% probability)
        if self.aug_hflip and random.random() > 0.5:
            dis_yuv = dis_yuv.flip(-1)  # flip W dimension
            if ref_yuv is not None:
                ref_yuv = ref_yuv.flip(-1)
        
        # Temporal flip (50% probability) - reverse frame order
        if self.aug_tflip and random.random() > 0.5:
            dis_yuv = dis_yuv.flip(t_dim)  # flip T dimension
            if ref_yuv is not None:
                ref_yuv = ref_yuv.flip(t_dim)
        
        # Y-channel brightness jitter (dis only, doesn't affect ref)
        if self.aug_brightness > 0 and random.random() > 0.5:
            mag = self.aug_brightness
            gain = 1.0 + random.uniform(-mag, mag)
            bias = random.uniform(-mag * 0.4, mag * 0.4)
            dis_yuv = dis_yuv.clone()
            if is_multi_clip:
                dis_yuv[:, 0] = (dis_yuv[:, 0] * gain + bias).clamp(0, 1)
            else:
                dis_yuv[0] = (dis_yuv[0] * gain + bias).clamp(0, 1)
        
        return dis_yuv, ref_yuv

    # ------------------------------------------------------------------
    #  Vectorized preprocess helpers (bit-exact equivalent to legacy loops)
    # ------------------------------------------------------------------
    @staticmethod
    def _crop_stack_patches_vec(
        x_3t: torch.Tensor,
        positions: List[Tuple[int, int]],
        ph: int,
        pw: int,
    ) -> torch.Tensor:
        """Vectorized patch crop + stack.

        Equivalent to (bit-exact) the legacy loop::

            patches = []
            for (y, x) in positions:
                p = x_3t[:, :, y:y+ph, x:x+pw]           # [3, T, h', w']
                if p.shape[2] != ph or p.shape[3] != pw: # boundary overflow
                    p = F.pad(p, (0, pw - p.shape[3], 0, ph - p.shape[2]))
                patches.append(p)
            return torch.stack(patches, dim=0)           # [P, 3, T, ph, pw]

        Fast path: when **all** positions stay within [0, H-ph] × [0, W-pw]
        (the common case; _get_patch_positions / _legacy_grid_positions
        already clamp to valid range), we replace the Python for-loop
        with a single advanced-index gather:

            idx_h[p, i] = positions[p].y + i,  i in [0, ph)
            idx_w[p, j] = positions[p].x + j,  j in [0, pw)
            gathered = x_3t[:, :, idx_h[:, :, None], idx_w[:, None, :]]
                       # → [3, T, P, ph, pw]
            return gathered.permute(2, 0, 1, 3, 4).contiguous()

        This is bit-exact identical to the loop (same index positions,
        same copy semantics), just without Python overhead + without the
        intermediate view-list that ``torch.stack`` would have to copy.

        When any position overflows the frame we fall back to the legacy
        loop verbatim — preserves F.pad's zero-padding behavior.

        Args:
            x_3t: ``[3, T, H, W]`` source tensor.
            positions: list of ``(y, x)`` top-left coords (Python ints).
            ph, pw: patch height/width.

        Returns:
            ``[P, 3, T, ph, pw]`` contiguous tensor.
        """
        P = len(positions)
        if P == 0:
            # Return empty with matching shape — matches torch.stack([]) ValueError
            # behavior by choosing a 0-length dim-0; callers never hit this in
            # practice (GMS always asks for >=1 patch).
            return x_3t.new_empty((0, x_3t.shape[0], x_3t.shape[1], ph, pw))

        _, _T, H, W = x_3t.shape

        # Fast path check: all patches inside frame bounds?
        any_overflow = False
        for (y, x) in positions:
            if y + ph > H or x + pw > W or y < 0 or x < 0:
                any_overflow = True
                break

        if not any_overflow:
            # Build [P, ph] and [P, pw] index grids
            ys = torch.as_tensor([p[0] for p in positions],
                                 dtype=torch.long, device=x_3t.device)
            xs = torch.as_tensor([p[1] for p in positions],
                                 dtype=torch.long, device=x_3t.device)
            dh = torch.arange(ph, dtype=torch.long, device=x_3t.device)
            dw = torch.arange(pw, dtype=torch.long, device=x_3t.device)
            idx_h = ys.unsqueeze(1) + dh.unsqueeze(0)   # [P, ph]
            idx_w = xs.unsqueeze(1) + dw.unsqueeze(0)   # [P, pw]
            # Advanced index: x_3t[:, :, idx_h[:, :, None], idx_w[:, None, :]]
            # Result shape: [3, T, P, ph, pw]
            gathered = x_3t[:, :, idx_h.unsqueeze(-1), idx_w.unsqueeze(-2)]
            # -> [P, 3, T, ph, pw]
            return gathered.permute(2, 0, 1, 3, 4).contiguous()

        # Fallback: preserve exact legacy semantics for boundary overflow
        patches: List[torch.Tensor] = []
        for (y, x) in positions:
            patch = x_3t[:, :, y:y + ph, x:x + pw]
            if patch.shape[2] != ph or patch.shape[3] != pw:
                patch = F.pad(patch, (0, pw - patch.shape[3], 0, ph - patch.shape[2]))
            patches.append(patch)
        return torch.stack(patches, dim=0)

    def _resize_3t_hw_pair(
        self,
        dis_3t: torch.Tensor,
        ref_3t: Optional[torch.Tensor],
        tgt_h: int,
        tgt_w: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Batch-resize both dis and ref ``[3, T, H, W]`` tensors in one
        ``F.interpolate`` call.

        Bit-exact equivalent of calling ``_resize_3t_hw`` twice — because
        ``F.interpolate(mode='bilinear', align_corners=False[, antialias])``
        processes each frame independently, batching along the N-dim does
        not change per-image results.

        When ``ref_3t`` is None, degrades to single-tensor resize.
        When input already matches (tgt_h, tgt_w), the same-size skip
        preserves input identity (no interpolate noise).
        """
        # Fast-path: both already at target, skip interpolate entirely
        dis_skip = (dis_3t.shape[2] == tgt_h and dis_3t.shape[3] == tgt_w)
        if ref_3t is None:
            if dis_skip:
                return dis_3t, None
            return self._resize_3t_hw(dis_3t, tgt_h, tgt_w), None
        ref_skip = (ref_3t.shape[2] == tgt_h and ref_3t.shape[3] == tgt_w)
        if dis_skip and ref_skip:
            return dis_3t, ref_3t
        if dis_skip:
            # Only ref needs resize
            return dis_3t, self._resize_3t_hw(ref_3t, tgt_h, tgt_w)
        if ref_skip:
            return self._resize_3t_hw(dis_3t, tgt_h, tgt_w), ref_3t
        # Both need resize — potentially batch them via torch.cat along N-dim.
        # Must share the same (H, W) source shape to cat.
        if dis_3t.shape[2] != ref_3t.shape[2] or dis_3t.shape[3] != ref_3t.shape[3]:
            return (self._resize_3t_hw(dis_3t, tgt_h, tgt_w),
                    self._resize_3t_hw(ref_3t, tgt_h, tgt_w))

        # ──────────────────────────────────────────────────────────────
        # IMPORTANT: do NOT torch.cat for large inputs.
        #
        # On 4K float32 inputs, dis and ref are each ~760MB ([3,8,2160,3840]),
        # so torch.cat([dis, ref]) allocates a fresh ~1.52GB buffer (a full
        # memcpy of both tensors) before the interpolate even starts. That
        # memcpy dominates anything a single batched interpolate could save
        # over two serial calls — F.interpolate internally parallelizes with
        # OpenMP regardless of N, so batching N=8 vs N=16 yields near-zero
        # speedup on CPU. Net effect: on 4K, "batched" is *slower* than two
        # independent resizes.
        #
        # Heuristic: only batch when a single source tensor fits in ~256MB.
        # That keeps the fast-path useful for 1080p (~190MB, cat = ~380MB,
        # acceptable) while avoiding the 1.5GB disaster at 4K.
        # Bit-exact equivalence is preserved in both branches.
        # ──────────────────────────────────────────────────────────────
        src_numel = dis_3t.numel()  # same for ref (same shape)
        # Assume worst case 4 bytes per element (float32).
        BATCH_BYTES_LIMIT = 256 * 1024 * 1024  # 256 MB per-tensor threshold
        if src_numel * 4 > BATCH_BYTES_LIMIT:
            # Large input (e.g. 4K): two independent resizes, no cat.
            return (self._resize_3t_hw(dis_3t, tgt_h, tgt_w),
                    self._resize_3t_hw(ref_3t, tgt_h, tgt_w))

        # Small input (e.g. 1080p or lower): batch via cat along N.
        T_dis = dis_3t.shape[1]
        dis_frames = dis_3t.permute(1, 0, 2, 3).float()  # [T, 3, H, W]
        ref_frames = ref_3t.permute(1, 0, 2, 3).float()  # [T, 3, H, W]
        both = torch.cat([dis_frames, ref_frames], dim=0)  # [2T, 3, H, W]
        if self.resize_antialias:
            try:
                both_r = F.interpolate(
                    both, size=(tgt_h, tgt_w),
                    mode='bilinear', align_corners=False, antialias=True,
                )
            except TypeError:
                both_r = F.interpolate(
                    both, size=(tgt_h, tgt_w),
                    mode='bilinear', align_corners=False,
                )
        else:
            both_r = F.interpolate(
                both, size=(tgt_h, tgt_w),
                mode='bilinear', align_corners=False,
            )
        dis_r = both_r[:T_dis].permute(1, 0, 2, 3).contiguous()  # [3, T, th, tw]
        ref_r = both_r[T_dis:].permute(1, 0, 2, 3).contiguous()
        src_h = dis_3t.shape[2]
        if src_h >= 1500 and tgt_h < 1500:
            dis_r = self._emulate_offline_uhd_grid(dis_r)
            ref_r = self._emulate_offline_uhd_grid(ref_r)
        return dis_r, ref_r

    def _sample_gms(self, dis_yuv: torch.Tensor,
                    ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Extract GMS patches: same spatial positions across all T frames.
        Returns gms_dis: [P, 3, T, ph, pw], gms_ref (if ref provided).
        """
        _, T, H, W = dis_yuv.shape
        positions = self.gms_sampler._get_patch_positions(H, W)
        ph = pw = self.gms_sampler.patch_size

        result = {'gms_dis': self._crop_stack_patches_vec(dis_yuv, positions, ph, pw)}

        if ref_yuv is not None:
            result['gms_ref'] = self._crop_stack_patches_vec(ref_yuv, positions, ph, pw)

        return result

    def _reduce_detail_gms(self, out: dict) -> dict:
        """Reduce detail GMS patches according to configured mode."""
        mode = self.detail_gms_reduce
        if mode == 'none':
            return out
        dis = out.get('gms_dis')
        if dis is None or dis.dim() != 5 or dis.shape[0] <= 0:
            return out
        if mode == 'avg':
            out['gms_dis'] = dis.mean(dim=0, keepdim=True)
            if 'gms_ref' in out and out['gms_ref'] is not None:
                out['gms_ref'] = out['gms_ref'].mean(dim=0, keepdim=True)
            return out
        if mode == 'random1':
            p = int(dis.shape[0])
            pick = random.randint(0, p - 1) if self.is_train else (p // 2)
            out['gms_dis'] = dis[pick:pick + 1]
            if 'gms_ref' in out and out['gms_ref'] is not None:
                out['gms_ref'] = out['gms_ref'][pick:pick + 1]
            return out
        return out

    def _sample_detail_resize(self, dis_yuv: torch.Tensor,
                              ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Detail branch resize mode: use one resized global patch.
        Outputs gms_dis: [1, 3, T, ts, ts] to stay compatible with detail pipeline.
        """
        ts = int(self.detail_resize_sampler.target_size)
        dis_resized = self._resize_3t(dis_yuv, ts)
        result = {'gms_dis': dis_resized.unsqueeze(0)}
        if ref_yuv is not None:
            ref_resized = self._resize_3t(ref_yuv, ts)
            result['gms_ref'] = ref_resized.unsqueeze(0)
        return result

    def _gradient_topk_select(
        self,
        dis_base: torch.Tensor,
        ref_base: torch.Tensor,
        all_positions: list,
        sampler,
        ph: int, pw: int,
        H: int, W: int,
    ) -> list:
        """
        Gradient-weighted cell selection for GMS patches.
        Computes Sobel gradient magnitude difference between ref and dis per grid cell,
        then selects Top-K cells with highest gradient diff.

        Args:
            dis_base, ref_base: [3, T, H, W] after legacy PE resize
            all_positions: list of (y, x) grid positions (full grid, e.g. 49 for 7x7)
            sampler: GMSSampler with patches_per_frame, grid_size, is_train
            ph, pw: patch height/width (e.g. 224)
            H, W: frame dimensions after resize (e.g. 1080, 1920)

        Returns:
            selected positions (list of (y, x) tuples), length = patches_per_frame
        """
        P = int(sampler.patches_per_frame)
        if len(all_positions) <= P:
            return all_positions

        # Use first frame (Y channel) for gradient computation for efficiency
        # dis_base: [3, T, H, W]
        dis_y = dis_base[0, 0]  # [H, W]
        ref_y = ref_base[0, 0]  # [H, W]

        # Compute Sobel gradient magnitude for ref and dis
        dy = dis_y.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        ry = ref_y.unsqueeze(0).unsqueeze(0).float()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=dy.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=dy.device).view(1, 1, 3, 3)

        # Gradient magnitudes
        d_gx = F.conv2d(dy, sobel_x, padding=1)
        d_gy = F.conv2d(dy, sobel_y, padding=1)
        d_grad = torch.sqrt(d_gx**2 + d_gy**2 + 1e-8).squeeze()  # [H, W]

        r_gx = F.conv2d(ry, sobel_x, padding=1)
        r_gy = F.conv2d(ry, sobel_y, padding=1)
        r_grad = torch.sqrt(r_gx**2 + r_gy**2 + 1e-8).squeeze()  # [H, W]

        # Absolute gradient difference map
        grad_diff = torch.abs(d_grad - r_grad)  # [H, W]

        # Compute mean gradient diff per grid cell
        cell_scores = []
        for idx, (y, x) in enumerate(all_positions):
            y_end = min(y + ph, H)
            x_end = min(x + pw, W)
            cell_region = grad_diff[y:y_end, x:x_end]
            cell_scores.append((idx, float(cell_region.mean().item())))

        if self.gradient_topk_mode == 'topk_uniform':
            # Select Top-K cells by gradient diff, then pick random crop within each
            cell_scores.sort(key=lambda t: t[1], reverse=True)
            selected_indices = [cs[0] for cs in cell_scores[:P]]
        else:
            # 'weighted': probability-weighted random sampling (Top-K with soft selection)
            scores_tensor = torch.tensor([cs[1] for cs in cell_scores], dtype=torch.float32)
            # Temperature-scaled softmax for smoother selection
            probs = torch.softmax(scores_tensor * 5.0, dim=0)
            try:
                chosen = torch.multinomial(probs, P, replacement=False)
            except RuntimeError:
                # Fallback: uniform random
                chosen = torch.randperm(len(cell_scores))[:P]
            selected_indices = chosen.tolist()

        return [all_positions[i] for i in selected_indices]

    def _sample_semantic_gmsavg(self, dis_yuv: torch.Tensor,
                                ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Semantic branch GMS patch mode:
        - semantic_gms_mode='avg': average P patches -> [3,T,ph,pw]
        - semantic_gms_mode='random1': pick one patch (augmentation-style)
        - semantic_gms_mode='stack': keep all P patches -> [P,3,T,ph,pw]

        When sem_adaptive_patches=True, dynamically scales P based on video
        resolution relative to 1080p reference, capping at grid_size².
        """
        _, _T, H, W = dis_yuv.shape
        sampler = self.semantic_gms_sampler
        ph = pw = int(sampler.patch_size)

        # Resolution-adaptive patch count: scale by area ratio vs 1080p
        if self.sem_adaptive_patches and not self.semantic_gms_legacy_pe:
            ref_area = 1920 * 1080  # 1080p reference
            cur_area = max(H * W, 1)
            area_ratio = cur_area / ref_area
            base_patches = sampler.patches_per_frame
            # Scale linearly by area ratio, clamp to [base_patches, grid²]
            max_patches = sampler.grid_size * sampler.grid_size  # 7×7 = 49
            adaptive_patches = min(max_patches, max(base_patches, int(round(base_patches * area_ratio))))
            if adaptive_patches != base_patches:
                # Create temporary sampler with adapted patch count
                _adaptive_sampler = GMSSampler(
                    patch_size=ph,
                    patches_per_frame=adaptive_patches,
                    grid_size=sampler.grid_size,
                    is_train=sampler.is_train,
                )
                positions = _adaptive_sampler._get_patch_positions(H, W)
            else:
                positions = sampler._get_patch_positions(H, W)
            dis_base = dis_yuv
            ref_base = ref_yuv
        elif self.semantic_gms_legacy_pe:
            if W >= H:
                tgt_h, tgt_w = 1080, 1920
            else:
                tgt_h, tgt_w = 1920, 1080
            # Batch dis+ref resize into a single F.interpolate call when both
            # need resizing and share the same source (H,W). Bit-exact.
            dis_base, ref_base = self._resize_3t_hw_pair(
                dis_yuv, ref_yuv, tgt_h=tgt_h, tgt_w=tgt_w
            )
            positions = self._legacy_grid_positions(
                H=tgt_h, W=tgt_w, patch_h=ph, patch_w=pw, grid_size=int(sampler.grid_size)
            )
            # ── Gradient Top-K sampling: select cells with highest ref-dis gradient diff ──
            if self.gradient_topk_sampling and self.is_train and ref_base is not None:
                # Get ALL grid positions (before patches_per_frame truncation)
                all_grid_positions = self._legacy_grid_positions_full(
                    H=tgt_h, W=tgt_w, patch_h=ph, patch_w=pw, grid_size=int(sampler.grid_size)
                )
                positions = self._gradient_topk_select(
                    dis_base, ref_base, all_grid_positions, sampler, ph, pw, tgt_h, tgt_w
                )
        elif self.semantic_gms_adaptive_crop:
            # ── Scheme G: Adaptive Crop-Resize ──
            # Crop size scales with resolution so every patch covers the same
            # fraction of the frame as a 224×224 crop on 1080p.
            # 4K (2160p): crop 448×448 → resize to 224×224
            # 1080p:      crop 224×224 → no resize needed
            # This preserves more high-freq detail than full-frame resize.
            ref_long = 1920  # 1080p reference long edge
            cur_long = float(max(H, W))
            scale_factor = cur_long / ref_long  # 4K: ~2.0, 1080p: ~1.0
            if self.adaptive_crop_max_scale > 0:
                scale_factor = min(scale_factor, self.adaptive_crop_max_scale)
            crop_h = min(int(round(ph * scale_factor)), H)
            crop_w = min(int(round(pw * scale_factor)), W)
            # Grid positions based on crop size on original resolution
            positions = self._legacy_grid_positions(
                H=H, W=W, patch_h=crop_h, patch_w=crop_w,
                grid_size=int(sampler.grid_size),
            )
            # Crop larger patches then resize to standard patch size
            dis_patches = self._crop_and_resize_patches(
                dis_yuv, positions, crop_h, crop_w, ph, pw)
            ref_patches_list = None
            if ref_yuv is not None:
                ref_patches_list = self._crop_and_resize_patches(
                    ref_yuv, positions, crop_h, crop_w, ph, pw)
            # Pack and reduce (same logic as below, but patches already resized)
            dis_pack = torch.stack(dis_patches, dim=0)  # [P, 3, T, ph, pw]
            pick_idx = None
            if self.semantic_gms_mode == 'random1':
                p = int(dis_pack.shape[0])
                pick_idx = random.randint(0, p - 1) if self.is_train else (p // 2)
                dis_mean = dis_pack[pick_idx]
            elif self.semantic_gms_mode == 'stack':
                dis_mean = dis_pack
            else:
                dis_mean = dis_pack.mean(dim=0)
            ts = int(self.semantic_target_size or ph)
            if ph != ts:
                if self.semantic_gms_mode == 'stack':
                    dis_mean = self._resize_pack_3t(dis_mean, ts)
                else:
                    dis_mean = self._resize_3t(dis_mean, ts)
            result = {'resize_dis': dis_mean}
            if ref_patches_list is not None:
                ref_pack = torch.stack(ref_patches_list, dim=0)
                if self.semantic_gms_mode == 'random1':
                    if pick_idx is None:
                        p = int(ref_pack.shape[0])
                        pick_idx = random.randint(0, p - 1) if self.is_train else (p // 2)
                    pick_idx = max(0, min(int(pick_idx), int(ref_pack.shape[0]) - 1))
                    ref_mean = ref_pack[pick_idx]
                elif self.semantic_gms_mode == 'stack':
                    ref_mean = ref_pack
                else:
                    ref_mean = ref_pack.mean(dim=0)
                if ph != ts:
                    if self.semantic_gms_mode == 'stack':
                        ref_mean = self._resize_pack_3t(ref_mean, ts)
                    else:
                        ref_mean = self._resize_3t(ref_mean, ts)
                result['resize_ref'] = ref_mean
            return result
        else:
            dis_base = dis_yuv
            ref_base = ref_yuv
            positions = sampler._get_patch_positions(H, W)

        # Vectorized patch crop (bit-exact equivalent of the legacy for-loop).
        dis_pack = self._crop_stack_patches_vec(dis_base, positions, ph, pw)  # [P,3,T,ph,pw]
        pick_idx = None
        if self.semantic_gms_mode == 'random1':
            p = int(dis_pack.shape[0])
            pick_idx = random.randint(0, p - 1) if self.is_train else (p // 2)
            dis_mean = dis_pack[pick_idx]
        elif self.semantic_gms_mode == 'stack':
            dis_mean = dis_pack
        else:
            dis_mean = dis_pack.mean(dim=0)  # [3,T,ph,pw]
        ts = int(self.semantic_target_size or ph)
        if ph != ts:
            if self.semantic_gms_mode == 'stack':
                dis_mean = self._resize_pack_3t(dis_mean, ts)
            else:
                dis_mean = self._resize_3t(dis_mean, ts)
        result = {'resize_dis': dis_mean}

        if ref_base is not None:
            ref_pack = self._crop_stack_patches_vec(ref_base, positions, ph, pw)
            if self.semantic_gms_mode == 'random1':
                if pick_idx is None:
                    p = int(ref_pack.shape[0])
                    pick_idx = random.randint(0, p - 1) if self.is_train else (p // 2)
                pick_idx = max(0, min(int(pick_idx), int(ref_pack.shape[0]) - 1))
                ref_mean = ref_pack[pick_idx]
            elif self.semantic_gms_mode == 'stack':
                ref_mean = ref_pack
            else:
                ref_mean = ref_pack.mean(dim=0)
            if ph != ts:
                if self.semantic_gms_mode == 'stack':
                    ref_mean = self._resize_pack_3t(ref_mean, ts)
                else:
                    ref_mean = self._resize_3t(ref_mean, ts)
            result['resize_ref'] = ref_mean
        return result

    def _sample_semantic_mss(self, dis_yuv: torch.Tensor,
                              ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Multi-Scale Spatial (MSS) sampling: produce BOTH global resize AND GMS patches.

        Returns:
          resize_dis:   [3, T, ts, ts]     — global resize (captures overall quality)
          mss_gms_dis:  [P, 3, T, ph, pw]  — local GMS patches (captures local artifacts)
          resize_ref / mss_gms_ref:         — same for reference (when FR)
        """
        # 1) Global: resize full frame to target_size (default 224)
        _, T, H, W = dis_yuv.shape
        ts = int(self.semantic_target_size or 224)
        # Batch dis+ref resize when both present (same source shape).
        # Bit-exact equivalent to two independent F.interpolate calls.
        dis_resized_3t, ref_resized_3t = self._resize_3t_hw_pair(
            dis_yuv, ref_yuv, tgt_h=ts, tgt_w=ts
        )
        result = {'resize_dis': dis_resized_3t}  # [3, T, ts, ts]
        if ref_resized_3t is not None:
            result['resize_ref'] = ref_resized_3t

        # 2) Local: GMS patches at 224x224
        sampler = self.mss_gms_sampler
        ph = pw = int(sampler.patch_size)
        positions = sampler._get_patch_positions(H, W)

        # Vectorized patch crop (bit-exact equivalent of the legacy for-loops).
        result['mss_gms_dis'] = self._crop_stack_patches_vec(dis_yuv, positions, ph, pw)
        if ref_yuv is not None:
            result['mss_gms_ref'] = self._crop_stack_patches_vec(ref_yuv, positions, ph, pw)

        return result

    def _legacy_grid_positions_full(
        self,
        H: int, W: int,
        patch_h: int, patch_w: int,
        grid_size: int,
    ) -> List[Tuple[int, int]]:
        """Like _legacy_grid_positions but returns ALL grid²=49 positions (no truncation).
        Used by gradient_topk_select which does its own selection."""
        gs = max(1, int(grid_size))
        gh = max(1, H // gs)
        gw = max(1, W // gs)
        positions: List[Tuple[int, int]] = []
        for r in range(gs):
            for c in range(gs):
                y_s = r * gh
                x_s = c * gw
                # Center crop within each cell (deterministic for gradient computation)
                y_off = max(0, (gh - patch_h) // 2)
                x_off = max(0, (gw - patch_w) // 2)
                y = min(max(y_s + y_off, 0), max(0, H - patch_h))
                x = min(max(x_s + x_off, 0), max(0, W - patch_w))
                positions.append((y, x))
        return positions

    def _legacy_grid_positions(
        self,
        H: int,
        W: int,
        patch_h: int,
        patch_w: int,
        grid_size: int,
    ) -> List[Tuple[int, int]]:
        """PE-style grid sampler used by spatial_sample path in legacy code.

        Generates positions from a grid_size × grid_size grid, then limits
        to patches_per_frame positions (matching GMSSampler._get_patch_positions
        behavior). Without this limit, 7×7=49 patches cause eval OOM when
        test_batch_mul doubles the batch size.
        """
        gs = max(1, int(grid_size))
        gh = max(1, H // gs)
        gw = max(1, W // gs)
        positions: List[Tuple[int, int]] = []
        for r in range(gs):
            for c in range(gs):
                y_s = r * gh
                x_s = c * gw
                if self.is_train:
                    y_off_max = max(0, gh - patch_h)
                    x_off_max = max(0, gw - patch_w)
                    y_off = random.randint(0, y_off_max) if y_off_max > 0 else 0
                    x_off = random.randint(0, x_off_max) if x_off_max > 0 else 0
                else:
                    y_off = max(0, (gh - patch_h) // 2)
                    x_off = max(0, (gw - patch_w) // 2)
                y = min(max(y_s + y_off, 0), max(0, H - patch_h))
                x = min(max(x_s + x_off, 0), max(0, W - patch_w))
                positions.append((y, x))
        # Limit to patches_per_frame to match GMSSampler behavior and prevent
        # eval OOM (gs²=49 >> patches_per_frame=16 with test_batch_mul=2x).
        max_patches = int(self.semantic_gms_sampler.patches_per_frame) if self.semantic_gms_sampler else len(positions)
        if len(positions) > max_patches:
            orig_count = len(positions)
            if self.is_train:
                positions = random.sample(positions, max_patches)
            else:
                # Deterministic: evenly spaced selection
                step = len(positions) / max_patches
                positions = [positions[int(i * step)] for i in range(max_patches)]
            if not hasattr(self, '_legacy_grid_trunc_logged'):
                logger.info(
                    "[legacy_grid] Truncated grid positions: %d -> %d "
                    "(grid=%dx%d, patches_per_frame=%d, is_train=%s)",
                    orig_count, max_patches, gs, gs, max_patches, self.is_train,
                )
                self._legacy_grid_trunc_logged = True
        return positions

    def _resize_pack_3t(self, x_p3t: torch.Tensor, target_size: int) -> torch.Tensor:
        """Resize [P,3,T,H,W] -> [P,3,T,target,target]."""
        p, c, t, h, w = x_p3t.shape
        frames = x_p3t.permute(0, 2, 1, 3, 4).reshape(p * t, c, h, w).float()
        resized = self._resize_frames(frames, target_size)
        return resized.reshape(p, t, c, target_size, target_size).permute(0, 2, 1, 3, 4).contiguous()

    def _sample_semantic_fragment(self, dis_yuv: torch.Tensor,
                                  ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Semantic branch fragment mode:
        build fragment composite then (optionally) resize to semantic target size.
        """
        composites = self.semantic_fragment_sampler.sample_composite(dis_yuv, ref_yuv)
        dis_comp = composites['dis_composite']  # [3,T,ch,cw]
        ts = int(self.semantic_target_size or dis_comp.shape[-1])
        if dis_comp.shape[-1] != ts:
            dis_comp = self._resize_3t(dis_comp, ts)
        result = {'resize_dis': dis_comp}

        if 'ref_composite' in composites:
            ref_comp = composites['ref_composite']
            if ref_comp.shape[-1] != ts:
                ref_comp = self._resize_3t(ref_comp, ts)
            result['resize_ref'] = ref_comp
        return result

    def _resize_3t(self, x_3t: torch.Tensor, target_size: int) -> torch.Tensor:
        """Resize [3,T,H,W] -> [3,T,target,target] using the dataset resize policy."""
        frames = x_3t.permute(1, 0, 2, 3).float()  # [T,3,H,W]
        resized = self._resize_frames(frames, target_size)
        return resized.permute(1, 0, 2, 3).contiguous()

    def _resize_3t_hw(self, x_3t: torch.Tensor, tgt_h: int, tgt_w: int) -> torch.Tensor:
        """Resize [3,T,H,W] -> [3,T,tgt_h,tgt_w]. Skips if already target size."""
        cur_h, cur_w = x_3t.shape[2], x_3t.shape[3]
        if cur_h == tgt_h and cur_w == tgt_w:
            # Already at target resolution — skip interpolation to avoid
            # floating-point noise from bilinear on same-size input.
            # This is critical when reading pre-resized YUV files (e.g.
            # Phase2_Resize1080p): the data is already 1080p so LPE resize
            # should be a no-op, not a precision-degrading identity interpolation.
            return x_3t
        frames = x_3t.permute(1, 0, 2, 3).float()
        if self.resize_antialias:
            try:
                resized = F.interpolate(
                    frames, size=(tgt_h, tgt_w),
                    mode='bilinear', align_corners=False, antialias=True,
                )
            except TypeError:
                resized = F.interpolate(
                    frames, size=(tgt_h, tgt_w),
                    mode='bilinear', align_corners=False,
                )
        else:
            resized = F.interpolate(
                frames, size=(tgt_h, tgt_w),
                mode='bilinear', align_corners=False,
            )
        out = resized.permute(1, 0, 2, 3).contiguous()
        # UHD (>=1500 lines) → sub-UHD downsample must be bit-exact with the
        # offline preprocessing pipeline that produced Phase2_Resize1080p —
        # otherwise Phase-2 raw predictions drift ~0.008 (SRCC delta ~0.0015).
        # See tools/build_train_4k_resize.py for the reference chain.
        if cur_h >= 1500 and tgt_h < 1500:
            out = self._emulate_offline_uhd_grid(out)
        return out

    @staticmethod
    def _emulate_offline_uhd_grid(x_3t: torch.Tensor) -> torch.Tensor:
        """Reproduce the byte-exact output of the offline 4K→1080p YUV writer.

        Emulates the shift8 → resize → reverse_shift8 → write YUV → re-read
        → shift8 chain used by ``tools/build_train_4k_resize.py`` so on-the-fly
        resize matches pre-resized YUV files (e.g. Phase2_Resize1080p) to
        machine precision. Called only when the source was >=1500 lines and
        the target is <1500 lines (i.e. a UHD→sub-UHD downsample).
        """
        _, T, tgt_h, tgt_w = x_3t.shape
        dst_h2, dst_w2 = tgt_h // 2, tgt_w // 2
        y = x_3t[0:1]                                            # [1, T, H, W]
        uv = x_3t[1:3]                                           # [2, T, H, W]
        # Downsample UV to chroma resolution (dst_h/2, dst_w/2)
        uv_small = F.interpolate(
            uv.permute(1, 0, 2, 3),                              # [T, 2, H, W]
            size=(dst_h2, dst_w2), mode='bilinear', align_corners=False,
        )
        # Quantize Y (full res) + UV (chroma res) to the n/255 grid (uint8).
        y_q = torch.clamp((y * 255.0).round(), 0.0, 255.0) / 255.0
        uv_q_small = torch.clamp((uv_small * 255.0).round(), 0.0, 255.0) / 255.0
        # Upsample the quantized UV back to full resolution — matches the
        # yuv_reader align_yuv step performed when reading the offline file.
        uv_q_full = F.interpolate(
            uv_q_small, size=(tgt_h, tgt_w),
            mode='bilinear', align_corners=False,
        ).permute(1, 0, 2, 3).contiguous()                        # [2, T, H, W]
        out = torch.cat([y_q, uv_q_full], dim=0)                  # [3, T, H, W]
        return out.contiguous()

    def _crop_and_resize_patches(
        self,
        yuv: torch.Tensor,
        positions: List[Tuple[int, int]],
        crop_h: int,
        crop_w: int,
        tgt_h: int,
        tgt_w: int,
    ) -> List[torch.Tensor]:
        """Crop patches at (crop_h, crop_w) from ``yuv`` and resize to (tgt_h, tgt_w).

        Used by Scheme G (Adaptive Crop-Resize).

        Args:
            yuv: ``[3, T, H, W]`` source tensor.
            positions: list of ``(y, x)`` top-left positions.
            crop_h, crop_w: crop window size on original resolution.
            tgt_h, tgt_w: target patch size after resize (usually 224×224).

        Returns:
            list of ``[3, T, tgt_h, tgt_w]`` patches.

        Bit-exact equivalent of the legacy per-patch loop:
          - Crop stage: replaced by ``_crop_stack_patches_vec`` (bit-exact).
          - Resize stage: a single ``F.interpolate`` on ``[P*T, 3, ch, cw]``
            is numerically identical to P independent calls on each
            ``[T, 3, ch, cw]`` tensor — ``F.interpolate`` processes each
            image independently, so batching along N changes nothing.
        """
        P = len(positions)
        if P == 0:
            return []

        # 1) Vectorized crop+stack -> [P, 3, T, crop_h, crop_w]. Preserves
        #    F.pad zero-padding for any boundary-overflowing position via
        #    the fallback in _crop_stack_patches_vec.
        packed = self._crop_stack_patches_vec(yuv, positions, crop_h, crop_w)

        need_resize = (crop_h != tgt_h or crop_w != tgt_w)
        if need_resize:
            P_, C_, T_, ch_, cw_ = packed.shape
            # [P, 3, T, ch, cw] -> [P, T, 3, ch, cw] -> [P*T, 3, ch, cw]
            flat = packed.permute(0, 2, 1, 3, 4).reshape(P_ * T_, C_, ch_, cw_).float()
            if self.resize_antialias:
                try:
                    flat = F.interpolate(
                        flat, size=(tgt_h, tgt_w),
                        mode='bilinear', align_corners=False, antialias=True,
                    )
                except TypeError:
                    flat = F.interpolate(
                        flat, size=(tgt_h, tgt_w),
                        mode='bilinear', align_corners=False,
                    )
            else:
                flat = F.interpolate(
                    flat, size=(tgt_h, tgt_w),
                    mode='bilinear', align_corners=False,
                )
            # Back to [P, 3, T, tgt_h, tgt_w]
            packed = flat.reshape(P_, T_, C_, tgt_h, tgt_w).permute(0, 2, 1, 3, 4).contiguous()

        # Unbind to list of [3, T, tgt_h, tgt_w] — matches legacy return type.
        return list(torch.unbind(packed, dim=0))

    def _resize_frames(self, frames: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Resize [T, C, H, W] with ImageNet-style interpolation defaults.
        Uses align_corners=False. Antialias can be disabled for strict legacy alignment.
        """
        if self.resize_antialias:
            try:
                return F.interpolate(
                    frames, size=(target_size, target_size),
                    mode='bilinear', align_corners=False, antialias=True,
                )
            except TypeError:
                return F.interpolate(
                    frames, size=(target_size, target_size),
                    mode='bilinear', align_corners=False,
                )
        else:
            return F.interpolate(
                frames, size=(target_size, target_size),
                mode='bilinear', align_corners=False,
            )

    def _sample_resize(self, dis_yuv: torch.Tensor,
                       ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Resize all frames to fixed size for semantic branch.
        Returns resize_dis: [3, T, ts, ts], resize_ref (if ref provided).
        """
        _, T, H, W = dis_yuv.shape
        ts = self.resize_sampler.target_size

        # [3, T, H, W] → [T, 3, H, W] → resize → [T, 3, ts, ts] → [3, T, ts, ts]
        frames = dis_yuv.permute(1, 0, 2, 3).float()  # [T, 3, H, W]
        resized = self._resize_frames(frames, ts)
        result = {'resize_dis': resized.permute(1, 0, 2, 3)}  # [3, T, ts, ts]

        if ref_yuv is not None:
            ref_frames = ref_yuv.permute(1, 0, 2, 3).float()
            ref_resized = self._resize_frames(ref_frames, ts)
            result['resize_ref'] = ref_resized.permute(1, 0, 2, 3)

        return result

    def _sample_fragment(self, dis_yuv: torch.Tensor,
                         ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Build fragment composite image (FastVQA/DOVER style).
        Outputs gms_dis: [1, 3, T, ch, cw] — P=1, compatible with GMS pipeline.
        """
        composites = self.fragment_sampler.sample_composite(dis_yuv, ref_yuv)
        # composites['dis_composite']: [3, T, ch, cw]
        result = {'gms_dis': composites['dis_composite'].unsqueeze(0)}  # [1, 3, T, ch, cw]
        if 'ref_composite' in composites:
            result['gms_ref'] = composites['ref_composite'].unsqueeze(0)  # [1, 3, T, ch, cw]
        return result

    def _sample_fupic(self, dis_yuv: torch.Tensor,
                      ref_yuv: Optional[torch.Tensor] = None) -> dict:
        """
        Build FuPiC tiles for detail branch inference.
        Outputs fupic_dis: [N, 3, T, th, tw], compatible with detail branch path.
        """
        tiled = self.fupic_sampler.sample(dis_yuv, ref_yuv)
        T = dis_yuv.shape[1]
        n_per_frame = int(tiled['tiles_per_frame'])
        dis_tiles = tiled['dis_tiles']  # [T*N,3,th,tw]
        result = {
            'fupic_dis': dis_tiles.view(T, n_per_frame, *dis_tiles.shape[1:]).permute(1, 2, 0, 3, 4).contiguous(),
        }
        if 'ref_tiles' in tiled:
            ref_tiles = tiled['ref_tiles']
            result['fupic_ref'] = ref_tiles.view(T, n_per_frame, *ref_tiles.shape[1:]).permute(1, 2, 0, 3, 4).contiguous()
        return result

    def _make_dummy(self, sample: SampleMeta) -> dict:
        """Create a dummy sample on error."""
        T = self.frame_sampler.num_frames
        dummy = {
            'dis_yuv': torch.zeros(3, T, 224, 224),
            'mos': torch.tensor(float('nan'), dtype=torch.float32),
            'video_id': sample.video_id,
            'dataset_name': sample.dataset_name,
            'height': torch.tensor(224.0),
            'width': torch.tensor(224.0),
            'num_clips': torch.tensor(1, dtype=torch.int64),
            'data_valid': torch.tensor(False),
            'vmaf_target': float('nan'),
        }
        if self.is_fr:
            dummy['ref_yuv'] = torch.zeros(3, T, 224, 224)
        if self.gms_sampler is not None:
            P = self.gms_sampler.patches_per_frame
            ph = self.gms_sampler.patch_size
            dummy['gms_dis'] = torch.zeros(P, 3, T, ph, ph)
            if self.is_fr:
                dummy['gms_ref'] = torch.zeros(P, 3, T, ph, ph)
        if self.detail_resize_sampler is not None:
            ts = int(self.detail_resize_sampler.target_size)
            dummy['gms_dis'] = torch.zeros(1, 3, T, ts, ts)
            if self.is_fr:
                dummy['gms_ref'] = torch.zeros(1, 3, T, ts, ts)
        if self.fragment_sampler is not None:
            ch = self.fragment_sampler.patch_size  # composite size
            dummy['gms_dis'] = torch.zeros(1, 3, T, ch, ch)
            if self.is_fr:
                dummy['gms_ref'] = torch.zeros(1, 3, T, ch, ch)
        if self.fupic_sampler is not None:
            ts = self.fupic_sampler.tile_size
            dummy['fupic_dis'] = torch.zeros(1, 3, T, ts, ts)
            if self.is_fr:
                dummy['fupic_ref'] = torch.zeros(1, 3, T, ts, ts)
        if self.resize_sampler is not None:
            ts = self.resize_sampler.target_size
            dummy['resize_dis'] = torch.zeros(3, T, ts, ts)
            if self.is_fr:
                dummy['resize_ref'] = torch.zeros(3, T, ts, ts)
        elif self.semantic_gms_sampler is not None or self.semantic_fragment_sampler is not None:
            ts = int(self.semantic_target_size or 224)
            if self.semantic_gms_sampler is not None and self.semantic_gms_mode == 'stack':
                p_sem = int(self.semantic_gms_sampler.patches_per_frame)
                dummy['resize_dis'] = torch.zeros(p_sem, 3, T, ts, ts)
                if self.is_fr:
                    dummy['resize_ref'] = torch.zeros(p_sem, 3, T, ts, ts)
            else:
                dummy['resize_dis'] = torch.zeros(3, T, ts, ts)
                if self.is_fr:
                    dummy['resize_ref'] = torch.zeros(3, T, ts, ts)
        return dummy


class MockDataset(Dataset):
    """Mock dataset for --dry_run mode. Produces all spatial_info tensors."""

    def __init__(self, size: int = 20, num_frames: int = 8, is_fr: bool = True,
                 gms_patches: int = 8, gms_patch_size: int = 256, resize_size: int = 224,
                 detail_sampler: str = 'gms', vif_branch_cfg: Optional[dict] = None,
                 enable_detail_branch: bool = True, enable_semantic_branch: bool = True):
        self.size = size
        self.num_frames = num_frames
        self.is_fr = is_fr
        self.gms_patches = gms_patches
        self.gms_patch_size = gms_patch_size
        self.resize_size = resize_size
        self.detail_sampler = detail_sampler
        self.vif_branch_cfg = vif_branch_cfg or {}
        self.enable_detail_branch = bool(enable_detail_branch)
        self.enable_semantic_branch = bool(enable_semantic_branch)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        H, W = 480, 640
        T = self.num_frames
        P = self.gms_patches
        ph = self.gms_patch_size
        ts = self.resize_size

        result = {
            'dis_yuv': torch.rand(3, T, H, W),
            'mos': torch.rand(1).item(),
            'video_id': f'mock_video_{index}',
            'dataset_name': 'mock',
            'height': torch.tensor(float(H)),
            'width': torch.tensor(float(W)),
            'num_clips': torch.tensor(1, dtype=torch.int64),
            'data_valid': torch.tensor(True),
            'vmaf_target': float('nan'),
        }
        if self.enable_semantic_branch:
            result['resize_dis'] = torch.rand(3, T, ts, ts)
        if self.enable_detail_branch:
            if self.detail_sampler == 'fupic':
                result['fupic_dis'] = torch.rand(P, 3, T, ph, ph)
            else:
                result['gms_dis'] = torch.rand(P, 3, T, ph, ph)
        if self.is_fr:
            result['ref_yuv'] = torch.rand(3, T, H, W)
            if self.enable_detail_branch:
                if self.detail_sampler == 'fupic':
                    result['fupic_ref'] = torch.rand(P, 3, T, ph, ph)
                else:
                    result['gms_ref'] = torch.rand(P, 3, T, ph, ph)
            if self.enable_semantic_branch:
                result['resize_ref'] = torch.rand(3, T, ts, ts)
            mode = str(self.vif_branch_cfg.get('mode', 'aligned')).lower()
            align = bool(self.vif_branch_cfg.get('align_with_other_branches', True))
            if mode == 'dense' and not align:
                Td = min(T * 2, int(self.vif_branch_cfg.get('max_dense_frames', 32)))
                result['vif_dis_y_full'] = torch.rand(1, Td, H, W)
                result['vif_ref_y_full'] = torch.rand(1, Td, H, W)
                result['vif_frame_mask'] = torch.ones(Td)
        return result


def _pad_and_stack(tensors: list) -> torch.Tensor:
    """Pad tensors to the max shape in the batch, then stack."""
    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        for d in range(len(max_shape)):
            max_shape[d] = max(max_shape[d], t.shape[d])
    padded = []
    for t in tensors:
        pad_sizes = []
        # F.pad expects (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        for d in range(len(max_shape) - 1, -1, -1):
            pad_sizes.extend([0, max_shape[d] - t.shape[d]])
        if any(p > 0 for p in pad_sizes):
            t = F.pad(t, pad_sizes)
        padded.append(t)
    return torch.stack(padded)


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate for VQA datasets.
    Handles variable-resolution tensors (dis_yuv, ref_yuv) by padding to max shape in batch.
    Fixed-size tensors (gms_dis, resize_dis) stack normally.
    """
    if not batch:
        return {}

    result = {}
    keys = set()
    for sample in batch:
        keys.update(sample.keys())

    for key in keys:
        vals = [b.get(key, None) for b in batch]
        present = [v is not None for v in vals]
        if not any(present):
            continue
        if not all(present):
            # Tensor keys with mixed presence are dangerous (e.g. partial FR refs).
            first_present = next(v for v in vals if v is not None)
            if isinstance(first_present, torch.Tensor):
                raise RuntimeError(
                    f"Mixed tensor key presence in collate: '{key}' is missing in part of the batch. "
                    f"This usually indicates inconsistent FR/NR sample construction."
                )
            # Non-tensor metadata: keep placeholders for downstream optional logic.
            vals = [v if v is not None else None for v in vals]
            result[key] = vals
            continue

        vals = [v for v in vals if v is not None]
        if isinstance(vals[0], torch.Tensor):
            try:
                result[key] = torch.stack(vals)
            except RuntimeError:
                # Different spatial sizes — pad to max and stack
                result[key] = _pad_and_stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals

    return result


def build_datasets(cfg: dict) -> Tuple[Optional[Dataset], Dict[str, Dataset], Dict[str, Dataset]]:
    """
    Build train, val, and test datasets from config.

    Respects:
      - skip_val: when True, don't build val datasets (train==val for many setups)
      - test_eval_mode: 'fast' uses training-style sampling, 'full' uses test-style

    Returns:
        (train_dataset, val_datasets_dict, test_datasets_dict)
    """
    is_fr = cfg.get('task', 'fr').lower() == 'fr'
    dry_run = cfg.get('dry_run', False)
    num_frames = cfg.get('num_frames', 8)
    skip_val = cfg.get('skip_val', True)
    test_eval_mode = cfg.get('test_eval_mode', 'fast')
    vif_branch_cfg = cfg.get('vif_branch', {})
    output_dir = cfg.get('output_dir', None)
    ref_clip_cache_size = int(max(0, cfg.get('ref_clip_cache_size', 1)))
    ref_clip_cache_max_mb = float(max(0.0, cfg.get('ref_clip_cache_max_mb', 256.0)))
    sampled_clip_cache_mode = str(cfg.get('sampled_clip_cache_mode', 'off') or 'off').lower().strip()
    if sampled_clip_cache_mode not in ('off', 'read', 'write', 'readwrite'):
        logger.warning(
            "Unknown sampled_clip_cache_mode '%s', fallback to off",
            sampled_clip_cache_mode,
        )
        sampled_clip_cache_mode = 'off'
    branches_cfg = cfg.get('branches', '')
    if isinstance(branches_cfg, str):
        b_list = [b.strip().lower() for b in branches_cfg.split(',') if b.strip()]
    else:
        b_list = [str(b).strip().lower() for b in branches_cfg]
    vif_only_mode = (len(b_list) == 1 and b_list[0] == 'vif')
    detail_enabled = ('detail' in b_list)
    semantic_enabled = ('semantic' in b_list)
    pe_align_cfg = cfg.get('pe_align', {})
    if not isinstance(pe_align_cfg, dict):
        pe_align_cfg = dict(pe_align_cfg)
    pe_align_enabled = bool(pe_align_cfg.get('enabled', False))
    # Support swin_align and convnext_align with same data pipeline overrides
    swin_align_cfg = cfg.get('swin_align', {})
    if not isinstance(swin_align_cfg, dict):
        swin_align_cfg = dict(swin_align_cfg)
    convnext_align_cfg = cfg.get('convnext_align', {})
    if not isinstance(convnext_align_cfg, dict):
        convnext_align_cfg = dict(convnext_align_cfg)
    swin_align_enabled = bool(swin_align_cfg.get('enabled', False))
    convnext_align_enabled = bool(convnext_align_cfg.get('enabled', False))
    any_align_enabled = pe_align_enabled or swin_align_enabled or convnext_align_enabled
    if pe_align_enabled:
        active_align_cfg = pe_align_cfg
    elif swin_align_enabled:
        active_align_cfg = swin_align_cfg
    elif convnext_align_enabled:
        active_align_cfg = convnext_align_cfg
    else:
        active_align_cfg = {}

    # Align modes use the semantic GMS pipeline (resize_dis), regardless of
    # which branch name the YAML declares.  Force semantic_enabled so the
    # semantic spatial sampler is built and resize_dis is produced.
    if swin_align_enabled or convnext_align_enabled:
        semantic_enabled = True
        # These aligned models don't need the detail-branch gms_dis pipeline.
        detail_enabled = False
    temporal_sampler = str(cfg.get('temporal_sampler', 'multiclip')).lower().strip()
    sampling_rate = int(cfg.get('sampling_rate', 1))
    legacy_temporal_window_sampling = False
    if any_align_enabled:
        temporal_sampler = str(active_align_cfg.get('temporal_sampler', temporal_sampler)).lower().strip()
        sampling_rate = int(active_align_cfg.get('sampling_rate', sampling_rate))
        legacy_temporal_window_sampling = bool(
            active_align_cfg.get('legacy_temporal_window_sampling', legacy_temporal_window_sampling)
        )
    else:
        # semantic_branch fallback for legacy temporal window sampling
        sem_cfg = cfg.get('semantic_branch', {})
        if isinstance(sem_cfg, dict) and bool(sem_cfg.get('legacy_temporal_window_sampling', False)):
            legacy_temporal_window_sampling = True
    if temporal_sampler not in ('multiclip', 'uniform', 'uniform_clip', 'centered', 'random_stride', 'legacy_pe',
                                'consecutive_random', 'burst_4x2', 'segment_2x4', 'mixed_sampler',
                                'multi_clip_4x8'):
        logger.warning("Unknown temporal_sampler '%s', fallback to multiclip.", temporal_sampler)
        temporal_sampler = 'multiclip'
    use_cvqm_npy = bool(cfg.get('use_cvqm_npy', False))
    if any_align_enabled:
        use_cvqm_npy = bool(active_align_cfg.get('use_cvqm_npy', use_cvqm_npy))
    cvqm_npy_segments = int(cfg.get('cvqm_npy_segments', 4))
    # Frame cache: raw binary cache on NVMe/local-SSD/NAS for CVQM evaluation acceleration.
    # Resolution order (same as all other dataset paths via _env_path):
    #   1) HMF_VQA_USE_CACHE=0 → force disable (global kill switch)
    #   2) --frame_cache_root CLI arg / yaml key  (explicit override, highest priority)
    #   3) Auto-select by temporal_sampler:
    #       uniform/uniform_clip → HMF_VQA_FRAME_CACHE_ROOT_UNIFORM
    #       legacy_pe/multiclip  → HMF_VQA_FRAME_CACHE_ROOT (default)
    #   4) HMF_VQA_FRAME_CACHE_ROOT env var or PATH_PROFILES  (profile-aware fallback)
    # This ensures training and eval use matching cache automatically.
    from src.data.datasets.base_dataset import _env_path as _base_env_path  # noqa
    frame_cache_root = cfg.get('frame_cache_root', None)
    if frame_cache_root:
        frame_cache_root = str(frame_cache_root).strip() or None
    # Global kill switch: HMF_VQA_USE_CACHE=0 disables frame cache regardless of config
    if os.environ.get('HMF_VQA_USE_CACHE', '1').strip() == '0':
        if frame_cache_root:
            logger.info("[FrameCache] Disabled by HMF_VQA_USE_CACHE=0 (was: %s)", frame_cache_root)
        frame_cache_root = None
    elif frame_cache_root and os.path.isdir(frame_cache_root):
        # CLI / YAML explicit path exists on this machine — use it directly
        logger.info("[FrameCache] Enabled (explicit). cache_root=%s", frame_cache_root)
    else:
        # Auto-select cache based on temporal_sampler strategy
        is_uniform_sampler = temporal_sampler in ('uniform', 'uniform_clip')
        if is_uniform_sampler:
            # Try uniform-specific cache first
            profile_cache = _base_env_path('HMF_VQA_FRAME_CACHE_ROOT_UNIFORM', '')
            cache_label = 'uniform'
        else:
            profile_cache = ''
            cache_label = 'legacy_pe'

        if profile_cache and os.path.isdir(profile_cache):
            logger.info("[FrameCache] Auto-selected %s cache: %s (temporal_sampler=%s)",
                        cache_label, profile_cache, temporal_sampler)
            frame_cache_root = profile_cache
        else:
            # Fallback to default frame cache
            fallback_cache = _base_env_path('HMF_VQA_FRAME_CACHE_ROOT', '')
            if fallback_cache and os.path.isdir(fallback_cache):
                if frame_cache_root and frame_cache_root != fallback_cache:
                    logger.info("[FrameCache] YAML path %s not on this machine, "
                                "using profile path: %s", frame_cache_root, fallback_cache)
                else:
                    logger.info("[FrameCache] Enabled (profile). cache_root=%s", fallback_cache)
                frame_cache_root = fallback_cache
            elif frame_cache_root:
                logger.warning("[FrameCache] frame_cache_root not found: %s (disabled)", frame_cache_root)
                frame_cache_root = None
            # else: no cache configured at all — frame_cache_root stays None

    # ── multi_clip_4x8 temporal training: disable frame cache, switch CVQM to Resize1080p YUV ──
    # The frame cache only stores clip_0 (generated by single-clip mode); multi_clip_4x8 needs 4 clips,
    # so clip_1/2/3 will cache-miss and fall back to raw YUV reads. If Phase2 is a 4K video,
    # clip_0 is loaded from cache at 1080p and clip_1/2/3 are loaded from NAS at 4K, and torch.stack
    # will error out due to the resolution mismatch. Therefore in multi_clip_4x8 mode:
    #   1) Disable the frame cache; everything goes through direct YUV read
    #   2) Switch the CVQM Phase2 paths to the Resize1080p version (dis + ref both 1080p)
    #   3) Phase1 is already 1080p, no change needed
    _is_multi_clip_temporal = (temporal_sampler == 'multi_clip_4x8')
    _cvqm_phase2_resize_override = None  # Used to override CVQM phase_roots later
    _cvqm_ref_resize_override = None     # Used to override CVQM ref_root later
    if _is_multi_clip_temporal:
        if frame_cache_root:
            logger.info(
                "[MultiClipTemporal] Disabling frame cache for multi_clip_4x8 mode "
                "(cache only has clip_0, need 4 clips). Was: %s", frame_cache_root,
            )
            frame_cache_root = None
        # Switch CVQM Phase2 to the Resize1080p path
        _phase2_resize = _base_env_path('HMF_VQA_CVQM_PHASE2_RESIZE_ROOT', '')
        _ref_resize = _base_env_path('HMF_VQA_CVQM_REF_RESIZE_ROOT', '')
        if _phase2_resize and os.path.isdir(_phase2_resize):
            _cvqm_phase2_resize_override = _phase2_resize
            logger.info(
                "[MultiClipTemporal] CVQM Phase2 → Resize1080p: %s", _phase2_resize,
            )
        else:
            logger.warning(
                "[MultiClipTemporal] CVQM Phase2 Resize1080p path not found: %s. "
                "Phase2 4K videos may cause resolution mismatch errors!",
                _phase2_resize or '(not configured)',
            )
        if _ref_resize and os.path.isdir(_ref_resize):
            _cvqm_ref_resize_override = _ref_resize
            logger.info(
                "[MultiClipTemporal] CVQM REF → Resize1080p: %s", _ref_resize,
            )
        else:
            logger.warning(
                "[MultiClipTemporal] CVQM REF Resize1080p path not found: %s. "
                "REF 4K videos may cause resolution mismatch errors!",
                _ref_resize or '(not configured)',
            )

    requested_test_name = str(cfg.get('test_dataset', '') or '').strip()
    sampled_clip_cache_root = cfg.get('sampled_clip_cache_root', None)
    if sampled_clip_cache_root:
        sampled_clip_cache_root = str(sampled_clip_cache_root).strip() or None
    elif requested_test_name in ('CVQAD', 'CVQAC'):
        cvqad_cache_base = _base_env_path('HMF_VQA_CVQAD_CACHE_ROOT', 'CVQAD')
        if cvqad_cache_base:
            sampled_clip_cache_root = resolve_sampled_clip_cache_root(cvqad_cache_base, 'CVQAD')
    elif frame_cache_root:
        sampled_clip_cache_root = os.path.join(frame_cache_root, 'sampled_clip_cache_v1')
    elif output_dir:
        sampled_clip_cache_root = os.path.join(str(output_dir), 'sampled_clip_cache')
    cvqm_npy_colorspace = 'bt709'
    if any_align_enabled:
        align_cs = str(active_align_cfg.get('colorspace_mode', 'bt709_imagenet')).lower().strip()
        if align_cs.startswith('bt601'):
            cvqm_npy_colorspace = 'bt601'
    else:
        # Infer NPY colorspace from top-level colorspace config
        top_cs = str(cfg.get('colorspace', 'bt709_imagenet')).lower().strip()
        if top_cs.startswith('bt601'):
            cvqm_npy_colorspace = 'bt601'
    signal_range = str(cfg.get('signal_range', 'auto'))
    tenbit_mode = str(cfg.get('tenbit_mode', 'normalize'))
    resize_antialias = bool(cfg.get('resize_antialias', True))
    raw_yuv_backend = 'native'
    raw_yuv_matrix = 'bt709'
    container_yuv_matrix = 'bt709'
    container_yuv_direct = True  # Default: ffmpeg outputs YUV directly (no RGB roundtrip)
    if any_align_enabled:
        signal_range = str(active_align_cfg.get('force_signal_range', signal_range))
        tenbit_mode = str(active_align_cfg.get('tenbit_mode', tenbit_mode))
        resize_antialias = bool(active_align_cfg.get('resize_antialias', resize_antialias))
        raw_yuv_backend = str(active_align_cfg.get('raw_yuv_backend', raw_yuv_backend))
        raw_yuv_matrix = str(active_align_cfg.get('raw_yuv_matrix', raw_yuv_matrix))
        container_yuv_matrix = str(active_align_cfg.get('container_yuv_matrix', container_yuv_matrix))
        container_yuv_direct = bool(active_align_cfg.get('container_yuv_direct', container_yuv_direct))
    # Also check all align groups for container_yuv_direct even if not "enabled",
    # because the CLI --pe_align_container_yuv_direct writes into pe_align dict
    # regardless of pe_align.enabled status.
    if not container_yuv_direct:
        for align_key in ('pe_align', 'swin_align', 'convnext_align'):
            align_sub = cfg.get(align_key, {})
            if isinstance(align_sub, str):
                import ast as _ast
                align_sub = _ast.literal_eval(align_sub)
            if bool(align_sub.get('container_yuv_direct', False)):
                container_yuv_direct = True
                break

    path_info = get_dataset_path_runtime_info()
    dataset_path_fail_fast = bool(cfg.get('dataset_path_fail_fast', True))
    fr_strict_reference = bool(cfg.get('fr_strict_reference', True))
    logger.info(
        "Dataset path profile: profile=%s (source=%s), data_root=%s, gpu=%s",
        path_info.get('profile', 'unknown'),
        path_info.get('profile_source', 'unknown'),
        path_info.get('data_root', 'unknown'),
        path_info.get('gpu_names', []),
    )
    logger.info(
        "Dataset strict guards: layout_fail_fast=%s, fr_strict_reference=%s",
        str(dataset_path_fail_fast), str(fr_strict_reference),
    )
    logger.info(
        "Temporal sampler: strategy=%s, sampling_rate=%d, num_frames=%d, K_train=%d, K_test=%d, use_cvqm_npy=%s",
        temporal_sampler,
        sampling_rate,
        int(num_frames),
        int(cfg.get('num_clips_train', 2)),
        int(cfg.get('num_clips_test', 4)),
        str(use_cvqm_npy),
    )
    logger.info(
        "PE align reader/sampler switches: legacy_temporal_window_sampling=%s, raw_yuv_backend=%s, raw_yuv_matrix=%s, container_yuv_matrix=%s, container_yuv_direct=%s",
        str(legacy_temporal_window_sampling),
        str(raw_yuv_backend),
        str(raw_yuv_matrix),
        str(container_yuv_matrix),
        str(container_yuv_direct),
    )
    logger.info("CVQM npy colorspace proxy: %s", cvqm_npy_colorspace)
    logger.info(
        "[DataPipeline] colorspace=%s, tenbit_mode=%s, signal_range=%s, "
        "uv_upsample=%s, resize_antialias=%s, use_cvqm_npy=%s, cvqm_phase=%s",
        cfg.get('colorspace', '?'), cfg.get('tenbit_mode', '?'),
        cfg.get('signal_range', '?'), cfg.get('uv_upsample', '?'),
        cfg.get('resize_antialias', '?'), str(use_cvqm_npy),
        cfg.get('cvqm_phase', '?'),
    )
    if use_cvqm_npy:
        npy_root = path_info.get('npy_root', 'unknown')
        logger.info("[DataPipeline] NPY mode enabled. npy_root=%s", npy_root)
    else:
        logger.info("[DataPipeline] Raw YUV mode. Reading directly from video files.")

    # ── Consolidated data-read-mode summary (appears early in every run log) ──
    _cache_status = "OFF"
    _cache_detail = "YUV direct read (original path)"
    _decode_method = "numpy"
    if frame_cache_root:
        _cache_status = "ON"
        _cache_detail = frame_cache_root
        from .cache.frame_cache import _TORCH_DECODE_ENABLED, TORCH_DECODE_THREADS
        if _TORCH_DECODE_ENABLED:
            _decode_method = f"torch (threads={TORCH_DECODE_THREADS})"
        else:
            _decode_method = "numpy (HMF_VQA_TORCH_DECODE=0)"
    else:
        _decode_method = "numpy (cache OFF, YUV direct read)"
    logger.info(
        "┌─────────────────────────────────────────────────────────────┐"
    )
    logger.info(
        "│ [DataReadMode] cache=%s  decode=%s", _cache_status, _decode_method,
    )
    logger.info(
        "│ [DataReadMode] cache_root=%s", _cache_detail,
    )
    logger.info(
        "│ [DataReadMode] profile=%s  task=%s  tenbit=%s  uv=%s",
        path_info.get('profile', '?'), cfg.get('task', '?'),
        cfg.get('tenbit_mode', '?'), cfg.get('uv_upsample', '?'),
    )
    logger.info(
        "│ [DataReadMode] temporal_sampler=%s  target_label=%s",
        temporal_sampler, cfg.get('target_label_key', 'mos'),
    )
    # Log resolved training dataset paths
    _train_ds = str(cfg.get('train_datasets', '') or '').strip()
    logger.info(
        "│ [TrainData] datasets=%s", _train_ds,
    )
    for _ds_name in _train_ds.split(','):
        _ds_name = _ds_name.strip()
        if _ds_name:
            try:
                _ds_cfg = get_dataset_config(_ds_name)
                logger.info(
                    "│ [TrainData]   %s: root=%s", _ds_name, _ds_cfg.get('root', '?'),
                )
                _ref_root = _ds_cfg.get('ref_root', '')
                if _ref_root:
                    logger.info(
                        "│ [TrainData]   %s: ref_root=%s", _ds_name, _ref_root,
                    )
            except Exception:
                pass
    logger.info(
        "│ [DataReadMode] toggle: HMF_VQA_USE_CACHE=0 disables cache / HMF_VQA_TORCH_DECODE=0 disables torch",
    )
    logger.info(
        "└─────────────────────────────────────────────────────────────┘"
    )
    logger.info(
        "[RefClipCache] size=%d, max_entry_mb=%.1f (container refs only, exact aligned indices)",
        ref_clip_cache_size, ref_clip_cache_max_mb,
    )
    logger.info(
        "[SampledClipCache] mode=%s, root=%s",
        sampled_clip_cache_mode,
        sampled_clip_cache_root or 'OFF',
    )

    def _validate_dataset_layout_or_raise(dataset_name: str, split: str):
        if cfg.get('dry_run', False):
            return
        if not dataset_path_fail_fast:
            return
        validate_dataset_layout(dataset_name, split=split, require_ref=bool(is_fr))

    def _filter_fr_samples(samples: List[SampleMeta], dataset_name: str, split: str) -> List[SampleMeta]:
        """Validate FR samples to avoid silent FR->NR degradation."""
        if not is_fr:
            return samples
        kept = []
        invalid = []
        for s in samples:
            ref_path = str(getattr(s, 'ref_path', '') or '').strip()
            dis_path = str(getattr(s, 'dis_path', '') or '').strip()
            reason = None
            if not ref_path:
                reason = 'missing_ref_path'
            elif not os.path.exists(ref_path):
                reason = 'missing_ref_file'
            elif not dis_path:
                reason = 'missing_dis_path'
            elif not os.path.exists(dis_path):
                reason = 'missing_dis_file'

            if reason is not None:
                invalid.append((reason, s))
                continue
            kept.append(s)

        if invalid:
            counts: Dict[str, int] = {}
            for reason, _sample in invalid:
                counts[reason] = counts.get(reason, 0) + 1
            examples = []
            for reason, sample in invalid[:5]:
                examples.append(
                    f"video_id={sample.video_id}, reason={reason}, "
                    f"ref={sample.ref_path}, dis={sample.dis_path}"
                )
            msg_lines = [
                f"FR sample validation failed [{dataset_name}:{split}] with {len(invalid)} invalid samples.",
                f"Active profile: {path_info.get('profile', '?')} (source={path_info.get('profile_source', '?')})",
                "Counts: " + ', '.join(f"{k}={v}" for k, v in sorted(counts.items())),
                "Examples:",
            ]
            msg_lines.extend(f"  - {line}" for line in examples)
            msg_lines.append(
                "This usually means dataset roots or reference roots on the current machine are not mounted as expected."
            )
            msg = '\n'.join(msg_lines)
            if fr_strict_reference:
                raise FileNotFoundError(msg)
            logger.warning(
                "%s\nFalling back to dropping invalid FR samples because fr_strict_reference=0 "
                "(kept %d/%d).",
                msg, len(kept), len(samples),
            )
        return kept

    # ---- Spatial sampler config ----
    detail_sampler_name = str(cfg.get('detail_sampler', 'gms')).lower().strip()
    gms_patch = cfg.get('gms_patch', 256)
    gms_ppf = cfg.get('gms_patches_per_frame', 8)
    detail_resize_target = int(cfg.get('detail_resize_target', 224))
    detail_gms_reduce = 'none'

    sem_sampler_name = str(cfg.get('semantic_sampler', 'resize224')).lower().strip()
    semantic_sampler_mode = 'resize'
    semantic_gms_mode = 'avg'
    semantic_gms_legacy_pe = False
    resize_target = int(cfg.get('semantic_target_size', 224))
    semantic_gms_ppf = int(cfg.get('semantic_gms_patches_per_frame', gms_ppf))
    semantic_gms_grid = int(cfg.get('semantic_gms_grid_size', cfg.get('gms_grid_size', 7)))
    semantic_frag_h = int(cfg.get('semantic_fragment_grid_h', cfg.get('fragment_grid_h', 7)))
    semantic_frag_w = int(cfg.get('semantic_fragment_grid_w', cfg.get('fragment_grid_w', 7)))
    semantic_frag_size = int(cfg.get('semantic_fragment_size', cfg.get('fragment_size', 32)))

    # Parse detail sampler:
    #   resizeNNN / gmsNNN / gmsavgNNN / gmsaugNNN / fragment...
    if detail_sampler_name.startswith('resize'):
        suffix = detail_sampler_name[len('resize'):].strip()
        if suffix:
            try:
                detail_resize_target = int(suffix)
            except ValueError:
                pass
        detail_sampler_name = 'resize'
    elif detail_sampler_name.startswith('gmsavg'):
        m = re.search(r'(\d+)$', detail_sampler_name)
        if m:
            gms_patch = int(m.group(1))
        detail_sampler_name = 'gms'
        detail_gms_reduce = 'avg'
    elif detail_sampler_name.startswith('gmsaug'):
        m = re.search(r'(\d+)$', detail_sampler_name)
        if m:
            gms_patch = int(m.group(1))
        detail_sampler_name = 'gms'
        detail_gms_reduce = 'random1'
    elif detail_sampler_name.startswith('gms'):
        m = re.search(r'(\d+)$', detail_sampler_name)
        if m:
            gms_patch = int(m.group(1))
        detail_sampler_name = 'gms'

    # Parse semantic sampler family and target:
    #   resizeNNN / center_crop / gmsavgNNN / gmsaugNNN / fragmentNNN
    if sem_sampler_name.startswith('resize'):
        suffix = sem_sampler_name[len('resize'):].strip()
        if suffix:
            try:
                resize_target = int(suffix)
            except ValueError:
                pass
        semantic_sampler_mode = 'resize'
    elif sem_sampler_name == 'center_crop':
        semantic_sampler_mode = 'resize'
    elif sem_sampler_name.startswith('gmsavg') or sem_sampler_name.startswith('gms_avg'):
        semantic_sampler_mode = 'gmsavg'
        semantic_gms_mode = 'avg'
        m = re.search(r'(\d+)$', sem_sampler_name)
        if m:
            resize_target = int(m.group(1))
    elif sem_sampler_name.startswith('gmsaug') or sem_sampler_name.startswith('gms_aug'):
        semantic_sampler_mode = 'gmsavg'
        semantic_gms_mode = 'random1'
        m = re.search(r'(\d+)$', sem_sampler_name)
        if m:
            resize_target = int(m.group(1))
    elif sem_sampler_name.startswith('fragment'):
        semantic_sampler_mode = 'fragment'
        m = re.search(r'(\d+)$', sem_sampler_name)
        if m:
            resize_target = int(m.group(1))
    elif sem_sampler_name.startswith('mss'):
        semantic_sampler_mode = 'mss'
        m = re.search(r'(\d+)$', sem_sampler_name)
        if m:
            resize_target = int(m.group(1))
    else:
        logger.warning(
            "Unknown semantic_sampler '%s', fallback to resize224.",
            sem_sampler_name,
        )
        sem_sampler_name = 'resize224'
        semantic_sampler_mode = 'resize'
        resize_target = 224

    if any_align_enabled and semantic_sampler_mode == 'gmsavg':
        sem_mode_override = str(active_align_cfg.get('semantic_gms_mode', semantic_gms_mode)).lower().strip()
        if sem_mode_override in ('avg', 'random1', 'stack'):
            semantic_gms_mode = sem_mode_override
        semantic_gms_legacy_pe = bool(active_align_cfg.get('semantic_legacy_grid', False))
    elif not any_align_enabled and semantic_sampler_mode == 'gmsavg':
        # Also support semantic_gms_mode override from top-level or semantic_branch config
        sem_gms_mode_top = str(cfg.get('semantic_gms_mode', '')).lower().strip()
        if sem_gms_mode_top in ('avg', 'random1', 'stack'):
            semantic_gms_mode = sem_gms_mode_top
        sem_legacy_grid_top = bool(cfg.get('semantic_gms_legacy_pe', False))
        if sem_legacy_grid_top:
            semantic_gms_legacy_pe = True

    # Adaptive crop-resize (Scheme G)
    semantic_gms_adaptive_crop = bool(cfg.get('semantic_gms_adaptive_crop', False))
    adaptive_crop_max_scale = float(cfg.get('adaptive_crop_max_scale', 0.0))

    # For gmsavg/gmsaug ablations, default to 16 views if config is smaller.
    # --no_gms_min_patches 1 disables this fallback, allowing genuine use of <16 patches
    no_gms_min_patches = bool(cfg.get('no_gms_min_patches', False))
    if not no_gms_min_patches:
        if detail_gms_reduce in ('avg', 'random1') and int(gms_ppf) < 16:
            gms_ppf = 16
        if semantic_sampler_mode == 'gmsavg' and int(semantic_gms_ppf) < 16:
            semantic_gms_ppf = 16

    data_error_fail_fast = bool(cfg.get('data_error_fail_fast', True))
    data_error_max_count = int(cfg.get('data_error_max_count', 8))
    data_error_max_ratio = float(cfg.get('data_error_max_ratio', 0.05))
    fr_align_mode = str(cfg.get('fr_align_mode', 'auto')).lower().strip()
    fr_align_ratio_threshold = float(cfg.get('fr_align_ratio_threshold', 1.2))

    if dry_run:
        # For fragment mode, mock P=1, patch_size = grid * frag_size
        if detail_sampler_name == 'fragment':
            mock_p = 1
            mock_ph = cfg.get('fragment_grid_h', 7) * cfg.get('fragment_size', 32)
        elif detail_sampler_name == 'resize':
            mock_p = 1
            mock_ph = detail_resize_target
        elif detail_sampler_name == 'fupic':
            mock_p = 4
            mock_ph = cfg.get('fupic_tile', 224)
        else:
            mock_p = gms_ppf
            mock_ph = gms_patch
        train_ds = MockDataset(size=40, num_frames=num_frames, is_fr=is_fr,
                               gms_patches=mock_p, gms_patch_size=mock_ph,
                               resize_size=resize_target,
                               detail_sampler=detail_sampler_name,
                               vif_branch_cfg=vif_branch_cfg,
                               enable_detail_branch=detail_enabled,
                               enable_semantic_branch=semantic_enabled)
        val_ds = {} if skip_val else {
            'mock_val': MockDataset(size=10, num_frames=num_frames, is_fr=is_fr,
                                    gms_patches=mock_p, gms_patch_size=mock_ph,
                                    resize_size=resize_target,
                                    detail_sampler=detail_sampler_name,
                                    vif_branch_cfg=vif_branch_cfg,
                                    enable_detail_branch=detail_enabled,
                                    enable_semantic_branch=semantic_enabled)
        }
        test_ds = {
            'mock_test': MockDataset(size=10, num_frames=num_frames, is_fr=is_fr,
                                     gms_patches=mock_p, gms_patch_size=mock_ph,
                                     resize_size=resize_target,
                                     detail_sampler=detail_sampler_name,
                                     vif_branch_cfg=vif_branch_cfg,
                                     enable_detail_branch=detail_enabled,
                                     enable_semantic_branch=semantic_enabled)
        }
        return train_ds, val_ds, test_ds

    reader = VideoReaderFactory(
        container_decoder=cfg.get('container_decoder', 'pyav'),
        auto_probe=cfg.get('auto_probe', 'on') == 'on',
        default_width=cfg.get('width'),
        default_height=cfg.get('height'),
        default_bitdepth=cfg.get('bitdepth', 10),
        default_pix_fmt=cfg.get('pixel_format', 'yuv420p10le'),
        default_signal_range=signal_range,
        default_tenbit_mode=tenbit_mode,
        uv_upsample=cfg.get('uv_upsample', 'bicubic'),
        raw_yuv_backend=raw_yuv_backend,
        raw_yuv_matrix=raw_yuv_matrix,
        container_yuv_matrix=container_yuv_matrix,
        container_yuv_direct=container_yuv_direct,
    )

    # ---- Frame samplers ----
    train_frame_sampler = FrameSampler(
        num_frames=num_frames,
        num_clips=cfg.get('num_clips_train', 2),
        strategy=temporal_sampler,
        is_train=True,
        sampling_rate=sampling_rate,
        legacy_pe_window_sampling=legacy_temporal_window_sampling,
    )
    test_frame_sampler = FrameSampler(
        num_frames=num_frames,
        num_clips=cfg.get('num_clips_test', 4),
        strategy=temporal_sampler,
        is_train=False,
        sampling_rate=sampling_rate,
        legacy_pe_window_sampling=legacy_temporal_window_sampling,
    )

    # ---- Spatial samplers ----
    gms_sampler_train = None
    gms_sampler_test = None
    fragment_sampler_train = None
    fragment_sampler_test = None
    fupic_sampler_train = None
    fupic_sampler_test = None
    detail_resize_sampler_train = None
    detail_resize_sampler_test = None
    semantic_gms_sampler_train = None
    semantic_gms_sampler_test = None
    semantic_fragment_sampler_train = None
    semantic_fragment_sampler_test = None
    mss_gms_sampler_train = None
    mss_gms_sampler_test = None

    if detail_enabled:
        if detail_sampler_name == 'fullres':
            raise ValueError(
                "detail_sampler='fullres' is not supported by the current detail branch pipeline. "
                "Use one of: gms, gmsavg, gmsaug, fragment, fupic, resize."
            )
        if detail_sampler_name == 'resize':
            detail_resize_sampler_train = ResizeSampler(target_size=detail_resize_target)
            detail_resize_sampler_test = ResizeSampler(target_size=detail_resize_target)
            logger.info("Detail sampler: RESIZE (target=%d)", detail_resize_target)
        elif detail_sampler_name == 'fragment':
            g_h = int(cfg.get('fragment_grid_h', 7))
            g_w = int(cfg.get('fragment_grid_w', 7))
            if g_h != g_w:
                raise ValueError(
                    f"fragment_grid_h ({g_h}) must equal fragment_grid_w ({g_w}) for Swin detail backbone."
                )
            fragment_sampler_train = FragmentSampler(
                grid_h=g_h,
                grid_w=g_w,
                frag_size=cfg.get('fragment_size', 32),
                is_train=True,
            )
            fragment_sampler_test = FragmentSampler(
                grid_h=g_h,
                grid_w=g_w,
                frag_size=cfg.get('fragment_size', 32),
                is_train=False,
            )
            logger.info(f"Detail sampler: FRAGMENT (grid={g_h}x{g_w}, "
                         f"frag_size={cfg.get('fragment_size', 32)}, "
                         f"composite={fragment_sampler_train.patch_size}x"
                         f"{fragment_sampler_train.patch_size})")
        elif detail_sampler_name == 'fupic':
            fupic_sampler_train = FuPiCSampler(
                tile_size=cfg.get('fupic_tile', 224),
                stride=cfg.get('fupic_stride', 192),
            )
            fupic_sampler_test = FuPiCSampler(
                tile_size=cfg.get('fupic_tile', 224),
                stride=cfg.get('fupic_stride', 192),
            )
            logger.info(
                "Detail sampler: FUPIC (tile=%d, stride=%d)",
                cfg.get('fupic_tile', 224),
                cfg.get('fupic_stride', 192),
            )
        else:
            gms_sampler_train = GMSSampler(
                patch_size=gms_patch,
                patches_per_frame=gms_ppf,
                grid_size=cfg.get('gms_grid_size', 7),
                is_train=True,
            )
            gms_sampler_test = GMSSampler(
                patch_size=gms_patch,
                patches_per_frame=gms_ppf,
                grid_size=cfg.get('gms_grid_size', 7),
                is_train=False,
            )
            logger.info(
                "Detail sampler: GMS (patch=%d, P=%d, grid=%d)",
                gms_patch, gms_ppf, cfg.get('gms_grid_size', 7),
            )
    else:
        logger.info("Detail branch disabled: skip detail spatial sampler construction.")

    if semantic_enabled:
        if semantic_sampler_mode == 'resize':
            resize_sampler = build_spatial_sampler(sem_sampler_name, cfg, is_train=True)
            logger.info("Semantic sampler: %s (target_size=%d)", sem_sampler_name, resize_target)
        elif semantic_sampler_mode == 'gmsavg':
            resize_sampler = None
            semantic_gms_sampler_train = GMSSampler(
                patch_size=resize_target,
                patches_per_frame=semantic_gms_ppf,
                grid_size=semantic_gms_grid,
                is_train=True,
            )
            semantic_gms_sampler_test = GMSSampler(
                patch_size=resize_target,
                patches_per_frame=semantic_gms_ppf,
                grid_size=semantic_gms_grid,
                is_train=False,
            )
            logger.info(
                "Semantic sampler: GMS (%s, target=%d, P=%d, grid=%d)",
                semantic_gms_mode, resize_target, semantic_gms_ppf, semantic_gms_grid,
            )
        elif semantic_sampler_mode == 'fragment':
            resize_sampler = None
            semantic_fragment_sampler_train = FragmentSampler(
                grid_h=semantic_frag_h,
                grid_w=semantic_frag_w,
                frag_size=semantic_frag_size,
                is_train=True,
            )
            semantic_fragment_sampler_test = FragmentSampler(
                grid_h=semantic_frag_h,
                grid_w=semantic_frag_w,
                frag_size=semantic_frag_size,
                is_train=False,
            )
            logger.info(
                "Semantic sampler: FRAGMENT (target=%d, grid=%dx%d, frag=%d)",
                resize_target, semantic_frag_h, semantic_frag_w, semantic_frag_size,
            )
        elif semantic_sampler_mode == 'mss':
            # MSS: global resize + local GMS patches
            resize_sampler = None
            mss_patches = int(cfg.get('sem_mss_patches', 16))
            mss_gms_sampler_train = GMSSampler(
                patch_size=224,
                patches_per_frame=mss_patches,
                grid_size=semantic_gms_grid,
                is_train=True,
            )
            mss_gms_sampler_test = GMSSampler(
                patch_size=224,
                patches_per_frame=mss_patches,
                grid_size=semantic_gms_grid,
                is_train=False,
            )
            logger.info(
                "Semantic sampler: MSS (global_resize=%d, local_gms_patches=%d, grid=%d)",
                resize_target, mss_patches, semantic_gms_grid,
            )
        else:
            resize_sampler = build_spatial_sampler('resize224', cfg, is_train=True)
            logger.warning(
                "Unknown semantic sampler mode '%s', fallback to resize224.",
                semantic_sampler_mode,
            )
    else:
        resize_sampler = None
        logger.info("Semantic branch disabled: skip semantic spatial sampler construction.")

    # ---- Choose test samplers based on test_eval_mode ----
    if test_eval_mode == 'fast':
        # Same frame count as training, deterministic positions
        eff_test_frame = FrameSampler(
            num_frames=num_frames,
            num_clips=cfg.get('num_clips_train', 2),
            strategy=temporal_sampler,
            is_train=False,  # deterministic
            sampling_rate=sampling_rate,
            legacy_pe_window_sampling=legacy_temporal_window_sampling,
        )
        logger.info(f"Test eval mode: FAST (same clips as training, "
                     f"num_clips={cfg.get('num_clips_train', 2)}, "
                     f"strategy={temporal_sampler}, sampling_rate={sampling_rate})")
    else:
        eff_test_frame = test_frame_sampler
        logger.info(f"Test eval mode: FULL (num_clips={cfg.get('num_clips_test', 4)}, "
                    f"strategy={temporal_sampler}, sampling_rate={sampling_rate})")

    # ---- Build train datasets ----
    train_datasets = []
    train_cfg = cfg.get('train_datasets', '')
    if isinstance(train_cfg, str):
        train_dataset_names = train_cfg.split(',')
    elif isinstance(train_cfg, (list, tuple)):
        train_dataset_names = [str(x) for x in train_cfg]
    else:
        train_dataset_names = []
    for name in train_dataset_names:
        name = name.strip()
        if not name:
            continue
        try:
            _validate_dataset_layout_or_raise(name, 'train')
            parser = get_parser(name)
            samples = parser(mode='train')
            samples = _filter_fr_samples(samples, name, 'train')
            if samples:
                ds = VQADataset(
                    samples, reader, train_frame_sampler,
                    is_fr=is_fr, is_train=True,
                    gms_sampler=gms_sampler_train,
                    fupic_sampler=fupic_sampler_train,
                    detail_resize_sampler=detail_resize_sampler_train,
                    resize_sampler=resize_sampler,
                    fragment_sampler=fragment_sampler_train,
                    detail_gms_reduce=detail_gms_reduce,
                    semantic_gms_sampler=semantic_gms_sampler_train,
                    semantic_gms_mode=semantic_gms_mode,
                    semantic_gms_legacy_pe=semantic_gms_legacy_pe,
                    semantic_gms_adaptive_crop=semantic_gms_adaptive_crop,
                    adaptive_crop_max_scale=adaptive_crop_max_scale,
                    semantic_fragment_sampler=semantic_fragment_sampler_train,
                    semantic_target_size=resize_target,
                    mss_gms_sampler=mss_gms_sampler_train,
                    vif_branch_cfg=vif_branch_cfg,
                    output_dir=output_dir,
                    vif_only_mode=vif_only_mode,
                    enable_detail_branch=detail_enabled,
                    enable_semantic_branch=semantic_enabled,
                    data_error_fail_fast=data_error_fail_fast,
                    data_error_max_count=data_error_max_count,
                    data_error_max_ratio=data_error_max_ratio,
                    fr_align_mode=fr_align_mode,
                    fr_align_ratio_threshold=fr_align_ratio_threshold,
                    use_cvqm_npy=use_cvqm_npy,
                    cvqm_npy_segments=cvqm_npy_segments,
                    cvqm_npy_colorspace=cvqm_npy_colorspace,
                    resize_antialias=resize_antialias,
                    frame_cache_root=frame_cache_root,
                    sampled_clip_cache_mode=sampled_clip_cache_mode,
                    sampled_clip_cache_root=sampled_clip_cache_root,
                    ref_clip_cache_size=ref_clip_cache_size,
                    ref_clip_cache_max_mb=ref_clip_cache_max_mb,
                    aug_hflip=bool(cfg.get('aug_hflip', False)),
                    aug_tflip=bool(cfg.get('aug_tflip', False)),
                    aug_brightness=float(cfg.get('aug_brightness', 0.0)),
                    sem_adaptive_patches=bool(cfg.get('sem_adaptive_patches', False)),
                    gradient_topk_sampling=bool(cfg.get('gradient_topk_sampling', False)),
                    gradient_topk_mode=str(cfg.get('gradient_topk_mode', 'weighted')),
                )
                ds.log_vif_cache_status()
                ds.prebuild_vif_cache()
                train_datasets.append(ds)
                logger.info(f"Train dataset {name}: {len(samples)} samples")
        except Exception as e:
            if dataset_path_fail_fast or fr_strict_reference:
                raise
            logger.warning(f"Failed to load train dataset {name}: {e}")

    train_ds = ConcatDataset(train_datasets) if train_datasets else None
    allow_empty_train = bool(cfg.get('allow_empty_train', False))
    if train_ds is None and not allow_empty_train:
        raise RuntimeError(
            "No training samples were loaded. Please verify dataset paths/splits and FR reference availability."
        )

    # ---- Build val datasets (skip if skip_val) ----
    val_datasets = {}
    if not skip_val:
        for name in train_dataset_names:
            name = name.strip()
            if not name:
                continue
            try:
                _validate_dataset_layout_or_raise(name, 'val')
                parser = get_parser(name)
                samples = parser(mode='val')
                samples = _filter_fr_samples(samples, name, 'val')
                if samples:
                    ds = VQADataset(
                        samples, reader, eff_test_frame,
                        is_fr=is_fr, is_train=False,
                        gms_sampler=gms_sampler_test,
                        fupic_sampler=fupic_sampler_test,
                        detail_resize_sampler=detail_resize_sampler_test,
                        resize_sampler=resize_sampler,
                        fragment_sampler=fragment_sampler_test,
                        detail_gms_reduce=detail_gms_reduce,
                        semantic_gms_sampler=semantic_gms_sampler_test,
                        semantic_gms_mode=semantic_gms_mode,
                        semantic_gms_legacy_pe=semantic_gms_legacy_pe,
                        semantic_gms_adaptive_crop=semantic_gms_adaptive_crop,
                        adaptive_crop_max_scale=adaptive_crop_max_scale,
                        sem_adaptive_patches=bool(cfg.get('sem_adaptive_patches', False)),
                        semantic_fragment_sampler=semantic_fragment_sampler_test,
                        semantic_target_size=resize_target,
                        mss_gms_sampler=mss_gms_sampler_test,
                        vif_branch_cfg=vif_branch_cfg,
                        output_dir=output_dir,
                        vif_only_mode=vif_only_mode,
                        enable_detail_branch=detail_enabled,
                        enable_semantic_branch=semantic_enabled,
                        data_error_fail_fast=data_error_fail_fast,
                        data_error_max_count=data_error_max_count,
                        data_error_max_ratio=data_error_max_ratio,
                        fr_align_mode=fr_align_mode,
                        fr_align_ratio_threshold=fr_align_ratio_threshold,
                        use_cvqm_npy=use_cvqm_npy,
                        cvqm_npy_segments=cvqm_npy_segments,
                        cvqm_npy_colorspace=cvqm_npy_colorspace,
                        resize_antialias=resize_antialias,
                        frame_cache_root=frame_cache_root,
                        sampled_clip_cache_mode=sampled_clip_cache_mode,
                        sampled_clip_cache_root=sampled_clip_cache_root,
                        ref_clip_cache_size=ref_clip_cache_size,
                        ref_clip_cache_max_mb=ref_clip_cache_max_mb,
                    )
                    ds.log_vif_cache_status()
                    ds.prebuild_vif_cache()
                    val_datasets[name] = ds
                    logger.info(f"Val dataset {name}: {len(samples)} samples")
            except Exception as e:
                if dataset_path_fail_fast or fr_strict_reference:
                    raise
                logger.warning(f"Failed to load val dataset {name}: {e}")
    else:
        logger.info("Skipping val datasets (skip_val=True)")

    # ---- Build test datasets ----
    test_datasets = {}
    test_name = cfg.get('test_dataset', 'CVQM')
    test_dataset_mode = str(cfg.get('test_dataset_mode', 'test') or 'test').strip()
    if test_name:
        try:
            if test_name == 'CVQM':
                _validate_dataset_layout_or_raise('CVQM', 'test')
                cvqm_phase = cfg.get('cvqm_phase', 'all')
                samples = parse_cvqm(mode='test', cvqm_phase=cvqm_phase)
                # multi_clip_4x8 mode: replace Phase2 4K video paths with the Resize1080p version
                # Also replace REF paths with the Resize1080p version, ensuring dis/ref resolutions match
                if _is_multi_clip_temporal and samples:
                    _cvqm_cfg = get_dataset_config('CVQM')
                    _old_phase2_root = str(_cvqm_cfg.get('phase_roots', {}).get(2, '') or '')
                    _old_ref_root = str(_cvqm_cfg.get('ref_root', '') or '')
                    _replaced_dis = 0
                    _replaced_ref = 0
                    for s in samples:
                        # Replace Phase2 dis_path
                        if _cvqm_phase2_resize_override and getattr(s, 'stage', None) == 2:
                            if _old_phase2_root and s.dis_path.startswith(_old_phase2_root):
                                s.dis_path = s.dis_path.replace(_old_phase2_root, _cvqm_phase2_resize_override, 1)
                                _replaced_dis += 1
                        # Replace REF path (refs from any phase may be 4K)
                        if _cvqm_ref_resize_override and s.ref_path:
                            if _old_ref_root and s.ref_path.startswith(_old_ref_root):
                                s.ref_path = s.ref_path.replace(_old_ref_root, _cvqm_ref_resize_override, 1)
                                _replaced_ref += 1
                    if _replaced_dis > 0 or _replaced_ref > 0:
                        logger.info(
                            "[MultiClipTemporal] CVQM path override: %d dis paths → Resize1080p, "
                            "%d ref paths → Resize1080p",
                            _replaced_dis, _replaced_ref,
                        )
                samples = _filter_fr_samples(samples, test_name, 'test')
                if samples:
                    ds = VQADataset(
                        samples, reader, eff_test_frame,
                        is_fr=is_fr, is_train=False,
                        gms_sampler=gms_sampler_test,
                        fupic_sampler=fupic_sampler_test,
                        detail_resize_sampler=detail_resize_sampler_test,
                        resize_sampler=resize_sampler,
                        fragment_sampler=fragment_sampler_test,
                        detail_gms_reduce=detail_gms_reduce,
                        semantic_gms_sampler=semantic_gms_sampler_test,
                        semantic_gms_mode=semantic_gms_mode,
                        semantic_gms_legacy_pe=semantic_gms_legacy_pe,
                        semantic_gms_adaptive_crop=semantic_gms_adaptive_crop,
                        adaptive_crop_max_scale=adaptive_crop_max_scale,
                        semantic_fragment_sampler=semantic_fragment_sampler_test,
                        semantic_target_size=resize_target,
                        mss_gms_sampler=mss_gms_sampler_test,
                        vif_branch_cfg=vif_branch_cfg,
                        output_dir=output_dir,
                        vif_only_mode=vif_only_mode,
                        enable_detail_branch=detail_enabled,
                        enable_semantic_branch=semantic_enabled,
                        data_error_fail_fast=data_error_fail_fast,
                        data_error_max_count=data_error_max_count,
                        data_error_max_ratio=data_error_max_ratio,
                        fr_align_mode=fr_align_mode,
                        fr_align_ratio_threshold=fr_align_ratio_threshold,
                        use_cvqm_npy=use_cvqm_npy,
                        cvqm_npy_segments=cvqm_npy_segments,
                        cvqm_npy_colorspace=cvqm_npy_colorspace,
                        resize_antialias=resize_antialias,
                        frame_cache_root=frame_cache_root,
                        sampled_clip_cache_mode=sampled_clip_cache_mode,
                        sampled_clip_cache_root=sampled_clip_cache_root,
                        ref_clip_cache_size=ref_clip_cache_size,
                        ref_clip_cache_max_mb=ref_clip_cache_max_mb,
                    )
                    ds.log_vif_cache_status()
                    ds.prebuild_vif_cache()
                    test_datasets['CVQM'] = ds
                    logger.info(f"Test dataset CVQM: {len(samples)} samples "
                               f"(test_eval_mode={test_eval_mode})")
            else:
                _validate_dataset_layout_or_raise(test_name, test_dataset_mode)
                parser = get_parser(test_name)
                samples = parser(mode=test_dataset_mode)
                samples = _filter_fr_samples(samples, test_name, 'test')
                if samples:
                    ds = VQADataset(
                        samples, reader, eff_test_frame,
                        is_fr=is_fr, is_train=False,
                        gms_sampler=gms_sampler_test,
                        fupic_sampler=fupic_sampler_test,
                        detail_resize_sampler=detail_resize_sampler_test,
                        resize_sampler=resize_sampler,
                        fragment_sampler=fragment_sampler_test,
                        detail_gms_reduce=detail_gms_reduce,
                        semantic_gms_sampler=semantic_gms_sampler_test,
                        semantic_gms_mode=semantic_gms_mode,
                        semantic_gms_legacy_pe=semantic_gms_legacy_pe,
                        semantic_gms_adaptive_crop=semantic_gms_adaptive_crop,
                        adaptive_crop_max_scale=adaptive_crop_max_scale,
                        sem_adaptive_patches=bool(cfg.get('sem_adaptive_patches', False)),
                        semantic_fragment_sampler=semantic_fragment_sampler_test,
                        semantic_target_size=resize_target,
                        mss_gms_sampler=mss_gms_sampler_test,
                        vif_branch_cfg=vif_branch_cfg,
                        output_dir=output_dir,
                        vif_only_mode=vif_only_mode,
                        enable_detail_branch=detail_enabled,
                        enable_semantic_branch=semantic_enabled,
                        data_error_fail_fast=data_error_fail_fast,
                        data_error_max_count=data_error_max_count,
                        data_error_max_ratio=data_error_max_ratio,
                        fr_align_mode=fr_align_mode,
                        fr_align_ratio_threshold=fr_align_ratio_threshold,
                        use_cvqm_npy=use_cvqm_npy,
                        cvqm_npy_segments=cvqm_npy_segments,
                        cvqm_npy_colorspace=cvqm_npy_colorspace,
                        resize_antialias=resize_antialias,
                        frame_cache_root=frame_cache_root,
                        sampled_clip_cache_mode=sampled_clip_cache_mode,
                        sampled_clip_cache_root=sampled_clip_cache_root,
                        ref_clip_cache_size=ref_clip_cache_size,
                        ref_clip_cache_max_mb=ref_clip_cache_max_mb,
                    )
                    ds.log_vif_cache_status()
                    ds.prebuild_vif_cache()
                    test_datasets[test_name] = ds
                    logger.info(
                        "Test dataset %s: %d samples (mode=%s, test_eval_mode=%s)",
                        test_name, len(samples), test_dataset_mode, test_eval_mode,
                    )
        except Exception as e:
            if dataset_path_fail_fast or fr_strict_reference:
                raise
            logger.warning(f"Failed to load test dataset {test_name}: {e}")

    return train_ds, val_datasets, test_datasets


def build_dataloaders(
    cfg: dict,
    train_ds,
    val_datasets: dict,
    test_datasets: dict,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[Optional[DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    """Build DataLoaders with optional DDP samplers."""
    from ..utils.seed import worker_init_fn

    batch_size = cfg.get('batch_size', 1)
    val_batch_mul = float(cfg.get('val_batch_mul', 2))
    test_batch_mul = float(cfg.get('test_batch_mul', 2))
    num_workers = cfg.get('num_workers', 4)
    branches_cfg = cfg.get('branches', '')
    if isinstance(branches_cfg, str):
        b_list = [b.strip().lower() for b in branches_cfg.split(',') if b.strip()]
    else:
        b_list = [str(b).strip().lower() for b in branches_cfg]
    vif_cfg = cfg.get('vif_branch', {})
    vif_enabled = (
        str(cfg.get('task', 'fr')).lower() == 'fr' and
        ('vif' in b_list) and
        bool(vif_cfg.get('enable', True))
    )
    detail_enabled = ('detail' in b_list)
    semantic_enabled = ('semantic' in b_list)
    lightweight_semantic_only = semantic_enabled and (not detail_enabled) and (not vif_enabled)

    # -- Auto-adjust num_workers per machine profile --
    # V100/HPC (44 CPU cores, shared NAS): running 4 experiments × 8 workers = 32 workers
    # causes /dev/shm exhaustion and CPU oversubscription.
    # Reduce to 2 workers on HPC to safely run up to 4 experiments in parallel.
    # 4090 (176 CPU cores, NVMe): keep the config value (typically 8).
    # Override: HMF_VQA_NUM_WORKERS=N to force a specific value on any machine.
    _nw_override = os.environ.get('HMF_VQA_NUM_WORKERS', '').strip()
    if _nw_override.isdigit():
        _old_nw = num_workers
        num_workers = int(_nw_override)
        if _old_nw != num_workers:
            logger.info("[NumWorkers] Override via HMF_VQA_NUM_WORKERS: %d -> %d", _old_nw, num_workers)
    else:
        # No env override: trust the config value (yaml or CLI).
        # Previously had profile-based auto-adjust (hpc→2), but V100 machines
        # have sufficient memory (352GB RAM, 8GB /dev/shm) for higher worker counts.
        pass

    # -- Auto-reduce num_workers when /dev/shm is too small --
    # DataLoader multiprocessing uses shared memory (/dev/shm) for tensor transfer.
    # Some Docker containers have tiny /dev/shm (e.g. 64MB), causing crashes.
    # We check actual /dev/shm size instead of GPU model, because some HPC setups
    # (e.g. Alibaba Cloud DSW) have adequate /dev/shm even with V100/A100.
    # Set HMF_VQA_FORCE_NUM_WORKERS=1 to skip this auto-detection.
    _force_nw = os.environ.get('HMF_VQA_FORCE_NUM_WORKERS', '0') == '1'
    if num_workers > 0 and not _force_nw:
        _shm_too_small = False
        _shm_mb = 0
        try:
            _shm_stat = os.statvfs('/dev/shm')
            _shm_mb = int((_shm_stat.f_bavail * _shm_stat.f_frsize) / (1024 ** 2))
            # Need at least 2GB for safe multi-worker DataLoader with large tensors
            if _shm_mb < 2048:
                _shm_too_small = True
        except (OSError, ValueError):
            # Cannot stat /dev/shm — assume it might be restricted
            _shm_too_small = True
        if _shm_too_small:
            logger.warning(
                "/dev/shm is only %dMB (need >= 2048MB for multi-worker DataLoader). "
                "Auto-setting num_workers=0. "
                "Fix: docker run --shm-size=16g, or set HMF_VQA_FORCE_NUM_WORKERS=1 to override.",
                _shm_mb,
            )
            num_workers = 0
        elif _shm_mb > 0:
            logger.info("/dev/shm = %dMB, sufficient for num_workers=%d", _shm_mb, num_workers)

    pe_align_cfg = cfg.get('pe_align', {})
    pe_align_enabled = isinstance(pe_align_cfg, dict) and bool(pe_align_cfg.get('enabled', False))
    swin_align_cfg = cfg.get('swin_align', {})
    swin_align_enabled = isinstance(swin_align_cfg, dict) and bool(swin_align_cfg.get('enabled', False))
    convnext_align_cfg = cfg.get('convnext_align', {})
    convnext_align_enabled = isinstance(convnext_align_cfg, dict) and bool(convnext_align_cfg.get('enabled', False))
    any_align_enabled = pe_align_enabled or swin_align_enabled or convnext_align_enabled
    if pe_align_enabled:
        active_align_cfg_dl = pe_align_cfg
    elif swin_align_enabled:
        active_align_cfg_dl = swin_align_cfg
    elif convnext_align_enabled:
        active_align_cfg_dl = convnext_align_cfg
    else:
        active_align_cfg_dl = {}
    use_legacy_single_rank_sampler = bool(
        any_align_enabled and active_align_cfg_dl.get('legacy_single_rank_dist_sampler', False)
    )
    # semantic_branch fallback for legacy sampler/worker switches
    if not any_align_enabled:
        sem_cfg_dl = cfg.get('semantic_branch', {})
        if isinstance(sem_cfg_dl, dict):
            if bool(sem_cfg_dl.get('legacy_single_rank_dist_sampler', False)):
                use_legacy_single_rank_sampler = True
    distributed = world_size > 1
    if not distributed and use_legacy_single_rank_sampler and dist.is_available() and dist.is_initialized():
        distributed = True

    worker_init = worker_init_fn
    if any_align_enabled and bool(active_align_cfg_dl.get('legacy_disable_worker_init', False)):
        worker_init = None
    elif not any_align_enabled:
        sem_cfg_dl2 = cfg.get('semantic_branch', {})
        if isinstance(sem_cfg_dl2, dict) and bool(sem_cfg_dl2.get('legacy_disable_worker_init', False)):
            worker_init = None

    # Wrap worker_init_fn to pre-allocate frame cache buffer in each worker
    # Use the profile-resolved frame_cache_root (same logic as build_datasets)
    from src.data.datasets.base_dataset import _env_path as _dl_env_path  # noqa
    _resolved_fcr = _dl_env_path('HMF_VQA_FRAME_CACHE_ROOT', '')
    # Also check the explicit YAML/CLI value
    _yaml_fcr = str(cfg.get('frame_cache_root', '') or '').strip()
    _effective_fcr = (_yaml_fcr if _yaml_fcr and os.path.isdir(_yaml_fcr)
                      else _resolved_fcr if _resolved_fcr and os.path.isdir(_resolved_fcr)
                      else '')
    if _effective_fcr:
        _original_worker_init = worker_init
        def _worker_init_with_cache(worker_id):
            # Pre-allocate readinto buffer for 4K 10-bit 8-frame YUV420
            init_frame_cache_buffer._worker_id = worker_id  # pass to frame_cache for conditional logging
            init_frame_cache_buffer(max_width=4096, max_height=2160, num_frames=8, bitdepth=10)
            if _original_worker_init is not None:
                _original_worker_init(worker_id)
        worker_init = _worker_init_with_cache
        logger.info("[FrameCache] Worker init wrapped with frame cache buffer pre-allocation.")

        # Set OMP_NUM_THREADS via environment variable BEFORE DataLoader fork().
        # CRITICAL: torch.set_num_threads() must NEVER be called inside forked
        # DataLoader workers — it causes SEGFAULT or deadlock because the main
        # process has already initialized OpenMP (during model weight loading),
        # and fork() copies the corrupted lock state to child processes.
        # Environment variables are inherited cleanly by fork() and used by
        # OpenMP when it initializes in the child process for the first time.
        from .cache.frame_cache import TORCH_DECODE_THREADS, _TORCH_DECODE_ENABLED
        if _TORCH_DECODE_ENABLED:
            _omp_threads = str(TORCH_DECODE_THREADS)
            if 'OMP_NUM_THREADS' not in os.environ:
                os.environ['OMP_NUM_THREADS'] = _omp_threads
                os.environ['MKL_NUM_THREADS'] = _omp_threads
                logger.info("[FrameCache] Set OMP_NUM_THREADS=%s for worker processes.", _omp_threads)

    logger.info(
        "Dataloader switches: distributed=%s (world_size=%d, dist_initialized=%s, legacy_single_rank_dist_sampler=%s), worker_init_fn=%s",
        str(distributed),
        int(world_size),
        str(dist.is_available() and dist.is_initialized()),
        str(use_legacy_single_rank_sampler),
        "off" if worker_init is None else "on",
    )

    # -- Shared memory detection for multiprocessing safety -------------------
    # When /dev/shm is small (e.g. Docker default 64MB), large tensors passed
    # between DataLoader workers and main process via shared memory will crash.
    # We auto-detect the available shm size and decide:
    #   - persistent_workers: safe to enable unless shm is tiny
    #   - multiprocessing_context: fall back to 'file_system' sharing if shm < threshold
    _shm_bytes = _get_shm_size()
    _shm_mb = _shm_bytes / (1024 ** 2) if _shm_bytes else 0
    # Heuristic: each worker may hold ~200MB of 4K data in shared memory.
    # Require at least num_workers * 256MB + 512MB headroom.
    _shm_threshold_mb = num_workers * 256 + 512
    _use_persistent = num_workers > 0
    _mp_context = None  # default: /dev/shm based
    if _shm_bytes is not None and _shm_mb < _shm_threshold_mb:
        # Shared memory too small: use file_system sharing strategy
        import multiprocessing
        _mp_context = multiprocessing.get_context('fork')
        torch.multiprocessing.set_sharing_strategy('file_system')
        logger.warning(
            "Shared memory is small (%.0fMB < %.0fMB threshold for %d workers). "
            "Switching to file_system sharing strategy. Consider increasing /dev/shm "
            "or reducing num_workers.",
            _shm_mb, _shm_threshold_mb, num_workers,
        )
    if num_workers > 0:
        logger.info(
            "DataLoader: num_workers=%d, persistent_workers=%s, pin_memory=True, shm=%.0fMB",
            num_workers, _use_persistent, _shm_mb,
        )

    _dl_extra = {}
    if num_workers > 0:
        _dl_extra['persistent_workers'] = _use_persistent
        if _mp_context is not None:
            _dl_extra['multiprocessing_context'] = _mp_context

    train_loader = None
    if train_ds is not None:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) \
            if distributed else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init,
            **_dl_extra,
        )

    val_loaders = {}
    for name, ds in val_datasets.items():
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) \
            if distributed else None
        val_loaders[name] = DataLoader(
            ds,
            batch_size=max(1, int(round(batch_size * val_batch_mul))),
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init,
            **_dl_extra,
        )

    test_loaders = {}
    for name, ds in test_datasets.items():
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) \
            if distributed else None
        test_loaders[name] = DataLoader(
            ds,
            batch_size=max(1, int(round(batch_size * test_batch_mul))),
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init,
            **_dl_extra,
        )

    return train_loader, val_loaders, test_loaders
