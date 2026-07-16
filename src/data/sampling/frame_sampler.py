"""
Frame Sampler: temporal sampling strategies.
Supports: Uniform, Centered, RandomStride, MultiClip(K).
"""

import math
import random
import logging
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger('hmf_vqa.frame_sampler')


def _pad_indices(indices: List[int], total_frames: int, target_len: int) -> List[int]:
    """Pad indices by looping or repeating last frame when video is too short."""
    if len(indices) >= target_len:
        return indices[:target_len]
    # Loop padding
    padded = list(indices)
    while len(padded) < target_len:
        if total_frames > 0:
            padded.append(padded[len(padded) % len(indices)])
        else:
            padded.append(0)
    return padded


class FrameSampler:
    """
    Frame sampling strategies.

    Args:
        num_frames: T, number of frames per clip
        num_clips: K, number of clips to sample
        strategy: 'uniform' | 'centered' | 'random_stride' | 'multiclip' | 'legacy_pe'
        is_train: if True, use random offsets
    """

    def __init__(
        self,
        num_frames: int = 8,
        num_clips: int = 1,
        strategy: str = 'multiclip',
        is_train: bool = True,
        sampling_rate: int = 1,
        legacy_pe_window_sampling: bool = False,
    ):
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.strategy = strategy
        self.is_train = is_train
        self.sampling_rate = max(1, int(sampling_rate))
        # When enabled, legacy_pe follows old PE _get_video_indices window+linspace behavior.
        self.legacy_pe_window_sampling = bool(legacy_pe_window_sampling)
        self._pad_count = 0
        self._total_count = 0

    @property
    def pad_ratio(self) -> float:
        return self._pad_count / max(self._total_count, 1)

    def sample(self, total_frames: int) -> List[List[int]]:
        """
        Sample frame indices.

        Returns:
            List of K clips, each clip is a list of T frame indices.
        """
        self._total_count += 1

        if total_frames <= 0:
            self._pad_count += 1
            return [[0] * self.num_frames for _ in range(self.num_clips)]

        if total_frames < self.num_frames:
            self._pad_count += 1

        if self.strategy == 'uniform':
            clips = [self._uniform(total_frames) for _ in range(self.num_clips)]
        elif self.strategy == 'uniform_clip':
            clips = self._uniform_clip(total_frames)
        elif self.strategy == 'centered':
            clips = [self._centered(total_frames) for _ in range(self.num_clips)]
        elif self.strategy == 'random_stride':
            clips = [self._random_stride(total_frames) for _ in range(self.num_clips)]
        elif self.strategy == 'multiclip':
            clips = self._multiclip(total_frames)
        elif self.strategy == 'legacy_pe':
            clips = [self._legacy_pe(total_frames, chunk_nb=k) for k in range(self.num_clips)]
        elif self.strategy == 'consecutive_random':
            clips = [self._consecutive_random(total_frames) for _ in range(self.num_clips)]
        elif self.strategy == 'burst_4x2':
            clips = [self._burst_4x2(total_frames) for _ in range(self.num_clips)]
        elif self.strategy == 'segment_2x4':
            clips = [self._segment_2x4(total_frames) for _ in range(self.num_clips)]
        elif self.strategy == 'mixed_sampler':
            clips = [self._mixed_sampler(total_frames) for _ in range(self.num_clips)]
        elif self.strategy == 'multi_clip_4x8':
            clips = self._multi_clip_4x8(total_frames)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

        return clips

    def sample_aligned(
        self,
        total_ref: int,
        total_dis: int,
        align_mode: str = 'normalized',
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        FR mode: sample aligned indices for ref and dis (may differ in length).
        Supports:
          - normalized: map by normalized temporal positions
          - index: keep ref indices in dis domain (with clamp)

        Returns:
            (ref_clips, dis_clips): each is List of K clips of T indices.
        """
        ref_clips = self.sample(total_ref)
        dis_clips = []
        mode = str(align_mode).lower().strip()
        if mode not in ('normalized', 'index'):
            raise ValueError(f"Unknown FR align mode: {align_mode}")

        for clip_ref in ref_clips:
            if mode == 'index':
                if total_dis <= 0:
                    dis_clip = [0] * self.num_frames
                else:
                    dis_clip = [min(max(int(idx), 0), total_dis - 1) for idx in clip_ref]
            else:
                # Map ref positions to dis via normalized time.
                if total_ref <= 1 or total_dis <= 0:
                    dis_clip = [0] * self.num_frames
                else:
                    t_norm = [idx / max(total_ref - 1, 1) for idx in clip_ref]
                    dis_clip = [min(int(round(t * max(total_dis - 1, 0))), total_dis - 1)
                                for t in t_norm]
            dis_clips.append(dis_clip)

        return ref_clips, dis_clips

    def _uniform(self, total_frames: int) -> List[int]:
        """Uniformly sample T frames."""
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=np.int64).tolist()
        return _pad_indices(indices, total_frames, self.num_frames)

    def _uniform_clip(self, total_frames: int) -> List[List[int]]:
        """
        Uniform-Clip: divide video into K clips, each clip samples T frames
        with equal inter-frame spacing.

        Core idea: divide video (or each clip segment) into T equal sub-segments,
        then pick one frame from each sub-segment at the SAME relative position.

        Train: random shared offset within sub-segment (same offset for all T
               sub-segments → constant inter-frame spacing, different each epoch).
        Test:  center of each sub-segment (deterministic, matches eval cache).

        Example (K=1, T=8, total=500):
          sub_len = 500/8 = 62.5
          Test:  offset = 31 → [31, 93, 156, 218, 281, 343, 406, 468]
          Train: offset = random(0,61) = 20 → [20, 82, 145, 207, 270, 332, 395, 457]
          → inter-frame spacing is always ~62, covering the full video uniformly.
        """
        clips = []
        seg_len = total_frames / self.num_clips

        for k in range(self.num_clips):
            seg_start = int(k * seg_len)
            seg_end = int((k + 1) * seg_len)
            seg_frames = seg_end - seg_start

            if seg_frames <= 0:
                clips.append([0] * self.num_frames)
                continue

            if seg_frames <= self.num_frames:
                # Not enough frames: linspace fallback
                indices = np.linspace(
                    seg_start, seg_end - 1, self.num_frames, dtype=np.int64
                ).tolist()
            else:
                # Divide segment into T equal sub-segments
                sub_len = seg_frames / self.num_frames
                max_offset = max(0, int(sub_len) - 1)

                if self.is_train:
                    # Random shared offset: same relative position in every sub-segment
                    offset = random.randint(0, max_offset) if max_offset > 0 else 0
                else:
                    # Deterministic: center of each sub-segment
                    offset = int(sub_len / 2)

                indices = []
                for t in range(self.num_frames):
                    idx = seg_start + int(t * sub_len) + offset
                    indices.append(min(idx, total_frames - 1))

            indices = [min(max(idx, 0), total_frames - 1) for idx in indices]
            clips.append(_pad_indices(indices, total_frames, self.num_frames))

        return clips

    def _centered(self, total_frames: int) -> List[int]:
        """Sample T frames centered in the video."""
        center = total_frames // 2
        half_span = self.num_frames // 2
        start = max(0, center - half_span)
        indices = list(range(start, min(start + self.num_frames, total_frames)))
        return _pad_indices(indices, total_frames, self.num_frames)

    def _random_stride(self, total_frames: int) -> List[int]:
        """Sample with random stride."""
        if self.is_train and total_frames > self.num_frames:
            max_stride = total_frames // self.num_frames
            stride = random.randint(1, max(1, max_stride))
            max_start = total_frames - stride * self.num_frames
            start = random.randint(0, max(0, max_start))
            indices = [start + i * stride for i in range(self.num_frames)]
            indices = [min(idx, total_frames - 1) for idx in indices]
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=np.int64).tolist()
        return _pad_indices(indices, total_frames, self.num_frames)

    def _multiclip(self, total_frames: int) -> List[List[int]]:
        """
        MultiClip: divide video into K segments, sample T frames from each.
        Train: random offset within each segment.
        Test: centered within each segment.
        """
        clips = []
        seg_len = total_frames / self.num_clips

        for k in range(self.num_clips):
            seg_start = int(k * seg_len)
            seg_end = int((k + 1) * seg_len)
            seg_frames = seg_end - seg_start

            if seg_frames <= 0:
                clips.append([0] * self.num_frames)
                continue

            if seg_frames <= self.num_frames:
                # Not enough frames in segment - uniform across segment
                indices = np.linspace(seg_start, seg_end - 1, self.num_frames, dtype=np.int64).tolist()
            else:
                if self.is_train:
                    # Random start within segment
                    max_start = seg_end - self.num_frames
                    start = random.randint(seg_start, max(seg_start, max_start))
                    indices = list(range(start, start + self.num_frames))
                else:
                    # Centered in segment
                    center = (seg_start + seg_end) // 2
                    half = self.num_frames // 2
                    start = max(seg_start, center - half)
                    indices = list(range(start, start + self.num_frames))

            indices = [min(max(idx, 0), total_frames - 1) for idx in indices]
            clips.append(_pad_indices(indices, total_frames, self.num_frames))

        return clips

    def _legacy_pe(self, total_frames: int, chunk_nb: int = 0) -> List[int]:
        """
        Legacy PE-style temporal sampling:
          - default (legacy_pe_window_sampling=False):
              fixed-interval stride sampling (current HMF behavior)
          - legacy_pe_window_sampling=True:
              old PE _get_video_indices style:
                train: random temporal window + linspace in window
                eval: segment-based temporal step (or centered when K=1)
        """
        if not self.legacy_pe_window_sampling:
            return self._legacy_pe_stride(total_frames)

        t = max(1, int(self.num_frames))
        sr = max(1, int(self.sampling_rate))
        converted_len = int(t * sr)

        if total_frames <= 0:
            return [0] * t

        if self.is_train:
            seg_len = int(total_frames)
            if seg_len <= converted_len:
                num = max(1, seg_len // sr)
                index = np.linspace(0, seg_len, num=num)
                if num < t:
                    index = np.concatenate((index, np.ones(t - num) * seg_len))
                index = np.clip(index, 0, max(seg_len - 1, 0)).astype(np.int64)
                return _pad_indices(index.tolist(), total_frames, t)

            # Match old PE train sampling: random end in [converted_len, seg_len)
            end_idx = int(np.random.randint(converted_len, seg_len))
            start_idx = int(end_idx - converted_len)
            index = np.linspace(start_idx, end_idx, num=t)
            index = np.clip(index, start_idx, max(end_idx - 1, start_idx)).astype(np.int64)
            return _pad_indices(index.tolist(), total_frames, t)

        # Eval branch matches old PE test/val chunking.
        if self.num_clips > 1:
            temporal_step = max(float(total_frames - converted_len) / float(self.num_clips - 1), 0.0)
            temporal_start = int(chunk_nb * temporal_step)
        else:
            temporal_start = max(0, (total_frames - converted_len) // 2)
        bound = min(temporal_start + converted_len, total_frames)
        idx = [x for x in range(temporal_start, bound, sr)]
        if not idx:
            idx = [max(0, min(temporal_start, total_frames - 1))]
        while len(idx) < t:
            idx.append(idx[-1])
        return _pad_indices(idx[:t], total_frames, t)

    def _legacy_pe_stride(self, total_frames: int) -> List[int]:
        """Current HMF legacy_pe behavior: fixed-interval stride sampling."""
        t = max(1, int(self.num_frames))
        sr = max(1, int(self.sampling_rate))
        converted_len = int(t * sr)

        if total_frames <= 0:
            return [0] * t

        if total_frames <= converted_len:
            idx = np.linspace(0, max(total_frames - 1, 0), t, dtype=np.int64).tolist()
            return _pad_indices(idx, total_frames, t)

        if self.is_train:
            max_start = max(0, total_frames - converted_len)
            start = random.randint(0, max_start)
        else:
            start = max(0, (total_frames - converted_len) // 2)

        bound = min(start + converted_len, total_frames)
        idx = [i for i in range(start, bound, sr)]
        if not idx:
            idx = [start]
        while len(idx) < t:
            idx.append(idx[-1])
        return _pad_indices(idx[:t], total_frames, t)

    # ====================================================================
    # Round 11: 8-frame VQA sampling strategies (training_schemes_8frame_vqa)
    # ====================================================================

    def _consecutive_random(self, total_frames: int) -> List[int]:
        """Scheme 1 (Baseline/L3): Random consecutive 8 frames.

        Randomly pick a start point s, take frames [s, s+1, ..., s+7].
        Each epoch sees a different random start → different local segment.
        Test: centered 8 consecutive frames (deterministic).
        """
        t = self.num_frames
        if total_frames <= t:
            return _pad_indices(list(range(total_frames)), total_frames, t)
        if self.is_train:
            start = random.randint(0, total_frames - t)
        else:
            start = max(0, (total_frames - t) // 2)
        indices = list(range(start, start + t))
        return _pad_indices(indices, total_frames, t)

    def _burst_4x2(self, total_frames: int) -> List[int]:
        """Scheme 2: 4 bins × 2 adjacent frames (4×2 burst sampling).

        Divide video into 4 equal time bins. In each bin, randomly pick
        an anchor and take 2 adjacent frames → 4×2 = 8 frames total.
        Preserves local continuity while covering the full video.
        Test: deterministic anchors at bin centers.
        """
        t = self.num_frames  # should be 8
        n_bins = 4
        frames_per_bin = 2
        if total_frames <= t:
            return _pad_indices(list(range(total_frames)), total_frames, t)

        bin_size = total_frames / n_bins
        indices = []
        for b in range(n_bins):
            bin_start = int(b * bin_size)
            bin_end = int((b + 1) * bin_size)
            # Need at least frames_per_bin frames in the bin
            max_anchor = max(bin_start, bin_end - frames_per_bin)
            if self.is_train:
                anchor = random.randint(bin_start, max_anchor)
            else:
                anchor = (bin_start + max_anchor) // 2
            for f in range(frames_per_bin):
                idx = min(anchor + f, total_frames - 1)
                indices.append(idx)
        return _pad_indices(indices, total_frames, t)

    def _segment_2x4(self, total_frames: int) -> List[int]:
        """Scheme 3: 2 bins × 4 adjacent frames (2×4 segment sampling).

        Divide video into 2 halves. In each half, randomly pick a start
        and take 4 consecutive frames → 2×4 = 8 frames total.
        Stronger local continuity than 4×2, with coarser global coverage.
        Test: deterministic starts at segment centers.
        """
        t = self.num_frames  # should be 8
        n_segments = 2
        frames_per_seg = 4
        if total_frames <= t:
            return _pad_indices(list(range(total_frames)), total_frames, t)

        seg_size = total_frames / n_segments
        indices = []
        for s in range(n_segments):
            seg_start = int(s * seg_size)
            seg_end = int((s + 1) * seg_size)
            max_start = max(seg_start, seg_end - frames_per_seg)
            if self.is_train:
                start = random.randint(seg_start, max_start)
            else:
                start = (seg_start + max_start) // 2
            for f in range(frames_per_seg):
                idx = min(start + f, total_frames - 1)
                indices.append(idx)
        return _pad_indices(indices, total_frames, t)

    def _mixed_sampler(self, total_frames: int) -> List[int]:
        """Scheme 4: Mixed sampling (L3 / 4×2 / 2×4 random switch).

        Each sample randomly selects one of three sampling modes:
          50% → consecutive_random (L3)
          30% → burst_4x2
          20% → segment_2x4
        Test: uses consecutive_random (deterministic baseline).
        """
        if not self.is_train:
            return self._consecutive_random(total_frames)
        r = random.random()
        if r < 0.5:
            return self._consecutive_random(total_frames)
        elif r < 0.8:
            return self._burst_4x2(total_frames)
        else:
            return self._segment_2x4(total_frames)

    # ====================================================================
    # Multi-Clip 4x8: 4 clips × 8 consecutive frames for cross-clip temporal
    # ====================================================================

    def _multi_clip_4x8(self, total_frames: int) -> List[List[int]]:
        """Multi-Clip 4×8: divide video into 4 segments, each clip takes 8 consecutive frames.

        Used for cross-clip temporal interaction experiments.
        Train: random start within each segment.
        Test: centered start within each segment.
        Returns 4 clips, each with 8 consecutive frame indices.
        """
        num_clips = 4
        frames_per_clip = self.num_frames  # should be 8
        clips = []

        if total_frames <= frames_per_clip:
            # Video is too short, all clips use the same frames (pad)
            base = _pad_indices(list(range(total_frames)), total_frames, frames_per_clip)
            return [list(base) for _ in range(num_clips)]

        seg_len = total_frames / num_clips

        for k in range(num_clips):
            seg_start = int(k * seg_len)
            seg_end = int((k + 1) * seg_len)
            seg_frames = seg_end - seg_start

            if seg_frames <= frames_per_clip:
                # Not enough frames within the segment, allow sampling across segments
                # Use segment center as anchor, expand outward on both sides
                center = (seg_start + seg_end) // 2
                start = max(0, center - frames_per_clip // 2)
                start = min(start, total_frames - frames_per_clip)
                start = max(0, start)
                indices = list(range(start, start + frames_per_clip))
            else:
                if self.is_train:
                    # Training: randomly pick start position within the segment
                    max_start = seg_end - frames_per_clip
                    start = random.randint(seg_start, max(seg_start, max_start))
                    indices = list(range(start, start + frames_per_clip))
                else:
                    # Testing: center-crop frames within the segment
                    center = (seg_start + seg_end) // 2
                    start = max(seg_start, center - frames_per_clip // 2)
                    # Ensure it does not exceed segment boundaries
                    start = min(start, seg_end - frames_per_clip)
                    start = max(seg_start, start)
                    indices = list(range(start, start + frames_per_clip))

            # Ensure indices are within the valid range
            indices = [min(max(idx, 0), total_frames - 1) for idx in indices]
            clips.append(_pad_indices(indices, total_frames, frames_per_clip))

        return clips

    def log_stats(self):
        """Log padding statistics."""
        if self._total_count > 0:
            logger.info(
                f"FrameSampler stats: {self._pad_count}/{self._total_count} "
                f"samples needed padding ({self.pad_ratio:.1%})"
            )
