"""
CVQM dataset parser. Adapted from legacy (standalone, no imports from old code).
Supports two-phase evaluation and phase filtering.
"""

import os
import logging
import pandas as pd
from typing import List, Optional

from .base_dataset import SampleMeta, get_dataset_config

logger = logging.getLogger('hmf_vqa.cvqm')


def parse_cvqm(mode: str = 'test', cvqm_phase: str = 'all') -> List[SampleMeta]:
    """
    Parse CVQM dataset labels.

    Args:
        mode: 'val' or 'test'
        cvqm_phase: '1', '2', or 'all'

    Returns:
        List of SampleMeta
    """
    cfg = get_dataset_config('CVQM')
    root = cfg['root']
    ann_key = 'test_ann' if mode == 'test' else 'val_ann'
    ann_file = cfg.get(ann_key, cfg.get('test_ann'))
    path_to_file = os.path.join(root, ann_file)

    if not os.path.exists(path_to_file):
        logger.warning(f"CVQM annotation not found: {path_to_file}")
        return []

    df = pd.read_excel(path_to_file)

    # Phase filtering
    cvqm_phase = str(cvqm_phase)  # ensure string comparison (YAML may parse '1' as int)
    if cvqm_phase in ('1', 'phase1') and 'Phase' in df.columns:
        df = df[df['Phase'] == 1]
        logger.info(f"[CVQM] Phase1 only: {len(df)} samples")
    elif cvqm_phase in ('2', 'phase2') and 'Phase' in df.columns:
        df = df[df['Phase'] == 2]
        logger.info(f"[CVQM] Phase2 only: {len(df)} samples")
    else:
        if 'Phase' in df.columns:
            counts = df['Phase'].value_counts().to_dict()
            logger.info(f"[CVQM] All phases: {len(df)} samples ({counts})")

    phase_roots = cfg.get('phase_roots', {})
    ref_root = cfg.get('ref_root', None)

    # Pre-load reference files
    ref_files = []
    if ref_root and os.path.exists(ref_root):
        ref_files = os.listdir(ref_root)

    samples = []
    for _, row in df.iterrows():
        key = str(row['Key'])
        phase_raw = row.get('Phase', 1)
        try:
            phase = int(float(phase_raw))
        except Exception:
            ptxt = str(phase_raw).strip().lower()
            phase = 2 if '2' in ptxt else 1
        if phase not in (1, 2):
            phase = 1

        # Determine video path prefix based on phase
        prefix = phase_roots.get(phase, root)
        dis_path = os.path.join(prefix, key)

        # MOS
        mos = float(row.get('MOS', row.get('mos', 0)))
        label = mos / 10.0  # Normalize to [0, 1]

        # Resolution
        width = int(row['width']) if 'width' in row and pd.notna(row.get('width')) else 1920
        height = int(row['height']) if 'height' in row and pd.notna(row.get('height')) else 1080
        bit_depth = 10
        if 'bit depth' in row and pd.notna(row.get('bit depth')):
            bit_depth = int(row['bit depth'])
        elif 'bit_depth' in row and pd.notna(row.get('bit_depth')):
            bit_depth = int(row['bit_depth'])

        # Reference path
        sequence = str(row.get('Sequence', row.get('sequence', '')))
        ref_path = None
        if sequence and ref_files:
            for f in ref_files:
                if f == sequence + '.yuv' or f == sequence + '.mp4':
                    ref_path = os.path.join(ref_root, f)
                    break
                if f.startswith(sequence + '_'):
                    ref_path = os.path.join(ref_root, f)
                    break

        # Optional VMAF supervision from annotation (kept in raw 0~100 scale).
        vmaf_target = None
        for key_vmaf in ('VMAF', 'vmaf', 'Vmaf'):
            if key_vmaf in row and pd.notna(row.get(key_vmaf)):
                try:
                    vmaf_target = float(row[key_vmaf])
                except Exception:
                    vmaf_target = None
                break

        extra = {'phase': phase, 'sequence': sequence, 'raw_mos': mos}
        if sequence:
            extra['content_id'] = sequence
        if vmaf_target is not None:
            extra['vmaf_target'] = vmaf_target

        samples.append(SampleMeta(
            dataset_name='CVQM',
            video_id=key,
            ref_path=ref_path,
            dis_path=dis_path,
            width=width,
            height=height,
            bitdepth=bit_depth,
            pix_fmt='yuv420p10le' if bit_depth == 10 else 'yuv420p',
            mos=label,
            extra=extra,
            split=mode,
            stage=phase,
        ))

    if samples:
        p1 = sum(1 for s in samples if s.stage == 1)
        p2 = sum(1 for s in samples if s.stage == 2)
        miss_ref = sum(1 for s in samples if not s.ref_path)
        logger.info(
            "[CVQM] Parsed %d samples (stage1=%d, stage2=%d, missing_ref=%d)",
            len(samples), p1, p2, miss_ref,
        )

    return samples


def split_cvqm_by_stage(samples: List[SampleMeta]) -> dict:
    """Split CVQM samples by stage (phase) for two-stage evaluation."""
    stage1 = [s for s in samples if s.stage == 1]
    stage2 = [s for s in samples if s.stage == 2]
    return {'stage1': stage1, 'stage2': stage2, 'all': samples}
