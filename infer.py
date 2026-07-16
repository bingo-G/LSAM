"""
infer.py — LSAM full-reference video-quality-assessment inference.

Three input modes are supported (pick one):

  1. ``--ref FILE  --dis FILE``           (single pair)
  2. ``--ref_dir DIR  --dis_dir DIR``      (batch, paired by basename / sequence)
  3. ``--manifest FILE.csv``               (batch, explicit pairing in CSV)


Timing profiler is off by default; add ``--timing`` (optionally with
``--timing_repeats N``) to opt in.

Usage
-----
Single pair:
    python infer.py --ref /data/refs/BasketballDrive.yuv \\
                    --dis /data/dis/BasketballDrive_QP22.yuv \\
                    --width 1920 --height 1080 --bitdepth 10 \\
                    --eval_ckpt weights/lsam.pth

Batch (paired by basename / sequence name):
    python infer.py --ref_dir /data/refs --dis_dir /data/dis \\
                    --eval_ckpt weights/lsam.pth --output scores.csv

Batch (explicit manifest CSV):
    python infer.py --manifest pairs.csv --eval_ckpt weights/lsam.pth --output scores.csv

Multi-GPU (large batches):
    torchrun --nproc_per_node=4 infer.py --manifest pairs.csv \\
        --eval_ckpt weights/lsam.pth --output scores.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


_CUSTOM_NAME = 'Custom'
_DEFAULT_CONFIG = 'configs/lsam.yaml'
_VIDEO_EXTENSIONS = ('.yuv', '.mp4', '.mkv', '.y4m', '.webm', '.mov')


# ---------------------------------------------------------------------------
# Manifest / pair discovery
# ---------------------------------------------------------------------------

def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTENSIONS


def _walk_videos(root: str) -> List[str]:
    """Recursively walk ``root`` and return all video files (relative to root)."""
    if not os.path.isdir(root):
        raise SystemExit(f"Directory not found: {root}")
    out: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if _is_video(fn):
                full = os.path.join(dirpath, fn)
                out.append(os.path.relpath(full, root))
    return sorted(out)


def _extract_sequence(dis_basename: str) -> str:
    """Heuristic: strip a trailing ``_QP<NN>`` marker from a dis filename.

    Examples:
        ``HM/BasketballDrive_QP22.yuv``        → ``BasketballDrive``
        ``VVENC/Tango_QP37.mp4``               → ``Tango``
        ``Bus_1920x1080_30fps.yuv``            → ``Bus_1920x1080_30fps``
    """
    name = os.path.basename(dis_basename)
    stem, _ = os.path.splitext(name)
    stem = re.sub(r'_QP\d+$', '', stem, flags=re.IGNORECASE)
    return stem


def _build_ref_index(ref_dir: str) -> Dict[str, str]:
    """Map ``sequence_stem`` → absolute ref path.

    First tries an exact stem match (basename without extension), then a
    prefix match against the common reference-naming convention where ref
    filenames usually carry extra resolution / fps tags
    (e.g. ``BasketballDrive_1920x1080_50fps_10bit.yuv``).
    """
    index: Dict[str, str] = {}
    prefix_index: Dict[str, str] = {}
    for rel in _walk_videos(ref_dir):
        stem, _ = os.path.splitext(os.path.basename(rel))
        index[stem] = os.path.join(ref_dir, rel)
        # Use the leading token (before first '_') as a coarse prefix key.
        head = stem.split('_', 1)[0]
        prefix_index.setdefault(head, os.path.join(ref_dir, rel))
        # Also index by the full leading "sequence_<...>" up to first numeric token
        m = re.match(r'^([A-Za-z][A-Za-z0-9]+)', stem)
        if m and m.group(1) not in prefix_index:
            prefix_index[m.group(1)] = os.path.join(ref_dir, rel)
    # Merge: prefix index entries are fallbacks
    for k, v in prefix_index.items():
        index.setdefault(k, v)
    return index


def _pair_dirs(ref_dir: str, dis_dir: str) -> List[dict]:
    """Pair every video under ``dis_dir`` with the best-matching ref."""
    ref_index = _build_ref_index(ref_dir)
    rows: List[dict] = []
    missed = 0
    for rel in _walk_videos(dis_dir):
        dis_path = os.path.join(dis_dir, rel)
        seq = _extract_sequence(rel)
        ref_path = ref_index.get(seq)
        if ref_path is None:
            # try prefix
            head = seq.split('_', 1)[0]
            ref_path = ref_index.get(head)
        if ref_path is None:
            missed += 1
            print(f'[infer.py] WARN: no reference match for {rel} (sequence={seq})',
                  file=sys.stderr)
            continue
        rows.append({
            'dis_path': dis_path,
            'ref_path': ref_path,
            'video_id': rel,
        })
    if not rows:
        raise SystemExit(
            f'No (ref, dis) pairs could be built from ref_dir={ref_dir}, dis_dir={dis_dir}.'
        )
    print(f'[infer.py] Paired {len(rows)} videos '
          f'({missed} skipped for missing reference) from {dis_dir}',
          file=sys.stderr)
    return rows


def _load_manifest(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('dis_path'):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"manifest '{path}' produced 0 rows")
    return rows


def _single_pair(args: argparse.Namespace) -> List[dict]:
    if not args.dis or not args.ref:
        raise SystemExit('Single-pair mode requires both --ref and --dis')
    row = {
        'dis_path': args.dis,
        'ref_path': args.ref,
        'video_id': args.video_id or os.path.basename(args.dis),
    }
    if args.width:
        row['width'] = str(args.width)
    if args.height:
        row['height'] = str(args.height)
    if args.bitdepth:
        row['bitdepth'] = str(args.bitdepth)
    if args.pixel_format:
        row['pixel_format'] = args.pixel_format
    if args.mos is not None:
        row['mos'] = str(args.mos)
    return [row]


# ---------------------------------------------------------------------------
# Custom dataset registration (in-memory)
# ---------------------------------------------------------------------------

def _register_custom_dataset(rows: List[dict]) -> None:
    from src.data.datasets import PARSERS
    from src.data.datasets.base_dataset import DATASET_CONFIGS, SampleMeta

    DATASET_CONFIGS[_CUSTOM_NAME] = {
        'root': '', 'train_ann': '', 'val_ann': '', 'ref_root': '',
    }

    def _parse_custom(mode: str = 'test') -> list:
        samples = []
        for i, row in enumerate(rows):
            dis_path = str(row.get('dis_path', '')).strip()
            ref_path = str(row.get('ref_path', '')).strip() or None
            video_id = str(row.get('video_id', '')).strip() or os.path.basename(dis_path)

            def _opt_int(key: str, default: Optional[int] = None) -> Optional[int]:
                v = row.get(key)
                if v is None or v == '' or v == 'None':
                    return default
                try:
                    return int(float(v))
                except (TypeError, ValueError):
                    return default

            width = _opt_int('width', None)
            height = _opt_int('height', None)
            bitdepth = _opt_int('bitdepth', 10)
            pix_fmt = str(row.get('pixel_format', 'yuv420p10le') or 'yuv420p10le').strip()
            try:
                mos = float(row.get('mos', 0.0) or 0.0)
            except (TypeError, ValueError):
                mos = 0.0

            samples.append(SampleMeta(
                dataset_name=_CUSTOM_NAME,
                video_id=video_id,
                ref_path=ref_path,
                dis_path=dis_path,
                width=width,
                height=height,
                bitdepth=bitdepth,
                pix_fmt=pix_fmt,
                mos=mos,
                extra={'row_index': i},
                split='test',
            ))
        return samples

    PARSERS[_CUSTOM_NAME] = _parse_custom


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _find_results_csv(output_dir: str) -> Optional[str]:
    candidates = sorted(
        glob.glob(os.path.join(output_dir, f'test_{_CUSTOM_NAME}_results.csv'))
        + glob.glob(os.path.join(output_dir, 'test_*_results.csv')),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _postprocess_and_report(src_csv: str, out_csv: str,
                            emit_timing: bool = False,
                            num_clips: int = 1,
                            num_frames_per_clip: int = 8) -> None:
    import json as _json
    import numpy as np
    from src.utils.score_postprocess import apply_score_mapping, compute_correlations

    with open(src_csv, newline='') as f:
        clean = [ln for ln in f if not ln.startswith('#')]
    rows = list(csv.DictReader(clean))
    if not rows:
        print(f'[infer.py] WARNING: results CSV is empty: {src_csv}', file=sys.stderr)
        return

    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Dedupe replicated timing rows (video_id contains "__repeatN"). Keep the
    # first occurrence for CSV output — score is deterministic so pick any.
    def _strip_repeat_tag(vid: str) -> str:
        idx = vid.rfind('__repeat')
        return vid[:idx] if idx >= 0 else vid

    seen_ids = set()
    dedup_rows = []
    for r in rows:
        vid = _strip_repeat_tag(str(r.get('video_id', '')))
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        r_out = dict(r)
        r_out['video_id'] = vid
        dedup_rows.append(r_out)

    model_out = np.array([_f(r.get('pred_raw', r.get('pred_score', 0.0))) for r in dedup_rows])
    mos = np.array([_f(r.get('mos', 0.0)) for r in dedup_rows])
    heights = [int(_f(r.get('orig_height', r.get('height', 0)) or 0)) for r in dedup_rows]
    has_mos = bool(np.any(mos != 0))

    pred = apply_score_mapping(model_out, heights=heights)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
    fields = ['video_id', 'mos', 'pred', 'height', 'width']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(dedup_rows):
            w.writerow({
                'video_id': r.get('video_id', ''),
                'mos': f'{float(mos[i]):.4f}',
                'pred': f'{float(pred[i]):.6f}',
                'height': r.get('height', ''),
                'width': r.get('width', ''),
            })

    print('')
    print('═' * 60)
    print(f'[infer.py] {len(dedup_rows)} unique video(s) processed  '
          f'({len(rows) - len(dedup_rows)} timing-replica rows deduped)')
    print(f'[infer.py] Output:   {os.path.abspath(out_csv)}')

    if has_mos and len(dedup_rows) >= 2:
        m = compute_correlations(pred, mos)
        print('')
        print(f'  Correlations vs MOS (n={len(dedup_rows)}):')
        for k in ('SRCC', 'PLCC', 'KRCC', 'RMSE'):
            print(f'    {k:<6s}  {m.get(k, float("nan")):>9.4f}')
    elif not has_mos:
        print('  (no MOS supplied → correlations skipped)')
    print('═' * 60)

    out_dir = os.path.dirname(os.path.abspath(out_csv)) or '.'

    # ── Optional timing report (diagnostic — off by default). ──
    if emit_timing:
        _print_timing_report(out_dir, dedup_rows, num_clips, num_frames_per_clip)

    # ── Clean up intermediate files. Keep timing_summary.json when the
    # caller asked for diagnostics; otherwise everything but the deliverable
    # CSV is removed. ──
    junk = [
        os.path.join(out_dir, 'eval_config.json'),
        os.path.join(out_dir, 'eval_results.json'),
        os.path.join(out_dir, f'test_{_CUSTOM_NAME}_results.csv'),
        os.path.join(out_dir, f'test_{_CUSTOM_NAME}_timing.json'),
    ]
    if not emit_timing:
        junk.append(os.path.join(out_dir, 'timing_summary.json'))
    for p in junk:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def _print_timing_report(out_dir: str, rows: list,
                         num_clips: int, num_frames_per_clip: int) -> None:
    """Print a concise METHOD timing report (English) to stdout.

    "Method" = data preprocess + model forward, i.e. the per-frame cost of the
    algorithm itself. It EXCLUDES disk IO, YUV decode and H2D copies.

    Numbers are computed from the wall-clock section of ``timing_summary.json``
    divided by the number of *processed* frames (= videos x clips x frames),
    which is robust to the multi-clip accounting.
    """
    import json as _json
    raw_path = os.path.join(out_dir, 'timing_summary.json')
    if not os.path.isfile(raw_path):
        print('[infer.py] (no timing_summary.json produced — timing report skipped)',
              file=sys.stderr)
        return
    try:
        with open(raw_path) as f:
            raw = _json.load(f)
    except (OSError, ValueError) as e:
        print(f'[infer.py] failed to read timing_summary.json: {e}', file=sys.stderr)
        return
    stats = next(iter(raw.values())) if isinstance(raw, dict) and raw else raw
    if not isinstance(stats, dict):
        return

    K = max(1, int(num_clips))
    T = max(1, int(num_frames_per_clip))
    wc  = stats.get('wall_clock', {}) or {}
    by_res = stats.get('by_resolution', {}) or stats.get('by_decoded_resolution', {}) or {}

    def _f(v, d=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return d

    n_v = int(stats.get('num_videos', len(rows)))
    sampled   = int(stats.get('total_frames', n_v * T))
    processed = max(1, sampled * K)

    gp_sec  = _f(wc.get('gpu_preprocess_sec'))   # GPU preprocess (HMF_VQA_GPU_PREPROCESS=2)
    pf_sec  = _f(wc.get('pure_forward_sec'))     # pure model(...) forward, excludes H2D
    cpu_pre = _f(wc.get('preprocess_sec'))       # CPU preprocess (fallback when GPU prep off)
    prep_sec   = gp_sec if gp_sec > 1e-9 else cpu_pre
    prep_where = 'GPU' if gp_sec > 1e-9 else 'CPU'
    method_sec = prep_sec + pf_sec

    def _pf_ms(sec):  # per processed frame, in ms
        return sec / processed * 1000.0
    def _fps(sec):
        return (processed / sec) if sec > 1e-9 else 0.0

    print('')
    print('=' * 70)
    print('  METHOD TIMING  (data preprocess + model forward)')
    print('  Per-frame cost of the algorithm itself.')
    print('  Excludes disk IO, YUV decode and H2D copies.')
    print('=' * 70)
    print(f'  Data preprocess ({prep_where})     : {_pf_ms(prep_sec):8.3f} ms/frame')
    print(f'  Model forward             : {_pf_ms(pf_sec):8.3f} ms/frame')
    print(f'  {"-" * 56}')
    print(f'  Method (preproc+forward)  : {_pf_ms(method_sec):8.3f} ms/frame'
          f'   ->  {_fps(method_sec):7.2f} FPS')

    # Per-resolution breakdown (only meaningful when inputs span >1 resolution).
    if len(by_res) >= 1:
        KT = K * T
        print('')
        print(f'  {"Resolution":<10s}{"n":>4s}  {"preproc ms":>12s}'
              f'{"forward ms":>12s}{"method ms":>12s}{"FPS":>9s}')
        print(f'  {"-"*10}{"-"*4}  {"-"*12}{"-"*12}{"-"*12}{"-"*9}')
        for res, e in sorted(by_res.items()):
            if not isinstance(e, dict):
                continue
            n_i = int(e.get('num_videos', 0))
            gp_ms = _f(e.get('avg_gpu_preprocess_ms'))       # per video
            pf_ms = _f(e.get('avg_pure_forward_ms'))         # per video
            pre_pf = (gp_ms if gp_ms > 1e-9 else _f(e.get('avg_preprocess_ms'))) / KT
            fwd_pf = pf_ms / KT
            meth_pf = pre_pf + fwd_pf
            fps_r = (1000.0 / meth_pf) if meth_pf > 1e-9 else 0.0
            print(f'  {res:<10s}{n_i:>4d}  {pre_pf:>12.3f}{fwd_pf:>12.3f}'
                  f'{meth_pf:>12.3f}{fps_r:>9.1f}')
    print('=' * 70)
    print(f'  Raw timing JSON: {os.path.abspath(raw_path)}')
    print('=' * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='LSAM full-reference VQA inference')
    p.add_argument('--config', type=str, default=_DEFAULT_CONFIG,
                   help=f'Model YAML (default: {_DEFAULT_CONFIG}).')
    p.add_argument('--eval_ckpt', type=str, required=True,
                   help='Path to the LSAM checkpoint (e.g. weights/lsam.pth).')

    # Single-pair
    p.add_argument('--ref', type=str, help='Reference video path (single-pair mode).')
    p.add_argument('--dis', type=str, help='Distorted video path (single-pair mode).')
    p.add_argument('--video_id', type=str, help='Optional id; defaults to basename(dis).')
    p.add_argument('--width', type=int)
    p.add_argument('--height', type=int)
    p.add_argument('--bitdepth', type=int)
    p.add_argument('--pixel_format', type=str)
    p.add_argument('--mos', type=float, help='Optional MOS for the single pair.')

    # Batch-dir
    p.add_argument('--ref_dir', type=str, help='Directory of reference videos (batch).')
    p.add_argument('--dis_dir', type=str, help='Directory of distorted videos (batch).')

    # Manifest
    p.add_argument('--manifest', type=str,
                   help='CSV cols: dis_path[, ref_path][, width][, height][, bitdepth]'
                        '[, pixel_format][, video_id][, mos]')

    # Output
    p.add_argument('--output', type=str, default='outputs/inference_results.csv',
                   help='Output CSV path (default: outputs/inference_results.csv).')

    # Forwarded
    p.add_argument('--num_clips_test', type=int, default=None,
                   help='Override num_clips_test (default 5, set in configs/lsam.yaml).')
    p.add_argument('--tta_scheme', type=str, default=None,
                   help='Optional TTA scheme name (e.g. M1a_clip5_mean).')
    p.add_argument('--batch_size', type=int, default=1,
                   help='DataLoader batch size (default 1 — single-pair friendly).')
    p.add_argument('--num_workers', type=int, default=2,
                   help='DataLoader worker count (default 2).')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--cpu', action='store_true')
    # ── Diagnostics (off by default) ──
    p.add_argument('--timing', action='store_true',
                   help='Emit a fine-grained wall-clock / FPS breakdown to stdout '
                        'and keep timing_summary.json in the output directory. '
                        'Off by default — deliverable artifacts stay silent.')
    p.add_argument('--timing_repeats', type=int, default=3,
                   help='(with --timing) Replicate each input row this many times so '
                        'cuDNN autotune + CUDA context init on the first call do not '
                        'pollute the average. Default 3 — the first repeat is treated '
                        'as warmup and its timing is dropped.')
    p.add_argument('--timing_warmup_batches', type=int, default=None,
                   help='(with --timing) Batches to exclude from timing averages. '
                        'Defaults to timing_repeats-1 repeats worth of the input '
                        '(so exactly one clean warmup pass is discarded).')
    return p


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = _build_parser()
    args, extra_argv = parser.parse_known_args()

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.manifest:
        rows = _load_manifest(args.manifest)
    elif args.ref_dir and args.dis_dir:
        rows = _pair_dirs(args.ref_dir, args.dis_dir)
    else:
        rows = _single_pair(args)

    # ── Timing-mode replication ─────────────────────────────────────────
    # A single-video run has exactly one model call, which means cuDNN
    # autotune + CUDA context init all land on that call and dominate
    # `pure_forward_sec`. Replicate the input rows so the first pass acts
    # as a warmup that is later dropped from the timing average.
    original_row_count = len(rows)
    repeats = 1
    if args.timing and int(args.timing_repeats) > 1:
        repeats = int(args.timing_repeats)
        replicated: List[dict] = []
        for r_idx in range(repeats):
            for r in rows:
                copy = dict(r)
                base_id = copy.get('video_id') or os.path.basename(copy.get('dis_path', ''))
                copy['video_id'] = f'{base_id}__repeat{r_idx}'
                replicated.append(copy)
        rows = replicated

    _register_custom_dataset(rows)

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Disable the internal per-batch rescale — the final score is produced by
    # _postprocess_and_report() from the raw model outputs.
    os.environ.setdefault('HMF_EVAL_RESCALE', 'none')
    os.environ.setdefault('HMF_VQA_NO_LOG_FILE', '1')
    # Default to GPU preprocessing (mode 2): resize_dis is rebuilt on the GPU
    # (UV upsample + resize + patch gather), so the reported timing reflects the
    # fast path WITHOUT any extra flag. This is numerically equivalent to the CPU
    # path (same torch ops / params) and produces identical scores for .yuv input
    # with the shipped lsam config (legacy_pe + gmsavg + stack).
    # Override with `HMF_VQA_GPU_PREPROCESS=0` to force the legacy CPU preprocess.
    os.environ.setdefault('HMF_VQA_GPU_PREPROCESS', '2')

    forwarded = [
        sys.argv[0],
        '--config', args.config,
        '--eval_ckpt', args.eval_ckpt,
        '--test_dataset', _CUSTOM_NAME,
        '--dataset_path_fail_fast', '0',
        '--output_dir', output_dir,
        '--eval_rescale', 'none',
        # Released checkpoint contains the full model — no separate PE weights needed.
        '--pretrained', '0',
        '--train_datasets', '',
        '--batch_size', str(max(1, int(args.batch_size))),
        '--num_workers', str(max(0, int(args.num_workers))),
        '--enable_timing', '1' if args.timing else '0',
    ]
    # Auto-derive warmup: skip the first replicated pass entirely (one full
    # sweep through original_row_count videos) so cuDNN autotune doesn't
    # skew the average. User can override with --timing_warmup_batches.
    if args.timing_warmup_batches is not None:
        _warmup = max(0, int(args.timing_warmup_batches))
    elif args.timing and repeats > 1:
        # batch_size=1 assumed for single-pair; ceil-divide for larger batch.
        bs = max(1, int(args.batch_size))
        _warmup = (original_row_count + bs - 1) // bs
    else:
        _warmup = 0
    forwarded += ['--timing_warmup_batches', str(_warmup)]
    if args.amp and not args.no_amp:
        forwarded.append('--amp')
    if args.num_clips_test is not None:
        forwarded += ['--num_clips_test', str(args.num_clips_test)]
    if args.tta_scheme:
        forwarded += ['--tta_scheme', args.tta_scheme]
    forwarded += extra_argv

    sys.argv = forwarded
    from src.engine.eval_engine import main as eval_main
    eval_main()

    src_csv = _find_results_csv(output_dir)
    if src_csv is None:
        print(f'[infer.py] WARNING: no results CSV found under {output_dir}', file=sys.stderr)
        return 1

    # Derive clip config for the timing report before eval_config.json gets deleted.
    num_clips_used, num_frames_used = 1, 8
    _cfg_path = os.path.join(output_dir, 'eval_config.json')
    if os.path.isfile(_cfg_path):
        try:
            import json as _json
            _cfg = _json.load(open(_cfg_path))
            num_clips_used = int(_cfg.get('num_clips_test', 1) or 1)
            num_frames_used = int(_cfg.get('num_frames', 8) or 8)
        except (OSError, ValueError, TypeError):
            pass

    _postprocess_and_report(src_csv, output_path,
                            emit_timing=bool(args.timing),
                            num_clips=num_clips_used,
                            num_frames_per_clip=num_frames_used)
    return 0


if __name__ == '__main__':
    sys.exit(main())
