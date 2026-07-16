"""
Internal evaluation engine invoked by ``infer.py``.

Not intended to be run directly — ``infer.py`` builds an argv list, sets a
few environment variables, and calls :func:`main` here to drive dataloader
construction, model loading, and per-video scoring.
"""

import os
import sys

# ---- Weight download location: project's weights/ folder ----
# This file lives at <project>/src/engine/eval_engine.py — walk up three
# levels to recover the project root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir))
_WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, 'weights')
os.makedirs(_WEIGHTS_DIR, exist_ok=True)
os.environ.setdefault('TORCH_HOME', _WEIGHTS_DIR)
os.environ.setdefault('HF_HOME', os.path.join(_WEIGHTS_DIR, 'huggingface'))

import json
import logging
import datetime
import time
import csv
from collections import Counter
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import build_config, merge_dict, DotDict, cfg_to_dotdict, DEFAULT_CONFIG, load_yaml
from src.utils.dist import init_distributed, is_main_process, get_world_size, get_rank
from src.utils.seed import set_seed
from src.data.datamodule import build_datasets, build_dataloaders
from src.data.datasets.base_dataset import (
    resolve_cvqad_cache_index_path,
    resolve_sampled_clip_cache_root,
)
from src.engine.evaluator import (
    evaluate, evaluate_cvqm_by_phase, save_inference_csv, print_timing_report,
    TTA_SCHEMES, get_tta_scheme, list_tta_schemes, list_tta_pooling_strategies,
)
from src.engine.ckpt import load_checkpoint
from src.engine.logger import get_logger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Helper functions (aligned with train.py)
# ---------------------------------------------------------------------------

def _get_active_align_cfg(cfg: dict):
    """Return (align_key, align_cfg_dict) for whichever align mode is active, or (None, {})."""
    for key in ('pe_align', 'swin_align', 'convnext_align'):
        sub = cfg.get(key, {})
        if isinstance(sub, dict) and bool(sub.get('enabled', False)):
            return key, sub
    return None, {}


def _get_semantic_branch_cfg(cfg: dict) -> dict:
    """Return semantic_branch config dict (always available, no 'enabled' gate)."""
    sub = cfg.get('semantic_branch', {})
    if isinstance(sub, dict):
        return sub
    return dict(sub) if sub else {}


def _freeze_repro_bypass_modules(model: nn.Module, cfg: dict):
    """Freeze FusionHead/Aggregator when semantic_branch uses repro_mlp head.

    These modules are bypassed in the forward pass (use_semantic_direct_score)
    and should not consume optimizer state or DDP gradient buckets.
    Must be called BEFORE DDP wrapping.
    """
    sem_cfg = _get_semantic_branch_cfg(cfg)
    if sem_cfg.get('head_mode', 'projection') != 'repro_mlp':
        return 0
    bypass_prefixes = ('fusion_head.', 'aggregator.')
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(bp) for bp in bypass_prefixes):
            param.requires_grad = False
            frozen_count += 1
    return frozen_count


def _setup_eval_rescale(cfg: dict, log):
    """Set HMF_EVAL_RESCALE env var, matching train.py logic exactly."""
    align_key, align_cfg = _get_active_align_cfg(cfg)
    align_enabled = align_key is not None
    sem_cfg = _get_semantic_branch_cfg(cfg)

    rescale_mode = 'zscore'  # default
    if align_enabled:
        rescale_mode = str(align_cfg.get('eval_rescale', 'zscore'))
    elif sem_cfg.get('eval_rescale', 'zscore') != 'zscore':
        rescale_mode = str(sem_cfg.get('eval_rescale', 'zscore'))

    os.environ['HMF_EVAL_RESCALE'] = rescale_mode
    return rescale_mode


def _load_config_from_exp(exp_dir: str, cli_args: list) -> dict:
    """Load config.json from an experiment directory and merge with CLI args.

    This ensures the model architecture matches the training run exactly.
    CLI args (e.g., --eval_ckpt, --output_dir) still take precedence.

    IMPORTANT: config.json saved by train.py uses ``json.dump(cfg, default=str)``
    which serializes ALL values as strings.  We need to recover the original types:
    - Nested dicts: "{'key': 'value'}" → dict (via ast.literal_eval)
    - Integers: "8" → 8
    - Floats: "1.2" → 1.2
    - Booleans: "True"/"False" → True/False
    - None: "None" → None
    """
    import ast
    import copy as _cp

    config_json_path = os.path.join(exp_dir, 'config.json')
    if not os.path.isfile(config_json_path):
        raise FileNotFoundError(
            f'config.json not found in experiment directory: {exp_dir}\n'
            f'Expected: {config_json_path}'
        )

    with open(config_json_path, 'r') as f:
        raw_cfg = json.load(f)

    def _recover_types(value):
        """Recover Python types from string serialization."""
        if not isinstance(value, str):
            return value
        # Try to parse as Python literal (handles dicts, lists, ints, floats, bools, None)
        try:
            parsed = ast.literal_eval(value)
            return parsed
        except (ValueError, SyntaxError):
            pass
        return value

    # Recover types for all values
    exp_cfg = {}
    for k, v in raw_cfg.items():
        recovered = _recover_types(v)
        # For nested dicts, recursively recover types of their values too
        if isinstance(recovered, dict):
            recovered = {sk: _recover_types(sv) if isinstance(sv, str) else sv
                         for sk, sv in recovered.items()}
        exp_cfg[k] = recovered

    # Remove training-only keys that conflict with eval
    _train_only_keys = {
        'output_dir', 'save_dir', 'run_signature', 'parent_dir_name',
        '_cli_overrides', 'batch_exp_name', 'resume',
    }
    for k in _train_only_keys:
        exp_cfg.pop(k, None)

    return exp_cfg


def _print_checkpoint_info(ckpt_path: str, checkpoint: dict, log):
    """Print checkpoint metadata for verification."""
    epoch = checkpoint.get('epoch', '?')
    best_metric_name = checkpoint.get('best_metric_name', '?')
    best_metric_value = checkpoint.get('best_metric_value', '?')
    dataset = checkpoint.get('dataset', '?')

    log.info(f'Checkpoint info:')
    log.info(f'  Path: {ckpt_path}')
    log.info(f'  Epoch: {epoch}')
    log.info(f'  Best metric: {best_metric_name} = {best_metric_value}')
    if dataset != '?':
        log.info(f'  Dataset: {dataset}')

    # File size
    try:
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        log.info(f'  File size: {size_mb:.1f} MB')
    except OSError:
        pass

    # Model keys summary
    model_sd = checkpoint.get('model', checkpoint.get('state_dict', {}))
    if model_sd:
        n_params = sum(v.numel() for v in model_sd.values() if hasattr(v, 'numel'))
        log.info(f'  Model parameters: {n_params:,} ({n_params/1e6:.1f}M)')


def _save_eval_summary(results: dict, output_dir: str, ckpt_path: str,
                       rescale_mode: str, cfg: dict, log):
    """Save comprehensive evaluation summary JSON."""
    summary = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': ckpt_path,
        'rescale_mode': rescale_mode,
        'task': cfg.get('task', '?'),
        'semantic_backbone': cfg.get('semantic_backbone', '?'),
        'semantic_fr_interaction': cfg.get('semantic_fr_interaction', '?'),
        'seed': cfg.get('seed', '?'),
        'results': {},
    }

    # Flatten results for easy reading
    for ds_name, ds_results in results.items():
        if isinstance(ds_results, dict):
            # Check if it's phase-separated (CVQM) or flat metrics
            first_val = next(iter(ds_results.values()), None)
            if isinstance(first_val, dict):
                # Phase-separated: {phase: {metric: value}}
                for phase, metrics in ds_results.items():
                    key = f'{ds_name}_{phase}'
                    summary['results'][key] = {
                        k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                        for k, v in metrics.items()
                    }
            else:
                # Flat metrics
                summary['results'][ds_name] = {
                    k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                    for k, v in ds_results.items()
                }

    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f'Evaluation summary saved: {results_path}')
    return summary


def _infer_batch_size(batch: dict) -> int:
    if not isinstance(batch, dict):
        return 0
    for key in ('mos', 'num_clips', 'height', 'width'):
        val = batch.get(key)
        if isinstance(val, torch.Tensor) and val.numel() > 0:
            return int(val.shape[0])
    vids = batch.get('video_id')
    if isinstance(vids, list):
        return len(vids)
    return 0


def _prebuild_loader_cache(loaders: dict, split_name: str, dry_run: bool, log) -> dict:
    summary = {}
    for name, loader in loaders.items():
        num_batches = len(loader)
        batch_count = 0
        sample_count = 0
        start_t = time.time()
        iterator = loader
        if is_main_process() and tqdm is not None:
            iterator = tqdm(
                loader,
                desc=f'Cache [{split_name}:{name}]',
                unit='batch',
                total=num_batches,
                dynamic_ncols=True,
                leave=False,
            )
        for batch_idx, batch in enumerate(iterator):
            if dry_run and batch_idx >= 1:
                break
            batch_count += 1
            sample_count += _infer_batch_size(batch)
            if is_main_process() and tqdm is None and (batch_count % 10 == 0 or batch_count == num_batches):
                log.info(
                    '[CacheOnly] %s[%s]: %d/%d batches, approx_samples=%d',
                    split_name, name, batch_count, num_batches, sample_count,
                )
        elapsed = time.time() - start_t
        summary[name] = {
            'batches': int(batch_count),
            'approx_samples': int(sample_count),
            'elapsed_sec': round(float(elapsed), 3),
        }
        if is_main_process():
            log.info(
                '[CacheOnly] Done %s[%s]: batches=%d approx_samples=%d elapsed=%.2fs',
                split_name, name, batch_count, sample_count, elapsed,
            )
    return summary


def _collect_dataset_samples(ds) -> list:
    if ds is None:
        return []
    if hasattr(ds, 'samples'):
        return list(getattr(ds, 'samples'))
    if hasattr(ds, 'datasets'):
        out = []
        for child in list(getattr(ds, 'datasets', [])):
            out.extend(_collect_dataset_samples(child))
        return out
    return []


def _collect_cvqad_cache_index_rows(ds) -> list:
    if ds is None:
        return []
    build_fn = getattr(ds, 'build_sampled_clip_cache_index_rows', None)
    if callable(build_fn):
        return list(build_fn())
    if hasattr(ds, 'datasets'):
        out = []
        for child in list(getattr(ds, 'datasets', [])):
            out.extend(_collect_cvqad_cache_index_rows(child))
        return out
    return []


def _export_cvqac_manifest(loaders: dict, cfg: dict, output_dir: str, log) -> dict:
    rows = []
    for _name, loader in dict(loaders or {}).items():
        dataset = getattr(loader, 'dataset', None)
        for sample in _collect_dataset_samples(dataset):
            if str(getattr(sample, 'dataset_name', '') or '').upper() not in ('CVQAD', 'CVQAC'):
                continue
            extra = dict(getattr(sample, 'extra', {}) or {})
            rows.append({
                'dataset_name': str(getattr(sample, 'dataset_name', '') or ''),
                'source_split': str(extra.get('source_split', getattr(sample, 'split', '')) or ''),
                'video_id': str(getattr(sample, 'video_id', '') or ''),
                'sequence': str(extra.get('sequence', '') or ''),
                'preset': str(extra.get('preset', '') or ''),
                'codec': str(extra.get('codec', '') or ''),
                'crf': str(extra.get('crf', '') or ''),
                'mos': float(getattr(sample, 'mos', 0.0)),
                'raw_mos': float(extra.get('raw_mos', 0.0)),
                'dis_path': str(getattr(sample, 'dis_path', '') or ''),
                'ref_path': str(getattr(sample, 'ref_path', '') or ''),
                'dis_bitdepth': int(getattr(sample, 'bitdepth', 8) or 8),
                'dis_pix_fmt': str(getattr(sample, 'pix_fmt', '') or ''),
                'ref_bitdepth': int(getattr(sample, 'ref_bitdepth', 8) or 8),
                'ref_pix_fmt': str(getattr(sample, 'ref_pix_fmt', '') or ''),
                'ref_actual_bitdepth': int(extra.get('ref_actual_bitdepth', 8) or 8),
                'ref_actual_pix_fmt': str(extra.get('ref_actual_pix_fmt', '') or ''),
            })

    if not rows:
        return {}

    rows.sort(key=lambda x: (x['source_split'], x['sequence'], x['preset'], x['codec'], x['crf']))

    cache_root = resolve_sampled_clip_cache_root(cfg.get('sampled_clip_cache_root', None), 'CVQAD')
    export_root = cache_root or output_dir
    os.makedirs(export_root, exist_ok=True)

    csv_path = os.path.join(export_root, 'cvqac_labels_all_splits.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    split_counter = Counter(row['source_split'] for row in rows)
    codec_counter = Counter(row['codec'] for row in rows)
    stats = {
        'total_distorted_samples': int(len(rows)),
        'total_unique_reference_sequences': int(len({row['ref_path'] for row in rows if row['ref_path']})),
        'total_unique_sequences': int(len({row['sequence'] for row in rows if row['sequence']})),
        'split_counts': dict(sorted(split_counter.items())),
        'codec_counts': dict(sorted(codec_counter.items())),
        'total_vvenc_samples': int(sum(1 for row in rows if row['codec'] == 'vvenc')),
        'total_10bit_distorted': int(sum(1 for row in rows if int(row['dis_bitdepth']) > 8)),
        'total_ref_aligned_to_10bit': int(sum(1 for row in rows if int(row['ref_bitdepth']) > 8)),
    }
    stats_path = os.path.join(export_root, 'cvqac_split_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    log.info('CVQAC manifest saved: %s', csv_path)
    log.info('CVQAC stats saved: %s', stats_path)
    return {
        'manifest_csv': csv_path,
        'stats_json': stats_path,
        **stats,
    }


def _export_cvqac_cache_index(loaders: dict, cfg: dict, output_dir: str, log) -> dict:
    rows = []
    for _name, loader in dict(loaders or {}).items():
        dataset = getattr(loader, 'dataset', None)
        rows.extend(_collect_cvqad_cache_index_rows(dataset))

    if not rows:
        return {}

    rows.sort(key=lambda x: (x.get('source_split', ''), x.get('sequence', ''), x.get('preset', ''), x.get('codec', ''), x.get('crf', '')))
    cache_root = resolve_sampled_clip_cache_root(cfg.get('sampled_clip_cache_root', None), 'CVQAD')
    export_path = resolve_cvqad_cache_index_path(
        cache_root,
        cfg.get('cvqad_cache_index_path', None),
        'CVQAD',
    )
    export_root = os.path.dirname(export_path) if export_path else (cache_root or output_dir)
    if export_root and os.path.isdir(export_root) and not os.access(export_root, os.W_OK):
        fallback_root = output_dir
        os.makedirs(fallback_root, exist_ok=True)
        export_path = os.path.join(fallback_root, 'cvqac_cache_index_all_splits.csv')
        export_root = fallback_root
        log.warning(
            'CVQAC offline cache index target is not writable, fallback to output_dir: %s',
            export_path,
        )
    os.makedirs(export_root, exist_ok=True)
    if not export_path:
        export_path = os.path.join(export_root, 'cvqac_cache_index_all_splits.csv')

    fieldnames = list(rows[0].keys())
    with open(export_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info('CVQAC offline cache index saved: %s', export_path)
    return {
        'cache_index_csv': export_path,
        'total_index_rows': int(len(rows)),
    }


def main():
    # ---- Handle eval-specific args before build_config ----
    # These args are not in argparse (eval-only) so we intercept them manually.
    config_from_exp = None
    exp_cfg = None
    eval_output_dir = None

    # Parse eval-specific args from sys.argv manually (before argparse)
    _argv = list(sys.argv[1:])

    def _pop_arg(argv, flag):
        """Remove --flag VALUE from argv list, return VALUE or None."""
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                val = argv[i + 1]
                argv.pop(i)  # remove flag
                argv.pop(i)  # remove value
                return val
        return None

    config_from_exp = _pop_arg(_argv, '--config_from_exp')
    eval_output_dir = _pop_arg(_argv, '--output_dir')

    # TTA parameters (eval-only, intercepted before build_config)
    tta_scheme_name = _pop_arg(_argv, '--tta_scheme')
    tta_pooling_arg = _pop_arg(_argv, '--tta_pooling')
    tta_hflip_arg = _pop_arg(_argv, '--tta_hflip')
    tta_num_clips_arg = _pop_arg(_argv, '--num_clips_test')

    # Rescale mode override (eval-only CLI arg)
    eval_rescale_arg = _pop_arg(_argv, '--eval_rescale')

    if config_from_exp:
        # Load experiment config.json
        exp_cfg = _load_config_from_exp(config_from_exp, _argv)

    # ---- Build config (YAML + CLI) ----
    import copy  # noqa: used below for deepcopy
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + _argv
    try:
        cfg = build_config()
    finally:
        sys.argv = original_argv

    # If experiment config loaded, merge it UNDER the CLI args
    # Priority: CLI args > experiment config.json > YAML config > defaults
    if exp_cfg is not None:
        # We need to merge exp_cfg into cfg, but CLI args should still win.
        # Strategy: the build_config() already handled YAML + CLI.
        # If no --config was specified in CLI, we use exp_cfg as the base.
        # If --config WAS specified, the YAML already won, and we skip exp_cfg.
        has_config_flag = '--config' in _argv
        if not has_config_flag:
            # No YAML config specified — use experiment config.json as base
            # Re-merge: DEFAULT → exp_cfg → CLI
            base = copy.deepcopy(DEFAULT_CONFIG)
            base = merge_dict(base, exp_cfg)
            # Now apply CLI overrides on top.
            # cli_overrides contains FLAT keys (e.g. 'pe_align_container_yuv_direct')
            # but build_config() already applied them into cfg via nested paths.
            # So we apply both: (1) flat keys directly, (2) nested dicts from cfg.
            cli_overrides = cfg.get('_cli_overrides', {})
            for k, v in cli_overrides.items():
                if k in base:
                    base[k] = v
            # Also merge nested dicts that may have been updated by CLI overrides.
            # build_config() wrote e.g. cfg['pe_align']['container_yuv_direct'] = True
            # but the flat-key loop above missed it because 'pe_align_container_yuv_direct'
            # is not a key in base (base uses nested 'pe_align' dict).
            for nest_key in ('pe_align', 'swin_align', 'convnext_align',
                             'semantic_branch', 'detail_branch'):
                if nest_key in cfg and isinstance(cfg[nest_key], dict):
                    if nest_key not in base or not isinstance(base.get(nest_key), dict):
                        base[nest_key] = {}
                    # Only override keys that were explicitly set via CLI
                    cfg_sub = cfg[nest_key]
                    base_sub = base[nest_key]
                    for sk, sv in cfg_sub.items():
                        # Check if this nested key was actually a CLI override
                        flat_key = f'{nest_key}_{sk}'
                        if flat_key in cli_overrides:
                            base_sub[sk] = sv
            # Preserve essential eval keys from cfg
            for ek in ('eval_ckpt', 'eval_ckpt_strict', 'eval_stage',
                       'output_dir', 'dry_run', 'amp', 'num_workers',
                       'test_batch_mul', 'val_batch_mul', 'frame_cache_root'):
                if ek in cfg and cfg[ek] is not None:
                    base[ek] = cfg[ek]
            cfg = cfg_to_dotdict(base)

    # Evaluation does not require train split to exist.
    cfg['allow_empty_train'] = True

    # Apply eval-specific output_dir if provided
    if eval_output_dir:
        cfg['output_dir'] = eval_output_dir

    # ``num_clips_test`` comes from the yaml (5 for CVQM by default) and
    # is overridable via the standard --num_clips_test CLI flag.

    # ---- Setup HMF_EVAL_RESCALE (MUST be before any evaluation) ----
    # Use a temporary logger for early setup
    _early_log = logging.getLogger('eval_setup')

    rescale_mode = _setup_eval_rescale(cfg, _early_log)

    if eval_rescale_arg:
        rescale_mode = eval_rescale_arg.strip().lower()
        os.environ['HMF_EVAL_RESCALE'] = rescale_mode

    # ---- Distributed ----
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    # ---- Seed ----
    set_seed(cfg.get('seed', 43))

    # ---- Logger ----
    output_dir = cfg.get('output_dir', 'output/eval')
    os.makedirs(output_dir, exist_ok=True)
    _log_file = None if os.environ.get('HMF_VQA_NO_LOG_FILE', '') == '1' \
        else os.path.join(output_dir, 'eval.log')
    log = get_logger('eval', _log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_main_process():
        log.info(f'Device: {device}, World size: {world_size}')
        log.info(f'Task: {cfg.get("task", "?")}, Backbone: {cfg.get("semantic_backbone", "?")}')
        if cfg.get('task', '').lower() == 'fr':
            log.info(f'FR interaction: {cfg.get("semantic_fr_interaction", "?")}')
        if config_from_exp:
            log.info(f'Config loaded from experiment: {config_from_exp}')

        # Save the final merged config for reference
        config_dump_path = os.path.join(output_dir, 'eval_config.json')
        eval_config_dump = dict(cfg)
        # Attach TTA CLI args for reference (full resolution happens later)
        eval_config_dump['_tta_scheme_arg'] = tta_scheme_name or ''
        eval_config_dump['_tta_pooling_arg'] = tta_pooling_arg or ''
        eval_config_dump['_tta_hflip_arg'] = tta_hflip_arg or ''
        eval_config_dump['_tta_num_clips_arg'] = tta_num_clips_arg or ''
        with open(config_dump_path, 'w') as f:
            json.dump(eval_config_dump, f, indent=2, default=str)
        log.info(f'Config saved: {config_dump_path}')

    # ---- Data ----
    train_ds, val_datasets, test_datasets = build_datasets(cfg)
    _, val_loaders, test_loaders = build_dataloaders(
        cfg, None, val_datasets, test_datasets, rank, world_size,
    )

    cache_only = bool(cfg.get('cache_only', False))
    export_cvqad_cache_index_only = bool(cfg.get('export_cvqad_cache_index_only', False))
    dry_run = cfg.get('dry_run', False)
    if export_cvqad_cache_index_only:
        if is_main_process():
            log.info('=' * 60)
            log.info('CVQAD cache-index export-only mode enabled.')
            log.info(f'test datasets: {list(test_loaders.keys())}')
            log.info('=' * 60)
        summary = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'output_dir': output_dir,
        }
        cvqac_cache_index = _export_cvqac_cache_index(test_loaders, cfg, output_dir, log) if is_main_process() else {}
        if cvqac_cache_index:
            summary['cvqac_cache_index'] = cvqac_cache_index
        if is_main_process():
            summary_path = os.path.join(output_dir, 'cvqad_cache_index_export_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            log.info(f'CVQAD cache-index export summary saved: {summary_path}')
            log.info('CVQAD cache-index export-only mode complete.')
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    if cache_only:
        if is_main_process():
            log.info('=' * 60)
            log.info('Cache-only mode enabled: iterating dataloaders to build data caches.')
            log.info(f'dry_run: {dry_run}, test datasets: {list(test_loaders.keys())}')
            log.info('=' * 60)
        summary = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'output_dir': output_dir,
            'val': _prebuild_loader_cache(val_loaders, 'val', dry_run, log),
            'test': _prebuild_loader_cache(test_loaders, 'test', dry_run, log),
        }
        cvqac_manifest = _export_cvqac_manifest(test_loaders, cfg, output_dir, log) if is_main_process() else {}
        if cvqac_manifest:
            summary['cvqac_manifest'] = cvqac_manifest
        cvqac_cache_index = _export_cvqac_cache_index(test_loaders, cfg, output_dir, log) if is_main_process() else {}
        if cvqac_cache_index:
            summary['cvqac_cache_index'] = cvqac_cache_index
        if is_main_process():
            summary_path = os.path.join(output_dir, 'cache_only_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            log.info(f'Cache-only summary saved: {summary_path}')
            log.info('Cache-only mode complete.')
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    # ---- Model ----
    from src.models import build_model
    model = build_model(cfg)
    model = model.to(device)

    # Freeze FusionHead/Aggregator for repro_mlp head (same as train.py)
    frozen = _freeze_repro_bypass_modules(model, cfg)
    if is_main_process() and frozen > 0:
        log.info(f'Frozen {frozen} repro_bypass parameters (FusionHead/Aggregator)')

    # Load checkpoint
    ckpt_path = cfg.get('eval_ckpt', None)
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt_strict = bool(cfg.get('eval_ckpt_strict', True))
        checkpoint = load_checkpoint(ckpt_path, model, strict=ckpt_strict)
        # Checkpoint metadata printing intentionally suppressed in this release.
    else:
        if is_main_process():
            log.warning(f'No checkpoint specified or not found: {ckpt_path}')
            log.warning('Running evaluation with random/pretrained weights!')

    # DDP
    if dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=cfg.get('find_unused_parameters', True),
        )

    use_amp = cfg.get('amp', True)
    eval_use_only_vif_branch = bool(cfg.get('eval_use_only_vif_branch', False))
    enable_timing = bool(cfg.get('enable_timing', True))  # Default ON for the engine
    timing_warmup_batches = int(cfg.get('timing_warmup_batches', 2) or 0)

    # ---- Resolve TTA settings ----
    tta_hflip = False
    tta_pooling = 'mean'
    tta_pool_kwargs = {}

    if tta_scheme_name:
        # Named TTA scheme overrides individual settings
        scheme = get_tta_scheme(tta_scheme_name)
        tta_hflip = scheme['tta_hflip']
        tta_pooling = scheme['pooling']
        tta_pool_kwargs = dict(scheme.get('pool_kwargs', {}))
        num_clips_override = scheme['num_clips']
        # Override num_clips in config to force data pipeline to sample the right number
        cfg['num_clips_test'] = num_clips_override
        cfg['test_eval_mode'] = 'full'  # force 'full' to use num_clips_test
        if is_main_process():
            log.info(f'TTA scheme: {tta_scheme_name} → {scheme["description"]}')
            log.info(f'  num_clips={num_clips_override}, tta_hflip={tta_hflip}, '
                     f'pooling={tta_pooling}, pool_kwargs={tta_pool_kwargs}')
    else:
        # Individual TTA overrides
        if tta_hflip_arg is not None:
            tta_hflip = tta_hflip_arg.lower() in ('1', 'true', 'yes')
        if tta_pooling_arg is not None:
            tta_pooling = tta_pooling_arg
        if tta_num_clips_arg is not None:
            cfg['num_clips_test'] = int(tta_num_clips_arg)
            cfg['test_eval_mode'] = 'full'

    # If TTA scheme changed num_clips, rebuild data loaders to get the right clip count
    if tta_scheme_name or tta_num_clips_arg:
        if is_main_process():
            log.info(f'Rebuilding data loaders with num_clips_test={cfg.get("num_clips_test")} '
                     f'test_eval_mode={cfg.get("test_eval_mode")}')
        train_ds, val_datasets, test_datasets = build_datasets(cfg)
        _, val_loaders, test_loaders = build_dataloaders(
            cfg, None, val_datasets, test_datasets, rank, world_size,
        )

    all_results = {}
    all_timing_stats = {}

    # Re-save config with resolved TTA settings
    if is_main_process():
        config_dump_path = os.path.join(output_dir, 'eval_config.json')
        eval_config_dump = dict(cfg)
        eval_config_dump['_tta_scheme'] = tta_scheme_name or ''
        eval_config_dump['_tta_pooling'] = tta_pooling
        eval_config_dump['_tta_hflip'] = tta_hflip
        eval_config_dump['_tta_pool_kwargs'] = tta_pool_kwargs
        with open(config_dump_path, 'w') as f:
            json.dump(eval_config_dump, f, indent=2, default=str)

    if is_main_process():
        log.info('=' * 60)
        log.info('Starting evaluation...')
        log.info(f'AMP: {use_amp}, dry_run: {dry_run}, timing: {enable_timing}')
        log.info(f'TTA: pooling={tta_pooling}, hflip={tta_hflip}, pool_kwargs={tta_pool_kwargs}')
        log.info(f'Test datasets: {list(test_loaders.keys())}')
        log.info('=' * 60)

    # ---- Evaluate test datasets ----
    for name, loader in test_loaders.items():
        if name.upper() == 'CVQM':
            results, per_video = evaluate_cvqm_by_phase(
                model,
                loader,
                device,
                use_amp=use_amp,
                dry_run=dry_run,
                use_only_vif_branch=eval_use_only_vif_branch,
                tta_hflip=tta_hflip,
                enable_timing=enable_timing,
                tta_pooling=tta_pooling,
                tta_pool_kwargs=tta_pool_kwargs,
                timing_warmup_batches=timing_warmup_batches,
            )
            # Extract timing_stats from the 'all' phase metrics (set by evaluate())
            _cvqm_all_metrics = results.get('all', {})
            _timing_stats = _cvqm_all_metrics.pop('timing_stats', None)
            if _timing_stats:
                all_timing_stats[f'test_{name}'] = _timing_stats

            all_results[f'test_{name}'] = results
            if is_main_process():
                log.info(f'{"=" * 60}')
                log.info(f'Test {name} (Phase-Based) Results:')
                log.info(f'{"=" * 60}')
                for phase, metrics in results.items():
                    srcc = metrics.get('SRCC', 0.0)
                    plcc = metrics.get('PLCC', 0.0)
                    krcc = metrics.get('KRCC', 0.0)
                    rmse = metrics.get('RMSE', 0.0)
                    log.info(f'  {phase:8s}: SRCC={srcc:.4f}  PLCC={plcc:.4f}  '
                             f'KRCC={krcc:.4f}  RMSE={rmse:.4f}')
                # Score = SRCC + PLCC (same as training best-model criterion)
                all_srcc = results.get('all', {}).get('SRCC', 0.0)
                all_plcc = results.get('all', {}).get('PLCC', 0.0)
                log.info(f'  {"score":8s}: SRCC+PLCC = {all_srcc + all_plcc:.4f} (all-phase)')
                p1_srcc = results.get('phase1', {}).get('SRCC', 0.0)
                p1_plcc = results.get('phase1', {}).get('PLCC', 0.0)
                log.info(f'  {"score":8s}: SRCC+PLCC = {p1_srcc + p1_plcc:.4f} (phase1)')

                log.info(f'{"-" * 60}')

                # Save per-video CSV with metrics header
                save_inference_csv(
                    per_video, output_dir, f'test_{name}_results.csv',
                    metrics=results.get('all', {}),
                    phase_metrics=results,
                )

                # Save phase-separated metrics JSON
                metrics_json_path = os.path.join(output_dir, f'test_{name}_metrics.json')
                with open(metrics_json_path, 'w') as f:
                    json.dump({
                        k: {mk: round(float(mv), 6) for mk, mv in v.items()}
                        for k, v in results.items()
                    }, f, indent=2)
                log.info(f'Phase metrics saved: {metrics_json_path}')

                # Save timing JSON
                if _timing_stats:
                    timing_json_path = os.path.join(output_dir, f'test_{name}_timing.json')
                    with open(timing_json_path, 'w') as f:
                        json.dump(_timing_stats, f, indent=2, default=str)
                    log.info(f'Timing stats saved: {timing_json_path}')

                # Generate scatter plots per phase
                from src.engine.plotter import plot_scatter
                for phase_name in ['all', 'phase1', 'phase2']:
                    phase_data = [r for r in per_video
                                  if phase_name == 'all' or
                                  r.get('stage') == int(phase_name[-1])]
                    if len(phase_data) < 3:
                        continue
                    preds = np.array([r['pred_rescaled'] for r in phase_data])
                    targets = np.array([r['mos'] for r in phase_data])
                    phase_metrics = results.get(phase_name, {})
                    try:
                        plot_scatter(
                            preds, targets,
                            output_dir=output_dir,
                            filename=f'scatter_{name}_{phase_name}.png',
                            title=f'{name} {phase_name} '
                                  f'(SRCC={phase_metrics.get("SRCC", 0):.4f}, '
                                  f'PLCC={phase_metrics.get("PLCC", 0):.4f})',
                        )
                    except Exception as e:
                        log.warning(f'Failed to plot scatter for {phase_name}: {e}')

        else:
            metrics, per_video = evaluate(
                model, loader, device, use_amp=use_amp, dry_run=dry_run,
                dataset_name=name,
                use_only_vif_branch=eval_use_only_vif_branch,
                tta_hflip=tta_hflip,
                enable_timing=enable_timing,
                tta_pooling=tta_pooling,
                tta_pool_kwargs=tta_pool_kwargs,
                timing_warmup_batches=timing_warmup_batches,
            )
            # Extract timing_stats
            _timing_stats = metrics.pop('timing_stats', None)
            if _timing_stats:
                all_timing_stats[f'test_{name}'] = _timing_stats

            all_results[f'test_{name}'] = metrics
            if is_main_process():
                log.info(f'=== Test {name} ===')
                log.info(f'  SRCC={metrics.get("SRCC", 0):.4f}  '
                         f'PLCC={metrics.get("PLCC", 0):.4f}  '
                         f'KRCC={metrics.get("KRCC", 0):.4f}  '
                         f'RMSE={metrics.get("RMSE", 0):.4f}')
                save_inference_csv(per_video, output_dir, f'test_{name}_results.csv')
                if _timing_stats:
                    timing_json_path = os.path.join(output_dir, f'test_{name}_timing.json')
                    with open(timing_json_path, 'w') as f:
                        json.dump(_timing_stats, f, indent=2, default=str)
                    log.info(f'Timing stats saved: {timing_json_path}')

    # ---- Evaluate val datasets (optional) ----
    for name, loader in val_loaders.items():
        metrics, per_video = evaluate(
            model, loader, device, use_amp=use_amp, dry_run=dry_run,
            dataset_name=name,
            use_only_vif_branch=eval_use_only_vif_branch,
            tta_hflip=tta_hflip,
            enable_timing=enable_timing,
            tta_pooling=tta_pooling,
            tta_pool_kwargs=tta_pool_kwargs,
            timing_warmup_batches=timing_warmup_batches,
        )
        _timing_stats = metrics.pop('timing_stats', None)
        if _timing_stats:
            all_timing_stats[f'val_{name}'] = _timing_stats

        all_results[f'val_{name}'] = metrics
        if is_main_process():
            log.info(f'=== Val {name} ===')
            log.info(f'  {metrics}')
            save_inference_csv(per_video, output_dir, f'val_{name}_results.csv')

    # Save comprehensive summary
    if is_main_process():
        _save_eval_summary(all_results, output_dir, ckpt_path or '',
                           rescale_mode, cfg, log)

        # Save aggregated timing summary
        if all_timing_stats:
            timing_summary_path = os.path.join(output_dir, 'timing_summary.json')
            with open(timing_summary_path, 'w') as f:
                json.dump(all_timing_stats, f, indent=2, default=str)
            log.info(f'Timing summary saved: {timing_summary_path}')

        log.info('=' * 60)
        log.info('Evaluation complete.')
        log.info(f'All results saved to: {output_dir}')
        log.info('=' * 60)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
