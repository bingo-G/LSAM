"""
Configuration system: YAML + CLI override.
Supports loading from YAML, merging CLI args, and dumping final config.
"""

import os
import copy
import yaml
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional


def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def save_yaml(cfg: dict, path: str):
    """Save config dict to YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def merge_dict(base: dict, override: dict) -> dict:
    """Deep-merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def build_run_signature(cfg: dict, *, cli_overrides: dict = None) -> str:
    """Generate a *short* run signature: only non-default CLI overrides + timestamp.

    New format:  ``<key1>=<val1>__<key2>=<val2>__<YYYYMMDD_HHMMSS>``
    If no overrides were given, falls back to ``default__<YYYYMMDD_HHMMSS>``.
    Full config is always dumped to ``config.json`` in the output directory.

    Args:
        cfg: Final merged config dict.
        cli_overrides: Dict of ``{key: value}`` for parameters explicitly set on the
            command line (or by the bash mode presets).  Only these appear in the
            directory name.  If *None*, falls back to legacy full-signature mode.
    """
    import datetime
    import hashlib
    import os
    import re

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if cli_overrides is not None:
        # ---- New short-signature mode ----
        # Skip keys that are purely structural / don't affect reproducibility
        _skip_keys = {
            'config', 'save_dir', 'output_dir', 'run_signature',
            'num_workers', 'log_interval', 'eval_interval',
            'test_eval_interval', 'debug_save_frames',
            'debug_num_videos_train', 'debug_num_videos_test',
            'debug_num_frames', 'debug_every_epochs',
            'dry_run', 'cache_only', 'resume',
            # Path-type args: containing '/' would create subfolders, and they do not affect training reproducibility
            'frame_cache_root', 'sampled_clip_cache_root',
        }
        parts = []
        for k, v in sorted(cli_overrides.items()):
            if k in _skip_keys:
                continue
            # Shorten common key names for readability
            short = _SHORT_KEY_MAP.get(k, k)
            parts.append(f"{short}={v}")

        if not parts:
            sig = f"default__{ts}"
        else:
            sig = '__'.join(parts) + f'__{ts}'
    else:
        # ---- Legacy full-signature mode (backward compat) ----
        parts = []
        parts.append(f"task={cfg.get('task', 'fr').lower()}")
        parts.append(f"br={cfg.get('branches', 'detail,semantic')}")
        parts.append(f"vif={cfg.get('vif_mode', 'late_fuse')}")
        vifb = cfg.get('vif_branch', {})
        if isinstance(vifb, dict):
            parts.append(f"vifb={vifb.get('mode', 'aligned')}")
            parts.append(f"vsrc={vifb.get('feature_source', 'libvmaf')}")
            parts.append(f"vsm={vifb.get('score_mode', 'learned')}")
            parts.append(f"vsf={vifb.get('score_fusion', 'none')}")
        parts.append(f"cs={cfg.get('colorspace', 'bt709_imagenet')}")
        parts.append(f"det={cfg.get('detail_sampler', 'gms')}")
        parts.append(f"sem={cfg.get('semantic_sampler', 'resize224')}")
        parts.append(f"tsam={cfg.get('temporal_sampler', 'multiclip')}")
        parts.append(f"sr={cfg.get('sampling_rate', 1)}")
        parts.append(f"T={cfg.get('num_frames', 8)}")
        parts.append(f"Ktr={cfg.get('num_clips_train', 2)}")
        parts.append(f"Kte={cfg.get('num_clips_test', 4)}")
        parts.append(f"tmp={cfg.get('temporal', 'tsm')}")
        parts.append(f"rape={cfg.get('rape', 'on')}")
        parts.append(f"st={cfg.get('scale_token', 'on')}")
        losses = cfg.get('losses', 'mse,rank,plcc,fidelity')
        parts.append(f"loss={losses}")
        parts.append(f"tgt={cfg.get('target_label_key', 'mos')}")
        parts.append(f"lm={cfg.get('lambda_mse', 1.0)}")
        parts.append(f"lr={cfg.get('lambda_rank', 0.1)}")
        parts.append(f"lp={cfg.get('lambda_plcc', 0.1)}")
        parts.append(f"lf={cfg.get('lambda_fidelity', 0.1)}")
        parts.append(f"rm={cfg.get('rank_margin', 0.05)}")
        fupic = cfg.get('fupic_tile', 224)
        fupic_s = cfg.get('fupic_stride', 192)
        parts.append(f"fupic={fupic}x{fupic_s}")
        pe_align = cfg.get('pe_align', {})
        if isinstance(pe_align, dict) and pe_align.get('enabled', False):
            parts.append("pea=on")
            parts.append(f"peam={pe_align.get('model_variant', 'semantic_nr')}")
            parts.append(f"pear={pe_align.get('eval_rescale', 'zscore')}")
        sig = '__'.join(parts) + f'__{ts}'

    # Filesystem safety: a single path component is typically limited to 255 bytes.
    max_len = int(os.getenv('HMF_RUNSIG_MAXLEN', '180'))
    if len(sig) <= max_len:
        return sig

    hash10 = hashlib.sha1(sig.encode('utf-8')).hexdigest()[:10]
    m = re.search(r'__\d{8}_\d{6}$', sig)
    ts_suffix = m.group(0) if m else ''
    core = sig[:-len(ts_suffix)] if ts_suffix else sig
    tail = f"__h={hash10}{ts_suffix}"
    keep = max(24, max_len - len(tail))
    return core[:keep] + tail


# Short names for common CLI override keys (used in run_signature directory name)
_SHORT_KEY_MAP = {
    'task': 'task',
    'train_datasets': 'data',
    'test_dataset': 'test',
    'cvqm_phase': 'phase',
    'branches': 'br',
    'detail_backbone': 'dbb',
    'semantic_backbone': 'bb',
    'semantic_fr_interaction': 'inter',
    'fr_interaction': 'dinter',
    'detail_sampler': 'dsam',
    'semantic_sampler': 'sem',
    'lr_backbone': 'lrbb',
    'lr_head': 'lrh',
    'batch_size': 'bs',
    'grad_accum': 'ga',
    'epochs': 'ep',
    'seed': 'seed',
    'sampling_rate': 'sr',
    'num_frames': 'T',
    'losses': 'loss',
    'target_label_key': 'tgt',
    'lambda_mse': 'lm',
    'lambda_rank': 'lr',
    'lambda_plcc': 'lp',
    'weight_decay': 'wd',
    'clip_grad': 'cg',
    'colorspace': 'cs',
    'tenbit_mode': 'tb',
    'fusion_type': 'fuse',
    'res_conditioner': 'rc',
    'res_conditioner_hidden': 'rch',
    'rope_resolution_scale': 'rope',
    'patch_embed_cond': 'pec',
    'resolution_token': 'rtok',
    # ---- Round 8 new parameters ----
    'gradient_topk_sampling': 'gradtk',
    'gradient_topk_mode': 'gtkm',
    'stochastic_depth_rate': 'sdr',
    'feature_mixup_alpha': 'fmix',
    'label_smooth_sigma': 'lsm',
    'spectral_decoupling_weight': 'sdw',
    'feature_whitening': 'fwh',
    'curriculum_learning': 'curri',
    'curriculum_warmup_epochs': 'cwep',
    # ---- Round 13 new parameters ----
    'ema_decay': 'ema',
    'fidelity_buffer_mode': 'fbuf',
    'legacy_spatial_cross_batch_accum': 'lscba',
    # ---- Round 9 new parameters ----
    'temporal_adaptor': 'tadp',
    'freeze_backbone': 'fzbb',
    'freeze_fr_modules': 'fzfr',
    'freeze_head': 'fzh',
    'cross_clip_temporal': 'cct',
    'pretrain_path': 'ptpath',
}


def build_parent_dir_name(cfg: dict) -> str:
    """Build a hierarchical parent directory name from task/model/dataset.

    Format: ``{task}_{branches}_{backbone}_{interaction}_{dataset}``

    When branches == 'semantic' (single branch, default), branches is omitted
    for backward compatibility.  Multi-branch examples include the branches list.

    Examples:
        - ``FR_pe_b16_topiq_deep_CVQM_1080p``            (single semantic, default)
        - ``NR_pe_b16_CVQM_whole``                        (single semantic NR)
        - ``FR_sem+det_pe_b16_topiq_deep_CVQM_mixed``    (multi-branch)
        - ``FR_sem+det+vif_pe_b16_topiq_deep_CVQM_1080p``  (3-branch)
    """
    task = cfg.get('task', 'fr').upper()
    bb = cfg.get('semantic_backbone', 'pe_b16')
    datasets = cfg.get('train_datasets', 'CVQM_self')

    # Parse enabled branches
    branches_raw = cfg.get('branches', 'semantic')
    if isinstance(branches_raw, str):
        br_list = sorted([b.strip().lower() for b in branches_raw.split(',') if b.strip()])
    else:
        br_list = sorted([str(b).strip().lower() for b in branches_raw])

    # Short branch names for directory
    _BR_SHORT = {'semantic': 'sem', 'detail': 'det', 'vif': 'vif'}
    br_str = '+'.join(_BR_SHORT.get(b, b) for b in br_list)
    is_single_semantic = (br_list == ['semantic'])

    # Shorten dataset name for directory (longest match first!)
    ds_short = datasets.replace('CVQM_self_whole,CVQM_self_4K,CVQM_self_4K_next', 'CVQM_all') \
                       .replace('CVQM_self_whole,CVQM_self_4K_next', 'CVQM_whole+4Knxt') \
                       .replace('CVQM_self,CVQM_self_4K_next', 'CVQM_mixed_nxt') \
                       .replace('CVQM_self,CVQM_self_4K', 'CVQM_mixed') \
                       .replace('CVQM_self_whole,CVQM_self_4K', 'CVQM_whole+4K') \
                       .replace('CVQM_self_whole', 'CVQM_whole') \
                       .replace('CVQM_self_4K_next', 'CVQM_4Knxt') \
                       .replace('CVQM_self_4K', 'CVQM_4K') \
                       .replace('CVQM_self', 'CVQM_1080p')

    # Build name parts
    parts = [task]
    if not is_single_semantic:
        parts.append(br_str)
    parts.append(bb)
    if task == 'FR':
        inter = cfg.get('semantic_fr_interaction', 'topiq_deep')
        parts.append(inter)
    parts.append(ds_short)
    return '_'.join(parts)


def get_arg_parser() -> argparse.ArgumentParser:
    """Build the full CLI argument parser."""
    p = argparse.ArgumentParser(description='HMF-VQA Training / Evaluation')

    # Config file
    p.add_argument('--config', type=str, default=None, help='Path to YAML config')

    # Task / Data
    p.add_argument('--task', type=str, default=None, choices=['fr', 'nr'])
    p.add_argument('--train_datasets', type=str, default=None,
                   help='Comma-separated train dataset names')
    p.add_argument('--test_dataset', type=str, default=None)
    p.add_argument('--test_dataset_mode', type=str, default=None,
                   help='Evaluation split selector for non-CVQM datasets '
                        '(e.g. test / all / dataset-specific modes).')
    p.add_argument('--dataset_path_fail_fast', type=int, default=None, choices=[0, 1],
                   help='Validate dataset root/annotation/reference layout before parsing')
    p.add_argument('--fr_strict_reference', type=int, default=None, choices=[0, 1],
                   help='In FR mode, raise error on missing or nonexistent ref/dis paths')
    p.add_argument('--cache_only', type=int, default=None, choices=[0, 1],
                   help='Build datasets (and VIF cache prebuild) then exit before training')
    p.add_argument('--export_cvqad_cache_index_only', type=int, default=None, choices=[0, 1],
                   help='Export CVQAD/CVQAC offline sampled-clip cache index CSV only, '
                        'without iterating dataloaders or rebuilding caches.')
    p.add_argument('--cvqad_cache_index_path', type=str, default=None,
                   help='Output path for exported CVQAD/CVQAC offline cache index CSV.')
    p.add_argument('--dry_run', action='store_true', help='Use mock data for smoke test')

    # Branches
    p.add_argument('--branches', type=str, default=None,
                   help='Comma-separated branch names: vif,detail,semantic')
    p.add_argument('--vif_mode', type=str, default=None,
                   choices=['off', 'late_fuse', 'inject'])
    p.add_argument('--vif_impl', type=str, default=None, choices=['legacy', 'vmaf_like'])
    p.add_argument('--vif_branch_enable', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_branch_mode', type=str, default=None, choices=['aligned', 'dense'])
    p.add_argument('--vif_align_with_other_branches', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_max_dense_frames', type=int, default=None)
    p.add_argument('--vif_use_grad_ratio', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_use_lap_ratio', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_use_ti', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_use_adm', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_use_motion', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_embed_dim', type=int, default=None)
    p.add_argument('--vif_regressor_type', type=str, default=None, choices=['linear_svr', 'mlp', 'mlp_residual'])
    p.add_argument('--vif_regressor_hidden_dim', type=int, default=None)
    p.add_argument('--vif_score_transform', type=str, default=None, choices=['sigmoid', 'tanh', 'linear'])
    p.add_argument('--vif_temporal_pool', type=str, default=None,
                   choices=['mean', 'mean_p10', 'p10', 'hmean', 'weighted_mix', 'learned_mix', 'mean_min'])
    p.add_argument('--vif_temporal_mean_weight', type=float, default=None)
    p.add_argument('--vif_temporal_p10_weight', type=float, default=None)
    p.add_argument('--vif_temporal_hmean_weight', type=float, default=None)
    p.add_argument('--vif_num_scales', type=int, default=None)
    p.add_argument('--vif_window_size', type=int, default=None)
    p.add_argument('--vif_quantile_pool_size', type=int, default=None)
    p.add_argument('--vif_dropout', type=float, default=None)
    p.add_argument('--vif_norm_momentum', type=float, default=None)
    p.add_argument('--vif_feature_norm_affine', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_score_range', type=float, default=None)
    p.add_argument('--vif_prior_blend', type=float, default=None)
    p.add_argument('--vif_score_mode', type=str, default=None,
                   choices=['learned', 'vmaf_prior', 'residual', 'external_prior', 'external_residual', 'external_blend'])
    p.add_argument('--vif_prior_residual_scale', type=float, default=None)
    p.add_argument('--vif_learnable_prior_residual_scale', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_learnable_prior_blend', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_use_in_fusion', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_residual_fusion', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_score_fusion', type=str, default=None,
                   choices=['none', 'residual', 'weighted_sum', 'vif_only'])
    p.add_argument('--vif_score_fusion_alpha', type=float, default=None)
    p.add_argument('--vif_cache_features', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_cache_force_rebuild', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_cache_prebuild', type=int, default=None, choices=[0, 1])
    p.add_argument('--vif_cache_dir', type=str, default=None)
    p.add_argument('--vif_cache_partition_total', type=int, default=None)
    p.add_argument('--vif_cache_partition_index', type=int, default=None)
    p.add_argument('--vif_cache_partition_on_remaining', type=int, default=None, choices=[0, 1],
                   help='Partition prebuild based on remaining cache misses instead of all videos')
    p.add_argument('--vif_cache_require_vmaf_score', type=int, default=None, choices=[0, 1],
                   help='Require libvmaf frame scores in cache; rebuild old cache without this field')
    p.add_argument('--vif_cache_device', type=str, default=None, choices=['cpu', 'cuda', 'auto'],
                   help='Device for VIF cache prebuild feature extraction')
    p.add_argument('--vif_feature_source', type=str, default=None, choices=['libvmaf'],
                   help='VIF feature extraction backend (libvmaf only)')
    p.add_argument('--vif_ffmpeg_bin', type=str, default=None,
                   help='ffmpeg binary path used by libvmaf backend')
    p.add_argument('--vif_libvmaf_use_cuda', type=int, default=None, choices=[0, 1],
                   help='Deprecated compatibility flag. libvmaf backend always runs on CPU.')
    p.add_argument('--vif_libvmaf_n_threads', type=int, default=None,
                   help='n_threads for libvmaf filter')
    p.add_argument('--vif_libvmaf_model', type=str, default=None,
                   help='Optional libvmaf model override string')
    p.add_argument('--vif_vmaf_model_dir', type=str, default=None,
                   help='Directory containing vmaf_v0.6.1.json and vmaf_4k_v0.6.1.json '
                        'for cache-time SVR reproduction')

    # ColorSpace
    p.add_argument('--colorspace', type=str, default=None,
                   choices=['bt709_imagenet', 'bt601_imagenet',
                            'bt709_clip', 'bt601_clip',
                            'bt709_raw', 'bt601_raw'])

    # Sampling
    p.add_argument('--num_frames', type=int, default=None)
    p.add_argument('--sampling_rate', type=int, default=None,
                   help='Temporal stride for legacy_pe sampler.')
    p.add_argument('--temporal_sampler', type=str, default=None,
                   choices=['multiclip', 'uniform', 'uniform_clip', 'centered', 'random_stride', 'legacy_pe',
                            'consecutive_random', 'burst_4x2', 'segment_2x4', 'mixed_sampler',
                            'multi_clip_4x8'],
                   help='Temporal sampling strategy.')
    p.add_argument('--num_clips_train', type=int, default=None)
    p.add_argument('--num_clips_test', type=int, default=None)
    p.add_argument('--fr_align_mode', type=str, default=None,
                   choices=['auto', 'normalized', 'index'],
                   help='FR temporal alignment mode between ref/dis frame indices')
    p.add_argument('--fr_align_ratio_threshold', type=float, default=None,
                   help='Auto mode ratio threshold for switching to index alignment')
    p.add_argument('--detail_sampler', type=str, default=None,
                   choices=['gms', 'gmsavg', 'gmsaug', 'fragment', 'fupic', 'fullres', 'resize'])
    p.add_argument('--detail_resize_target', type=int, default=None,
                   help='Target size for detail_sampler=resize (single resized patch).')
    p.add_argument('--gms_patch', type=int, default=None)
    p.add_argument('--gms_patches_per_frame', type=int, default=None)
    p.add_argument('--gms_grid_size', type=int, default=None)
    p.add_argument('--fragment_grid_h', type=int, default=None)
    p.add_argument('--fragment_grid_w', type=int, default=None)
    p.add_argument('--fragment_size', type=int, default=None)
    p.add_argument('--semantic_sampler', type=str, default=None,
                   help="'resizeNNN', 'center_crop', 'gmsavgNNN', 'gmsaugNNN', 'fragmentNNN', or 'mssNNN'")
    p.add_argument('--semantic_gms_mode', type=str, default=None,
                   choices=['avg', 'random1', 'stack'],
                   help='Semantic GMS patch reduction mode (when semantic_sampler=gmsavg*)')
    p.add_argument('--semantic_gms_legacy_pe', type=int, default=None, choices=[0, 1],
                   help='Use legacy PE grid layout for semantic GMS patches')
    p.add_argument('--semantic_gms_adaptive_crop', type=int, default=None, choices=[0, 1],
                   help='Adaptive crop-resize: crop size scales with resolution, '
                        'then resize to patch_size. Preserves more high-freq detail than full-frame resize.')
    p.add_argument('--adaptive_crop_max_scale', type=float, default=None,
                   help='Max scale factor for adaptive crop (default 0=unlimited, e.g. 2.0 caps at 448 for 224 base)')
    p.add_argument('--semantic_target_size', type=int, default=None)
    p.add_argument('--semantic_gms_patches_per_frame', type=int, default=None)
    p.add_argument('--no_gms_min_patches', type=int, default=0,
                   help='Disable the min-16 patch floor for gmsavg/gmsaug (default 0=keep floor, 1=allow <16 patches)')
    p.add_argument('--semantic_gms_grid_size', type=int, default=None)
    p.add_argument('--semantic_fragment_grid_h', type=int, default=None)
    p.add_argument('--semantic_fragment_grid_w', type=int, default=None)
    p.add_argument('--semantic_fragment_size', type=int, default=None)
    p.add_argument('--sem_mss_patches', type=int, default=None,
                   help='Number of GMS patches for MSS local scale (default 16)')
    p.add_argument('--sem_mss_fusion', type=str, default=None,
                   choices=['cross_attn', 'concat', 'add'],
                   help='MSS fusion mode: cross_attn (default), concat, or add')
    p.add_argument('--sem_adaptive_patches', type=int, default=None, choices=[0, 1],
                   help='Resolution-adaptive GMS: auto-scale patch count for 4K videos')
    p.add_argument('--fupic_tile', type=int, default=None)
    p.add_argument('--fupic_stride', type=int, default=None)

    # Temporal
    p.add_argument('--temporal', type=str, default=None,
                   choices=['off', 'tsm', '3d_light'])

    # Multi-resolution
    p.add_argument('--rape', type=str, default=None, choices=['off', 'on'])
    p.add_argument('--scale_token', type=str, default=None, choices=['off', 'on'])
    p.add_argument('--res_conditioner', type=str, default=None, choices=['off', 'on'],
                   help='FiLM-style resolution conditioner: modulates backbone features '
                        'with resolution scale/bias (replaces ineffective RAPE/ScaleToken side branch)')
    p.add_argument('--res_conditioner_hidden', type=int, default=None,
                   help='Hidden dim for ResolutionConditioner MLP (default 256)')
    p.add_argument('--rope_resolution_scale', type=int, default=None, choices=[0, 1],
                   help='Scale RoPE2D grid positions by resolution ratio vs 1080p. '
                        'Encodes absolute spatial extent into attention without extra params.')
    p.add_argument('--patch_embed_cond', type=int, default=None, choices=[0, 1],
                   help='Scheme F: inject resolution bias after patch embedding (conv1). '
                        'MLP(2→64→width) zero-init, added to every patch token.')
    p.add_argument('--resolution_token', type=int, default=None, choices=[0, 1],
                   help='Scheme H: prepend a resolution-derived token to the patch sequence. '
                        'MLP(2→64→width) small-init, participates in self-attention.')

    # ---- Round 8: Generalization & advanced sampling parameters ----
    p.add_argument('--gradient_topk_sampling', type=int, default=None, choices=[0, 1],
                   help='Enable gradient-weighted Top-K GMS cell selection (Sobel diff between ref/dis)')
    p.add_argument('--gradient_topk_mode', type=str, default=None,
                   choices=['weighted', 'topk_uniform'],
                   help='weighted=probability-weighted random; topk_uniform=select top-K then uniform crop')
    p.add_argument('--stochastic_depth_rate', type=float, default=None,
                   help='Max drop rate for StochasticDepth in PE ViT (0=off, 0.1=linear 0→0.1)')
    p.add_argument('--feature_mixup_alpha', type=float, default=None,
                   help='Beta distribution alpha for feature mixup (0=off, 0.2=light mixup)')
    p.add_argument('--label_smooth_sigma', type=float, default=None,
                   help='Gaussian noise sigma for label smoothing on targets (0=off)')
    p.add_argument('--spectral_decoupling_weight', type=float, default=None,
                   help='Spectral decoupling regularization weight on backbone params (0=off)')
    p.add_argument('--feature_whitening', type=int, default=None, choices=[0, 1],
                   help='Enable feature whitening (ZCA) after backbone output')
    p.add_argument('--curriculum_learning', type=int, default=None, choices=[0, 1],
                   help='Enable curriculum learning: easy-to-hard sample weighting')
    p.add_argument('--curriculum_warmup_epochs', type=int, default=None,
                   help='Number of warmup epochs before curriculum kicks in (default 5)')

    # ---- Round 9: Temporal adaptor parameters ----
    p.add_argument('--temporal_adaptor', type=str, default=None,
                   choices=['none', 'self_attn', 'conv1d',
                            'diff_fusion', 'burst_two_level', 'segment_gated', 'shared_temporal_mixer'],
                   help='Temporal adaptor type for SemanticBranch (none=TSM only)')
    p.add_argument('--freeze_backbone', type=int, default=None, choices=[0, 1],
                   help='Freeze backbone parameters for adaptor fine-tuning')
    p.add_argument('--freeze_fr_modules', type=int, default=None, choices=[0, 1],
                   help='Freeze FR interaction modules (gate/fusion/topiq_layer_merge)')
    p.add_argument('--freeze_head', type=int, default=None, choices=[0, 1],
                   help='Freeze head parameters (for cross-clip temporal fine-tuning)')
    p.add_argument('--cross_clip_temporal', type=str, default=None,
                   choices=['none', 'clip_conv1d', 'clip_self_attn'],
                   help='Cross-clip temporal interaction module type')
    p.add_argument('--pretrain_path', type=str, default=None,
                   help='Path to pretrained weights for partial loading (strict=False)')

    # ---- Phase2/4K ablation parameters ----
    # Data augmentation
    p.add_argument('--aug_hflip', type=int, default=None, choices=[0, 1],
                   help='Enable horizontal flip augmentation (FR: flips both ref & dis)')
    p.add_argument('--aug_tflip', type=int, default=None, choices=[0, 1],
                   help='Enable temporal flip augmentation (reverse frame order)')
    p.add_argument('--aug_brightness', type=float, default=None,
                   help='Y-channel brightness jitter magnitude (0=off, e.g. 0.05)')
    # Phase-aware loss weighting
    p.add_argument('--phase_loss_weight', type=float, default=None,
                   help='Loss weight multiplier for Phase2/4K samples (1.0=no weighting)')
    # Test-Time Augmentation
    p.add_argument('--tta_hflip', type=int, default=None, choices=[0, 1],
                   help='TTA: horizontal flip inference averaging')

    # Loss
    p.add_argument('--losses', type=str, default=None, help='mse,rank,plcc,fidelity')
    p.add_argument('--target_label_key', type=str, default=None, choices=['mos', 'vmaf_target', 'vmaf_cache', 'auto', 'ensemble'],
                   help='Primary supervision target for main losses')
    p.add_argument('--ensemble_vmaf_weight', type=float, default=None,
                   help='VMAF weight in ensemble mode: target = w*VMAF + (1-w)*MOS. '
                        'Default 0.5. Set 0.8 for VMAF-dominant, 0.3 for MOS-dominant.')
    p.add_argument('--lambda_mse', type=float, default=None)
    p.add_argument('--mse_normalize', type=int, default=None, choices=[0, 1],
                   help='Use z-score normalized MSE to mitigate score-scale mismatch')
    p.add_argument('--lambda_rank', type=float, default=None)
    p.add_argument('--lambda_plcc', type=float, default=None)
    p.add_argument('--lambda_fidelity', type=float, default=None)
    p.add_argument('--fidelity_alpha', type=float, default=None)
    p.add_argument('--fidelity_buffer_mode', type=str, default=None,
                   help='Fidelity loss buffer mode: auto (default) or force. '
                        'force: always buffer when grad_accum>1, increasing pair count.')
    p.add_argument('--legacy_spatial_cross_batch_accum', type=int, default=None, choices=[0, 1],
                   help='Option B: cross-batch accumulation of the fidelity buffer under legacy spatial mode. '
                        'When enabled, legacy spatial no longer steps independently per patch; instead the buffer is accumulated across batches and stepped uniformly. '
                        'Must be used together with --grad_accum and --fidelity_buffer_mode force. Default 0=off.')
    p.add_argument('--rank_margin', type=float, default=None)
    p.add_argument('--vif_mono_weight', type=float, default=None)
    p.add_argument('--vif_rank_weight', type=float, default=None)
    p.add_argument('--vif_distill_weight', type=float, default=None)
    p.add_argument('--vif_distill_frame_weight', type=float, default=None)
    p.add_argument('--vif_distill_target', type=str, default=None,
                   choices=['annotation', 'external', 'mixed', 'auto'])

    # IO / Decode
    p.add_argument('--container_decoder', type=str, default=None,
                   choices=['ffmpeg', 'pyav', 'decord'])
    p.add_argument('--auto_probe', type=str, default=None, choices=['on', 'off'])
    p.add_argument('--pixel_format', type=str, default=None)
    p.add_argument('--bitdepth', type=int, default=None)
    p.add_argument('--signal_range', type=str, default=None,
                   choices=['auto', 'limited', 'full'],
                   help='Raw YUV luma/chroma range normalization mode')
    p.add_argument('--tenbit_mode', type=str, default=None,
                   choices=['shift8'],
                   help='10-bit raw YUV decode mode: shift8 (right-shift to 8bit, optimal).')
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--uv_upsample', type=str, default=None,
                   choices=['bilinear', 'bicubic'])
    p.add_argument('--resize_antialias', type=int, default=None, choices=[0, 1],
                   help='Antialias flag for torchvision/F.interpolate resize (0=off, 1=on)')
    p.add_argument('--data_error_fail_fast', type=int, default=None, choices=[0, 1],
                   help='Fail training when dataloader decoding/path errors exceed thresholds')
    p.add_argument('--data_error_max_count', type=int, default=None,
                   help='Maximum tolerated dataloader errors before fail-fast triggers')
    p.add_argument('--data_error_max_ratio', type=float, default=None,
                   help='Maximum tolerated dataloader error ratio before fail-fast triggers')

    # Debug frames
    p.add_argument('--debug_save_frames', type=str, default=None, choices=['on', 'off'])
    p.add_argument('--debug_num_videos_train', type=int, default=None)
    p.add_argument('--debug_num_videos_test', type=int, default=None)
    p.add_argument('--debug_num_frames', type=int, default=None)
    p.add_argument('--debug_every_epochs', type=int, default=None)
    p.add_argument('--debug_seed', type=int, default=None)

    # DDP / Training
    p.add_argument('--dist_backend', type=str, default=None)
    p.add_argument('--dist_url', type=str, default='env://')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--amp', action='store_true', default=None)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--grad_accum', type=int, default=None)
    p.add_argument('--ema_decay', type=float, default=None,
                   help='EMA decay rate for model averaging (0=off, e.g. 0.999 or 0.9999). '
                        'When enabled, uses EMA model for evaluation and checkpoint saving.')
    p.add_argument('--clip_grad', type=float, default=None)
    p.add_argument('--find_unused_parameters', action='store_true', default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--save_ckpt_freq', type=int, default=None)
    p.add_argument('--save_latest_ckpt', type=int, default=None, choices=[0, 1],
                   help='Save latest checkpoint for resume (1=True, 0=False). '
                        'Default is 0 to keep best-only weights.')
    p.add_argument('--save_only_best_cvqm_inference', type=int, default=None, choices=[0, 1],
                   help='Save only one inference weight file: best on CVQM test (1=True, 0=False).')
    p.add_argument('--train_use_only_vif_branch', type=int, default=None, choices=[0, 1],
                   help='Use only VIF branch score as training objective output')
    p.add_argument('--eval_use_only_vif_branch', type=int, default=None, choices=[0, 1],
                   help='Use only VIF branch score during validation/testing')
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--batch_exp_name', type=str, default=None,
                   help='Experiment name within a batch run (set by train_ddp.sh batch mode)')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--resume_strict', type=int, default=None, choices=[0, 1],
                   help='Strictly match checkpoint keys when resuming (1=True, 0=False)')
    p.add_argument('--log_interval', type=int, default=None)
    p.add_argument('--debug_frames', type=int, default=None,
                   help='Save debug frames for visual inspection (1=True, 0=False)')
    p.add_argument('--debug_frames_interval', type=int, default=None,
                   help='Save debug frames every N epochs')
    p.add_argument('--eval_interval', type=int, default=None)
    p.add_argument('--test_eval_interval', type=int, default=None)
    p.add_argument('--val_batch_mul', type=float, default=None,
                   help='Validation dataloader batch multiplier (val_bs = batch_size * val_batch_mul)')
    p.add_argument('--test_batch_mul', type=float, default=None,
                   help='Test dataloader batch multiplier (test_bs = batch_size * test_batch_mul)')
    p.add_argument('--skip_val', type=int, default=None,
                   help='Skip val evaluation during training (1=skip, 0=evaluate)')
    p.add_argument('--test_eval_mode', type=str, default=None,
                   choices=['fast', 'full'],
                   help='fast=training-style sampling for test, full=full evaluation')
    p.add_argument('--pretrained', type=int, default=None,
                   help='Use pretrained backbone weights (1=True, 0=False)')

    # Optimizer
    p.add_argument('--lr_backbone', type=float, default=None)
    p.add_argument('--lr_head', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--optimizer', type=str, default=None, choices=['adamw', 'adam', 'sgd'])
    p.add_argument('--scheduler', type=str, default=None, choices=['cosine', 'step', 'plateau'])
    p.add_argument('--step_size', type=int, default=None)
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--momentum', type=float, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--warmup_ratio', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)

    # Fusion
    p.add_argument('--fusion_type', type=str, default=None,
                   choices=['late_concat_mlp', 'gated_fusion', 'inject_then_head'])

    # Backbone specifics
    p.add_argument('--detail_backbone', type=str, default=None)
    p.add_argument('--semantic_backbone', type=str, default=None)
    p.add_argument('--semantic_fr_interaction', type=str, default=None,
                   choices=['diff_only', 'diff_prod', 'concat_mlp', 'cosine_diff',
                            'diff_affine', 'topiq_like',
                            'concat_mlp_deep', 'diff_affine_res', 'topiq_deep', 'ensemble_vote'])
    p.add_argument('--fr_interaction', type=str, default=None,
                   choices=['diff_only', 'diff_crossattn'])
    p.add_argument('--pe_weights', type=str, default=None,
                   help='Path to Meta PE pretrained weights (.pt). '
                        'Auto-resolved for known variants if omitted.')

    # Semantic branch alignment (repro ablation switches, work without pe_align.enabled)
    p.add_argument('--sem_apply_renorm', type=int, default=None, choices=[0, 1],
                   help='Apply ReNormalize layer in semantic branch (PE needs this)')
    p.add_argument('--sem_temporal_pool', type=str, default=None,
                   choices=['tsm_mean', 'attention', 'mean'],
                   help='Semantic branch temporal pooling mode')
    p.add_argument('--sem_head_mode', type=str, default=None,
                   choices=['projection', 'repro_mlp'],
                   help='Semantic branch head mode')
    p.add_argument('--sem_attn_heads', type=int, default=None)
    p.add_argument('--sem_dropout', type=float, default=None)
    p.add_argument('--sem_head_hidden1', type=int, default=None)
    p.add_argument('--sem_head_hidden2', type=int, default=None)
    p.add_argument('--sem_score_sigmoid', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_multipatch_input', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_topiq_multilayer', type=int, default=None, choices=[0, 1],
                   help='Enable TOPIQ multi-layer FR interaction in semantic branch')
    p.add_argument('--sem_topiq_layer_ids', type=str, default=None,
                   help='Comma-separated 1-based PE layer indices for multi-layer FR, e.g. "1,4,8,12"')
    p.add_argument('--sem_patch_reduce', type=str, default=None, choices=['mean', 'median', 'max'])
    p.add_argument('--sem_patch_forward_chunk_size', type=int, default=None)
    p.add_argument('--sem_layer_decay_enable', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_layer_decay', type=float, default=None)
    p.add_argument('--sem_legacy_no_decay_rule', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_legacy_layer_id_rule', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_legacy_layer_id_prefix_bug', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_lr_scale_like_pe', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_num_sample_factor', type=float, default=None)
    p.add_argument('--sem_legacy_iter_lr', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_legacy_min_lr', type=float, default=None)
    p.add_argument('--sem_legacy_spatial_train_step', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_eval_rescale', type=str, default=None, choices=['zscore', 'logistic5'])
    p.add_argument('--sem_legacy_single_rank_dist_sampler', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_legacy_disable_worker_init', type=int, default=None, choices=[0, 1])
    p.add_argument('--sem_legacy_temporal_window_sampling', type=int, default=None, choices=[0, 1])

    # CVQM phase
    p.add_argument('--cvqm_phase', type=str, default=None,
                   choices=['all', '1', '2', 'phase1', 'phase2'],
                   help='CVQM test phase: all, 1/phase1, 2/phase2')
    p.add_argument('--use_cvqm_npy', type=int, default=None, choices=[0, 1],
                   help='Enable CVQM npy pre-extracted segment loading for CVQM test data.')
    p.add_argument('--cvqm_npy_segments', type=int, default=None,
                   help='Stored segment count in CVQM npy cache (legacy format).')
    p.add_argument('--frame_cache_root', type=str, default=None,
                   help='Root directory for CVQM frame cache (NVMe acceleration). '
                        'E.g., /your/path/to/CVQM_Test_Frame')
    p.add_argument('--sampled_clip_cache_mode', type=str, default=None,
                   choices=['off', 'read', 'write', 'readwrite'],
                   help='Disk cache mode for sampled per-clip tensors from container videos. '
                        'Useful for repeated CVQAD/AVT evaluation.')
    p.add_argument('--sampled_clip_cache_root', type=str, default=None,
                   help='Root directory for sampled clip cache. If empty, falls back to '
                        'frame_cache_root/sampled_clip_cache_v1 or output_dir/sampled_clip_cache.')
    p.add_argument('--ref_clip_cache_size', type=int, default=None,
                   help='In-memory LRU size for repeated FR reference container clips (0 disables).')
    p.add_argument('--ref_clip_cache_max_mb', type=float, default=None,
                   help='Max size per cached reference clip in MB; larger clips are not cached.')

    # PE alignment (standalone ablation profile).
    p.add_argument('--pe_align_enable', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_model_variant', type=str, default=None, choices=['semantic_nr'])
    p.add_argument('--pe_align_apply_renorm', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_colorspace_mode', type=str, default=None,
                   choices=['bt709_imagenet', 'bt709_raw', 'bt601_imagenet', 'bt601_raw',
                            'bt709_clip', 'bt601_clip', 'none'])
    p.add_argument('--pe_align_patch_reduce', type=str, default=None, choices=['mean', 'median', 'max'])
    p.add_argument('--pe_align_patch_forward_chunk_size', type=int, default=None)
    p.add_argument('--pe_align_eval_rescale', type=str, default=None, choices=['zscore', 'logistic5'])
    p.add_argument('--pe_align_lr_scale_like_pe', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_num_sample_factor', type=float, default=None)
    p.add_argument('--pe_align_layer_decay_enable', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_layer_decay', type=float, default=None)
    p.add_argument('--pe_align_tenbit_mode', type=str, default=None, choices=['normalize', 'shift8'])
    p.add_argument('--pe_align_force_signal_range', type=str, default=None,
                   choices=['auto', 'limited', 'full'])
    p.add_argument('--pe_align_use_cvqm_npy', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_semantic_gms_mode', type=str, default=None, choices=['avg', 'random1', 'stack'])
    p.add_argument('--pe_align_semantic_legacy_grid', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_temporal_sampler', type=str, default=None,
                   choices=['multiclip', 'uniform', 'uniform_clip', 'centered', 'random_stride', 'legacy_pe',
                            'consecutive_random', 'burst_4x2', 'segment_2x4', 'mixed_sampler'])
    p.add_argument('--pe_align_sampling_rate', type=int, default=None)
    p.add_argument('--pe_align_legacy_spatial_train_step', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_iter_lr', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_min_lr', type=float, default=None)
    p.add_argument('--pe_align_resize_antialias', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_no_decay_rule', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_layer_id_rule', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_layer_id_prefix_bug', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_single_rank_dist_sampler', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_disable_worker_init', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_legacy_temporal_window_sampling', type=int, default=None, choices=[0, 1])
    p.add_argument('--pe_align_raw_yuv_backend', type=str, default=None, choices=['native', 'legacy_cv2'])
    p.add_argument('--pe_align_raw_yuv_matrix', type=str, default=None, choices=['bt709', 'bt601'])
    p.add_argument('--pe_align_container_yuv_matrix', type=str, default=None, choices=['bt709', 'bt601'])
    p.add_argument('--pe_align_container_yuv_direct', type=int, default=None, choices=[0, 1],
                   help='Container decode: 1=ffmpeg outputs YUV directly (no RGB roundtrip), 0=RGB then convert')

    # Eval specific
    p.add_argument('--eval_ckpt', type=str, default=None)
    p.add_argument('--eval_ckpt_strict', type=int, default=None, choices=[0, 1],
                   help='Strictly match checkpoint keys during eval weight loading')
    p.add_argument('--eval_stage', type=str, default=None, choices=['stage1', 'stage2', 'both'])
    p.add_argument('--enable_timing', type=int, default=None, choices=[0, 1],
                   help='Enable timing profiling during evaluation (0=off, 1=on). Default: 1')
    p.add_argument('--timing_warmup_batches', type=int, default=None,
                   help='Number of initial batches to EXCLUDE from timing aggregation '
                        '(GPU JIT, NAS prefetch, first-hit cache miss). Default: 2')

    return p


# ---- Default config dict (matches default_fr.yaml) ----
DEFAULT_CONFIG = {
    'task': 'fr',
    'train_datasets': 'CVQM_self',
    # 'train_datasets': 'CVQM_self,Waterloo4K,BVIHD,HDR_VDC,MCML4K',
    'test_dataset': 'CVQM',
    'test_dataset_mode': 'test',
    'dataset_path_fail_fast': True,
    'fr_strict_reference': True,
    'cache_only': False,
    'export_cvqad_cache_index_only': False,
    'cvqad_cache_index_path': '',
    'dry_run': False,

    'branches': 'vif,detail,semantic',
    'vif_mode': 'late_fuse',
    'vif_scales': 4,
    'vif_feature_dim': 64,
    'vif_impl': 'vmaf_like',
    'vif_branch': {
        'enable': True,
        'mode': 'aligned',
        'align_with_other_branches': True,
        'max_dense_frames': 32,
        'use_adm': True,
        'use_motion': True,
        'use_grad_ratio': False,
        'use_lap_ratio': False,
        'use_ti': False,
        'embed_dim': 32,
        'regressor_type': 'linear_svr',
        'regressor_hidden_dim': 128,
        'score_transform': 'sigmoid',
        'temporal_pool': 'mean',
        'score_mode': 'learned',
        'use_in_fusion': True,
        'residual_fusion': False,
        'score_fusion': 'none',
        'score_fusion_alpha': 1.0,
        'cache_features': False,
        'cache_force_rebuild': False,
        'prebuild_cache': False,
        'cache_dir': './output/vif_cache_shared_ffmpeg12x_libvmaf_cpu',
        'cache_partition_total': 1,
        'cache_partition_index': 0,
        'cache_partition_on_remaining': False,
        'cache_require_vmaf_score': False,
        'cache_device': 'cpu',
        'feature_source': 'libvmaf',
        'ffmpeg_bin': 'ffmpeg',
        'libvmaf_use_cuda': False,
        'libvmaf_n_threads': 0,
        'libvmaf_model': '',
        'vmaf_model_dir': '',
        'num_scales': 4,
        'window_size': 5,
        'quantile_pool_size': 64,
        'temporal_mean_weight': 1.0,
        'temporal_p10_weight': 0.0,
        'temporal_hmean_weight': 0.0,
        'dropout': 0.1,
        'norm_momentum': 0.01,
        'feature_norm_affine': True,
        'score_range': 100.0,
        'prior_blend': 0.0,
        'prior_residual_scale': 0.25,
        'learnable_prior_residual_scale': False,
        'learnable_prior_blend': False,
    },
    'colorspace': 'bt709_imagenet',

    'num_frames': 8,
    'sampling_rate': 1,
    'temporal_sampler': 'multiclip',
    'num_clips_train': 2,
    'num_clips_test': 4,
    'fr_align_mode': 'auto',
    'fr_align_ratio_threshold': 1.2,
    'detail_sampler': 'gms',
    'detail_resize_target': 256,
    'gms_patch': 256,
    'gms_patches_per_frame': 8,
    'gms_grid_size': 7,
    'fragment_grid_h': 7,
    'fragment_grid_w': 7,
    'fragment_size': 32,
    'semantic_sampler': 'resize384',
    'semantic_target_size': 224,
    'semantic_gms_patches_per_frame': 8,
    'no_gms_min_patches': False,       # True = disable min-16 patch floor for gmsavg/gmsaug
    'semantic_gms_grid_size': 7,
    'semantic_gms_mode': '',           # '' = auto from sampler name; 'avg' | 'random1' | 'stack'
    'semantic_gms_legacy_pe': False,   # Legacy PE grid layout for semantic GMS
    'semantic_gms_adaptive_crop': False,  # Adaptive crop-resize: scales crop size by resolution
    'adaptive_crop_max_scale': 0.0,       # Max crop scale factor (0=unlimited, 2.0=max 448 for 224 base)
    'semantic_fragment_grid_h': 7,
    'semantic_fragment_grid_w': 7,
    'semantic_fragment_size': 32,
    'sem_mss_patches': 16,             # MSS local GMS patch count (only used when semantic_sampler starts with 'mss')
    'sem_adaptive_patches': False,      # Resolution-adaptive GMS: auto-scale patch count for 4K (area ratio vs 1080p)
    'fupic_tile': 224,
    'fupic_stride': 192,

    'temporal': 'tsm',
    'rape': 'on',
    'scale_token': 'on',
    'res_conditioner': 'off',          # FiLM-style: modulates backbone feats with resolution
    'res_conditioner_hidden': 256,     # Hidden dim for ResolutionConditioner MLP
    'rope_resolution_scale': False,    # Scale RoPE2D positions by resolution ratio vs 1080p
    'patch_embed_cond': False,         # Scheme F: resolution bias after patch embedding
    'resolution_token': False,         # Scheme H: resolution token prepended to patch sequence

    # ---- Round 8: Generalization & advanced sampling ----
    'gradient_topk_sampling': False,   # Gradient-weighted Top-K GMS cell selection
    'gradient_topk_mode': 'weighted',  # 'weighted' | 'topk_uniform'
    'stochastic_depth_rate': 0.0,      # Max drop rate for StochasticDepth in PE ViT (0=off)
    'feature_mixup_alpha': 0.0,        # Beta distribution alpha for feature mixup (0=off)
    'label_smooth_sigma': 0.0,         # Gaussian noise sigma for label smoothing (0=off)
    'spectral_decoupling_weight': 0.0, # Spectral decoupling reg weight (0=off)
    'feature_whitening': False,        # Feature whitening after backbone
    'curriculum_learning': False,      # Curriculum learning: easy-to-hard
    'curriculum_warmup_epochs': 5,     # Warmup before curriculum kicks in

    # ---- Round 9: Temporal adaptor ----
    'temporal_adaptor': 'none',        # 'none' | 'self_attn' | 'conv1d'
    'freeze_backbone': False,          # Freeze backbone for adaptor fine-tuning
    'freeze_fr_modules': False,        # Freeze FR interaction modules
    'freeze_head': False,              # Freeze head for cross-clip temporal fine-tuning
    'cross_clip_temporal': 'none',     # Cross-clip temporal module: 'none' | 'clip_conv1d' | 'clip_self_attn'
    'pretrain_path': None,             # Pretrained weights path for partial loading

    # ---- Phase2/4K ablation defaults (all off → no impact on existing experiments) ----
    'aug_hflip': False,
    'aug_tflip': False,
    'aug_brightness': 0.0,
    'phase_loss_weight': 1.0,     # 1.0 = no weighting; >1 = upweight Phase2/4K samples
    'tta_hflip': False,

    'losses': 'mse,rank,plcc,fidelity',
    'target_label_key': 'mos',
    'ensemble_vmaf_weight': 0.5,
    'lambda_mse': 1.0,
    'mse_normalize': False,
    'lambda_rank': 0.1,
    'lambda_plcc': 0.1,
    'lambda_fidelity': 0.1,
    'fidelity_alpha': 10.0,
    'fidelity_buffer_mode': 'auto',  # 'auto' or 'force'; force enables the buffer so pairwise loss can leverage a large batch
    'legacy_spatial_cross_batch_accum': False,  # Option B: legacy spatial cross-batch accumulation of fidelity buffer
    'rank_margin': 0.05,
    'loss': {
        'vif_mono_weight': 0.0,
        'vif_rank_weight': 0.0,
        'vif_distill_weight': 0.0,
        'vif_distill_frame_weight': 0.0,
        'vif_distill_target': 'annotation',
    },

    'container_decoder': 'pyav',
    'auto_probe': 'on',
    'pixel_format': 'yuv420p10le',
    'bitdepth': 10,
    'signal_range': 'auto',
    'tenbit_mode': 'shift8',            # G4 ablation: shift8 > normalize (+0.016 SRCC, optimal for direct 10bit YUV read)
    'uv_upsample': 'bilinear',          # G6a ablation: bilinear ≈ bicubic (slightly better, and faster)
    'resize_antialias': False,           # G7 ablation: on/off ≈ equal (off is faster, PE default)
    'data_error_fail_fast': True,
    'data_error_max_count': 8,
    'data_error_max_ratio': 0.05,

    'debug_save_frames': 'on',
    'debug_num_videos_train': 2,
    'debug_num_videos_test': 2,
    'debug_num_frames': 1,
    'debug_every_epochs': 5,
    'debug_seed': 42,

    'dist_backend': 'nccl',
    'dist_url': 'env://',
    'seed': 43,
    'amp': True,
    'grad_accum': 4,
    'ema_decay': 0.0,              # EMA decay rate (0=off, 0.999/0.9999=on)
    'clip_grad': 5.0,
    'find_unused_parameters': True,
    'save_dir': './output',
    'batch_exp_name': None,             # Set by train_ddp.sh batch mode for auto-skip detection
    'log_interval': 10,
    'eval_interval': 1,
    'test_eval_interval': 5,  # test set eval every N epochs (slower, 851 CVQM samples)
    'val_batch_mul': 2.0,
    'test_batch_mul': 2.0,
    'skip_val': True,  # skip val eval (train==val for most setups)
    'test_eval_mode': 'fast',  # 'fast' = training-style sampling, 'full' = full evaluation
    'pretrained': True,  # load ImageNet pretrained backbone weights via timm
    'debug_frames': False,  # save debug frames for visual inspection
    'debug_frames_interval': 5,  # save debug frames every N epochs

    'lr_backbone': 1e-4,
    'lr_head': 5e-4,
    'weight_decay': 0.05,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'step_size': 30,
    'gamma': 0.1,
    'momentum': 0.9,
    'epochs': 100,
    'warmup_ratio': 0.05,
    'batch_size': 1,
    'num_workers': 2,
    'patience': 20,
    'save_ckpt_freq': 5,
    'save_latest_ckpt': False,
    'save_only_best_cvqm_inference': False,
    'train_use_only_vif_branch': False,
    'eval_use_only_vif_branch': False,
    'resume': None,
    'resume_strict': False,

    'fusion_type': 'late_concat_mlp',
    'fusion_hidden': 256,
    'aggregator_mode': 'mean',
    'detail_backbone': 'swinv2_tiny',
    'semantic_backbone': 'convnextv2_tiny',
    'semantic_fr_interaction': 'diff_prod',
    'pe_weights': None,
    'fr_interaction': 'diff_only',

    # ---- Semantic branch alignment config (repro ablation switches) ----
    'semantic_branch': {
        'apply_renorm': False,
        'temporal_pool': 'tsm_mean',   # 'tsm_mean' | 'attention' | 'mean'
        'head_mode': 'projection',     # 'projection' | 'repro_mlp'
        'topiq_multilayer': True,      # TOPIQ multi-layer FR: extract from multiple PE layers, interact per-layer, concat+project
        'topiq_layer_ids': [1, 4, 7, 10],  # 1-based PE encoder layer indices for multi-layer FR
        'attn_heads': 8,
        'dropout': 0.1,
        'head_hidden1': 512,
        'head_hidden2': 128,
        'score_sigmoid': True,
        'multipatch_input': False,
        'patch_reduce': 'mean',        # 'mean' | 'median' | 'max'
        'patch_forward_chunk_size': 0,
        # Training alignment (works without pe_align.enabled)
        'layer_decay_enable': False,
        'layer_decay': 0.75,
        'legacy_no_decay_rule': False,
        'legacy_layer_id_rule': False,
        'legacy_layer_id_prefix_bug': True,
        'lr_scale_like_pe': False,
        'num_sample_factor': 1.0,
        'legacy_iter_lr': False,
        'legacy_min_lr': 1e-6,
        'legacy_spatial_train_step': False,
        'eval_rescale': 'zscore',      # 'zscore' | 'logistic5'
        # Data pipeline legacy alignment (only effective via semantic_branch path)
        'legacy_single_rank_dist_sampler': False,
        'legacy_disable_worker_init': False,
        'legacy_temporal_window_sampling': False,
        # MSS (Multi-Scale Spatial) — only active when semantic_sampler starts with 'mss'
        'mss': False,
        'mss_fusion': 'cross_attn',   # 'cross_attn' | 'concat' | 'add'
        # Temporal adaptor (Round 9): 'none' | 'self_attn' | 'conv1d'
        'temporal_adaptor': 'none',
        # Scheme E: Resolution-Scaled RoPE (forwarded from top-level by hmf_vqa.py)
        'rope_resolution_scale': False,
        # Scheme F: Patch Embedding Conditioning (forwarded from top-level by hmf_vqa.py)
        'patch_embed_cond': False,
        # Scheme H: Resolution Token (forwarded from top-level by hmf_vqa.py)
        'resolution_token': False,
    },

    'cvqm_phase': 'all',
    'use_cvqm_npy': False,
    'cvqm_npy_segments': 4,
    'frame_cache_root': None,  # NVMe frame cache root for CVQM eval acceleration
    'sampled_clip_cache_mode': 'off',  # off|read|write|readwrite for container sampled clips
    'sampled_clip_cache_root': None,  # default resolved from frame_cache_root/output_dir
    'ref_clip_cache_size': 1,  # Exact in-memory LRU for repeated FR container references
    'ref_clip_cache_max_mb': 256.0,  # Skip oversized clips (e.g. 4K) to keep RAM bounded
    'pe_align': {
        'enabled': False,
        'model_variant': 'semantic_nr',
        'apply_renorm': True,
        'colorspace_mode': 'bt601_imagenet',
        'attn_heads': 8,
        'dropout': 0.1,
        'head_hidden1': 512,
        'head_hidden2': 128,
        'score_sigmoid': True,
        'patch_reduce': 'mean',
        'patch_forward_chunk_size': 0,
        'eval_rescale': 'zscore',
        'lr_scale_like_pe': False,
        'num_sample_factor': 1.0,
        'layer_decay_enable': False,
        'layer_decay': 0.75,
        'use_cvqm_npy': False,
        'temporal_sampler': 'legacy_pe',
        'sampling_rate': 8,
        'semantic_gms_mode': 'stack',
        'semantic_legacy_grid': True,
        'tenbit_mode': 'shift8',
        'force_signal_range': 'auto',
        'legacy_spatial_train_step': False,
        'legacy_iter_lr': False,
        'legacy_min_lr': 1e-6,
        'resize_antialias': False,
        'legacy_no_decay_rule': False,
        'legacy_layer_id_rule': False,
        'legacy_layer_id_prefix_bug': True,
        'legacy_single_rank_dist_sampler': False,
        'legacy_disable_worker_init': False,
        'legacy_temporal_window_sampling': False,
        'raw_yuv_backend': 'native',
        'raw_yuv_matrix': 'bt709',
        'container_yuv_matrix': 'bt709',
        'container_yuv_direct': False,
    },
    'swin_align': {
        'enabled': False,
        'model_variant': 'detail_nr',
        'colorspace_mode': 'bt709_imagenet',
        'img_size': 256,
        'attn_heads': 8,
        'dropout': 0.1,
        'head_hidden1': 512,
        'head_hidden2': 128,
        'score_sigmoid': True,
        'patch_reduce': 'mean',
        'patch_forward_chunk_size': 0,
        'eval_rescale': 'zscore',
        'lr_scale_like_pe': False,
        'num_sample_factor': 1.0,
        'use_cvqm_npy': False,
        'temporal_sampler': 'legacy_pe',
        'sampling_rate': 8,
        'semantic_gms_mode': 'stack',
        'semantic_legacy_grid': True,
        'tenbit_mode': 'shift8',
        'force_signal_range': 'auto',
        'legacy_spatial_train_step': False,
        'legacy_iter_lr': False,
        'legacy_min_lr': 1e-6,
        'resize_antialias': False,
        'legacy_single_rank_dist_sampler': False,
        'legacy_disable_worker_init': False,
        'legacy_temporal_window_sampling': False,
        'raw_yuv_backend': 'native',
        'raw_yuv_matrix': 'bt709',
        'container_yuv_matrix': 'bt709',
        'container_yuv_direct': False,
    },
    'convnext_align': {
        'enabled': False,
        'model_variant': 'semantic_nr',
        'colorspace_mode': 'bt709_imagenet',
        'attn_heads': 8,
        'dropout': 0.1,
        'head_hidden1': 512,
        'head_hidden2': 128,
        'score_sigmoid': True,
        'patch_reduce': 'mean',
        'patch_forward_chunk_size': 0,
        'eval_rescale': 'zscore',
        'lr_scale_like_pe': False,
        'num_sample_factor': 1.0,
        'use_cvqm_npy': False,
        'temporal_sampler': 'legacy_pe',
        'sampling_rate': 8,
        'semantic_gms_mode': 'stack',
        'semantic_legacy_grid': True,
        'tenbit_mode': 'shift8',
        'force_signal_range': 'auto',
        'legacy_spatial_train_step': False,
        'legacy_iter_lr': False,
        'legacy_min_lr': 1e-6,
        'resize_antialias': False,
        'legacy_single_rank_dist_sampler': False,
        'legacy_disable_worker_init': False,
        'legacy_temporal_window_sampling': False,
        'raw_yuv_backend': 'native',
        'raw_yuv_matrix': 'bt709',
        'container_yuv_matrix': 'bt709',
        'container_yuv_direct': False,
    },
    'eval_ckpt': None,
    'eval_ckpt_strict': True,
    'eval_stage': 'both',
    'enable_timing': True,
    'timing_warmup_batches': 2,
    'width': None,
    'height': None,
}


def _validate_config_keys(cfg: dict):
    """Fail fast on unknown config keys to avoid silent no-op experiments."""
    allowed_top = set(DEFAULT_CONFIG.keys())
    unknown_top = sorted([k for k in cfg.keys() if k not in allowed_top])
    if unknown_top:
        raise KeyError(
            "Unknown top-level config keys detected: "
            f"{unknown_top}. Please rename/remove these keys."
        )

    # Nested dict keys that must stay in sync with code paths.
    nested_specs = {
        'vif_branch': set(DEFAULT_CONFIG['vif_branch'].keys()),
        'loss': set(DEFAULT_CONFIG['loss'].keys()),
        'semantic_branch': set(DEFAULT_CONFIG['semantic_branch'].keys()),
        'pe_align': set(DEFAULT_CONFIG['pe_align'].keys()),
        'swin_align': set(DEFAULT_CONFIG['swin_align'].keys()),
        'convnext_align': set(DEFAULT_CONFIG['convnext_align'].keys()),
    }
    for nk, allowed in nested_specs.items():
        sub = cfg.get(nk, {})
        if isinstance(sub, dict):
            unknown_sub = sorted([k for k in sub.keys() if k not in allowed])
            if unknown_sub:
                raise KeyError(
                    f"Unknown nested config keys in '{nk}': {unknown_sub}. "
                    "Please rename/remove these keys."
                )


def build_config(cli_args=None) -> dict:
    """
    Build final config by:
      1. Start with DEFAULT_CONFIG
      2. Merge YAML if --config provided
      3. Override with CLI arguments
    Returns a flat dict.
    """
    parser = get_arg_parser()
    args = parser.parse_args(cli_args)

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # Merge YAML
    if args.config is not None:
        yaml_cfg = load_yaml(args.config)
        cfg = merge_dict(cfg, yaml_cfg)

    def _set_nested(d: dict, path: List[str], value):
        cur = d
        for k in path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[path[-1]] = value

    nested_override_paths = {
        'vif_branch_enable': ['vif_branch', 'enable'],
        'vif_branch_mode': ['vif_branch', 'mode'],
        'vif_align_with_other_branches': ['vif_branch', 'align_with_other_branches'],
        'vif_max_dense_frames': ['vif_branch', 'max_dense_frames'],
        'vif_use_grad_ratio': ['vif_branch', 'use_grad_ratio'],
        'vif_use_lap_ratio': ['vif_branch', 'use_lap_ratio'],
        'vif_use_ti': ['vif_branch', 'use_ti'],
        'vif_use_adm': ['vif_branch', 'use_adm'],
        'vif_use_motion': ['vif_branch', 'use_motion'],
        'vif_embed_dim': ['vif_branch', 'embed_dim'],
        'vif_regressor_type': ['vif_branch', 'regressor_type'],
        'vif_regressor_hidden_dim': ['vif_branch', 'regressor_hidden_dim'],
        'vif_score_transform': ['vif_branch', 'score_transform'],
        'vif_temporal_pool': ['vif_branch', 'temporal_pool'],
        'vif_temporal_mean_weight': ['vif_branch', 'temporal_mean_weight'],
        'vif_temporal_p10_weight': ['vif_branch', 'temporal_p10_weight'],
        'vif_temporal_hmean_weight': ['vif_branch', 'temporal_hmean_weight'],
        'vif_num_scales': ['vif_branch', 'num_scales'],
        'vif_window_size': ['vif_branch', 'window_size'],
        'vif_quantile_pool_size': ['vif_branch', 'quantile_pool_size'],
        'vif_dropout': ['vif_branch', 'dropout'],
        'vif_norm_momentum': ['vif_branch', 'norm_momentum'],
        'vif_feature_norm_affine': ['vif_branch', 'feature_norm_affine'],
        'vif_score_range': ['vif_branch', 'score_range'],
        'vif_prior_blend': ['vif_branch', 'prior_blend'],
        'vif_score_mode': ['vif_branch', 'score_mode'],
        'vif_prior_residual_scale': ['vif_branch', 'prior_residual_scale'],
        'vif_learnable_prior_residual_scale': ['vif_branch', 'learnable_prior_residual_scale'],
        'vif_learnable_prior_blend': ['vif_branch', 'learnable_prior_blend'],
        'vif_use_in_fusion': ['vif_branch', 'use_in_fusion'],
        'vif_residual_fusion': ['vif_branch', 'residual_fusion'],
        'vif_score_fusion': ['vif_branch', 'score_fusion'],
        'vif_score_fusion_alpha': ['vif_branch', 'score_fusion_alpha'],
        'vif_cache_features': ['vif_branch', 'cache_features'],
        'vif_cache_force_rebuild': ['vif_branch', 'cache_force_rebuild'],
        'vif_cache_prebuild': ['vif_branch', 'prebuild_cache'],
        'vif_cache_dir': ['vif_branch', 'cache_dir'],
        'vif_cache_partition_total': ['vif_branch', 'cache_partition_total'],
        'vif_cache_partition_index': ['vif_branch', 'cache_partition_index'],
        'vif_cache_partition_on_remaining': ['vif_branch', 'cache_partition_on_remaining'],
        'vif_cache_require_vmaf_score': ['vif_branch', 'cache_require_vmaf_score'],
        'vif_cache_device': ['vif_branch', 'cache_device'],
        'vif_feature_source': ['vif_branch', 'feature_source'],
        'vif_ffmpeg_bin': ['vif_branch', 'ffmpeg_bin'],
        'vif_libvmaf_use_cuda': ['vif_branch', 'libvmaf_use_cuda'],
        'vif_libvmaf_n_threads': ['vif_branch', 'libvmaf_n_threads'],
        'vif_libvmaf_model': ['vif_branch', 'libvmaf_model'],
        'vif_vmaf_model_dir': ['vif_branch', 'vmaf_model_dir'],
        'vif_mono_weight': ['loss', 'vif_mono_weight'],
        'vif_rank_weight': ['loss', 'vif_rank_weight'],
        'vif_distill_weight': ['loss', 'vif_distill_weight'],
        'vif_distill_frame_weight': ['loss', 'vif_distill_frame_weight'],
        'vif_distill_target': ['loss', 'vif_distill_target'],
        # ---- semantic_branch ----
        'sem_apply_renorm': ['semantic_branch', 'apply_renorm'],
        'sem_temporal_pool': ['semantic_branch', 'temporal_pool'],
        'sem_head_mode': ['semantic_branch', 'head_mode'],
        'sem_attn_heads': ['semantic_branch', 'attn_heads'],
        'sem_dropout': ['semantic_branch', 'dropout'],
        'sem_head_hidden1': ['semantic_branch', 'head_hidden1'],
        'sem_head_hidden2': ['semantic_branch', 'head_hidden2'],
        'sem_score_sigmoid': ['semantic_branch', 'score_sigmoid'],
        'sem_multipatch_input': ['semantic_branch', 'multipatch_input'],
        'sem_topiq_multilayer': ['semantic_branch', 'topiq_multilayer'],
        'sem_topiq_layer_ids': ['semantic_branch', 'topiq_layer_ids'],
        'sem_patch_reduce': ['semantic_branch', 'patch_reduce'],
        'sem_patch_forward_chunk_size': ['semantic_branch', 'patch_forward_chunk_size'],
        'sem_mss_fusion': ['semantic_branch', 'mss_fusion'],
        'sem_layer_decay_enable': ['semantic_branch', 'layer_decay_enable'],
        'sem_layer_decay': ['semantic_branch', 'layer_decay'],
        'sem_legacy_no_decay_rule': ['semantic_branch', 'legacy_no_decay_rule'],
        'sem_legacy_layer_id_rule': ['semantic_branch', 'legacy_layer_id_rule'],
        'sem_legacy_layer_id_prefix_bug': ['semantic_branch', 'legacy_layer_id_prefix_bug'],
        'sem_lr_scale_like_pe': ['semantic_branch', 'lr_scale_like_pe'],
        'sem_num_sample_factor': ['semantic_branch', 'num_sample_factor'],
        'sem_legacy_iter_lr': ['semantic_branch', 'legacy_iter_lr'],
        'sem_legacy_min_lr': ['semantic_branch', 'legacy_min_lr'],
        'sem_legacy_spatial_train_step': ['semantic_branch', 'legacy_spatial_train_step'],
        'sem_eval_rescale': ['semantic_branch', 'eval_rescale'],
        'sem_legacy_single_rank_dist_sampler': ['semantic_branch', 'legacy_single_rank_dist_sampler'],
        'sem_legacy_disable_worker_init': ['semantic_branch', 'legacy_disable_worker_init'],
        'sem_legacy_temporal_window_sampling': ['semantic_branch', 'legacy_temporal_window_sampling'],
        # ---- semantic_branch temporal adaptor ----
        'temporal_adaptor': ['semantic_branch', 'temporal_adaptor'],
        # ---- pe_align ----
        'pe_align_enable': ['pe_align', 'enabled'],
        'pe_align_model_variant': ['pe_align', 'model_variant'],
        'pe_align_apply_renorm': ['pe_align', 'apply_renorm'],
        'pe_align_colorspace_mode': ['pe_align', 'colorspace_mode'],
        'pe_align_patch_reduce': ['pe_align', 'patch_reduce'],
        'pe_align_patch_forward_chunk_size': ['pe_align', 'patch_forward_chunk_size'],
        'pe_align_eval_rescale': ['pe_align', 'eval_rescale'],
        'pe_align_lr_scale_like_pe': ['pe_align', 'lr_scale_like_pe'],
        'pe_align_num_sample_factor': ['pe_align', 'num_sample_factor'],
        'pe_align_layer_decay_enable': ['pe_align', 'layer_decay_enable'],
        'pe_align_layer_decay': ['pe_align', 'layer_decay'],
        'pe_align_tenbit_mode': ['pe_align', 'tenbit_mode'],
        'pe_align_force_signal_range': ['pe_align', 'force_signal_range'],
        'pe_align_use_cvqm_npy': ['pe_align', 'use_cvqm_npy'],
        'pe_align_semantic_gms_mode': ['pe_align', 'semantic_gms_mode'],
        'pe_align_semantic_legacy_grid': ['pe_align', 'semantic_legacy_grid'],
        'pe_align_temporal_sampler': ['pe_align', 'temporal_sampler'],
        'pe_align_sampling_rate': ['pe_align', 'sampling_rate'],
        'pe_align_legacy_spatial_train_step': ['pe_align', 'legacy_spatial_train_step'],
        'pe_align_legacy_iter_lr': ['pe_align', 'legacy_iter_lr'],
        'pe_align_legacy_min_lr': ['pe_align', 'legacy_min_lr'],
        'pe_align_resize_antialias': ['pe_align', 'resize_antialias'],
        'pe_align_legacy_no_decay_rule': ['pe_align', 'legacy_no_decay_rule'],
        'pe_align_legacy_layer_id_rule': ['pe_align', 'legacy_layer_id_rule'],
        'pe_align_legacy_layer_id_prefix_bug': ['pe_align', 'legacy_layer_id_prefix_bug'],
        'pe_align_legacy_single_rank_dist_sampler': ['pe_align', 'legacy_single_rank_dist_sampler'],
        'pe_align_legacy_disable_worker_init': ['pe_align', 'legacy_disable_worker_init'],
        'pe_align_legacy_temporal_window_sampling': ['pe_align', 'legacy_temporal_window_sampling'],
        'pe_align_raw_yuv_backend': ['pe_align', 'raw_yuv_backend'],
        'pe_align_raw_yuv_matrix': ['pe_align', 'raw_yuv_matrix'],
        'pe_align_container_yuv_matrix': ['pe_align', 'container_yuv_matrix'],
        'pe_align_container_yuv_direct': ['pe_align', 'container_yuv_direct'],
        # ---- swin_align ----
        'swin_align_enable': ['swin_align', 'enabled'],
        'swin_align_model_variant': ['swin_align', 'model_variant'],
        'swin_align_colorspace_mode': ['swin_align', 'colorspace_mode'],
        'swin_align_img_size': ['swin_align', 'img_size'],
        'swin_align_patch_reduce': ['swin_align', 'patch_reduce'],
        'swin_align_patch_forward_chunk_size': ['swin_align', 'patch_forward_chunk_size'],
        'swin_align_eval_rescale': ['swin_align', 'eval_rescale'],
        'swin_align_lr_scale_like_pe': ['swin_align', 'lr_scale_like_pe'],
        'swin_align_num_sample_factor': ['swin_align', 'num_sample_factor'],
        'swin_align_tenbit_mode': ['swin_align', 'tenbit_mode'],
        'swin_align_force_signal_range': ['swin_align', 'force_signal_range'],
        'swin_align_use_cvqm_npy': ['swin_align', 'use_cvqm_npy'],
        'swin_align_semantic_gms_mode': ['swin_align', 'semantic_gms_mode'],
        'swin_align_semantic_legacy_grid': ['swin_align', 'semantic_legacy_grid'],
        'swin_align_temporal_sampler': ['swin_align', 'temporal_sampler'],
        'swin_align_sampling_rate': ['swin_align', 'sampling_rate'],
        'swin_align_legacy_spatial_train_step': ['swin_align', 'legacy_spatial_train_step'],
        'swin_align_legacy_iter_lr': ['swin_align', 'legacy_iter_lr'],
        'swin_align_legacy_min_lr': ['swin_align', 'legacy_min_lr'],
        'swin_align_resize_antialias': ['swin_align', 'resize_antialias'],
        'swin_align_legacy_single_rank_dist_sampler': ['swin_align', 'legacy_single_rank_dist_sampler'],
        'swin_align_legacy_disable_worker_init': ['swin_align', 'legacy_disable_worker_init'],
        'swin_align_legacy_temporal_window_sampling': ['swin_align', 'legacy_temporal_window_sampling'],
        'swin_align_raw_yuv_backend': ['swin_align', 'raw_yuv_backend'],
        'swin_align_raw_yuv_matrix': ['swin_align', 'raw_yuv_matrix'],
        'swin_align_container_yuv_matrix': ['swin_align', 'container_yuv_matrix'],
        'swin_align_container_yuv_direct': ['swin_align', 'container_yuv_direct'],
        # ---- convnext_align ----
        'convnext_align_enable': ['convnext_align', 'enabled'],
        'convnext_align_model_variant': ['convnext_align', 'model_variant'],
        'convnext_align_colorspace_mode': ['convnext_align', 'colorspace_mode'],
        'convnext_align_patch_reduce': ['convnext_align', 'patch_reduce'],
        'convnext_align_patch_forward_chunk_size': ['convnext_align', 'patch_forward_chunk_size'],
        'convnext_align_eval_rescale': ['convnext_align', 'eval_rescale'],
        'convnext_align_lr_scale_like_pe': ['convnext_align', 'lr_scale_like_pe'],
        'convnext_align_num_sample_factor': ['convnext_align', 'num_sample_factor'],
        'convnext_align_tenbit_mode': ['convnext_align', 'tenbit_mode'],
        'convnext_align_force_signal_range': ['convnext_align', 'force_signal_range'],
        'convnext_align_use_cvqm_npy': ['convnext_align', 'use_cvqm_npy'],
        'convnext_align_semantic_gms_mode': ['convnext_align', 'semantic_gms_mode'],
        'convnext_align_semantic_legacy_grid': ['convnext_align', 'semantic_legacy_grid'],
        'convnext_align_temporal_sampler': ['convnext_align', 'temporal_sampler'],
        'convnext_align_sampling_rate': ['convnext_align', 'sampling_rate'],
        'convnext_align_legacy_spatial_train_step': ['convnext_align', 'legacy_spatial_train_step'],
        'convnext_align_legacy_iter_lr': ['convnext_align', 'legacy_iter_lr'],
        'convnext_align_legacy_min_lr': ['convnext_align', 'legacy_min_lr'],
        'convnext_align_resize_antialias': ['convnext_align', 'resize_antialias'],
        'convnext_align_legacy_single_rank_dist_sampler': ['convnext_align', 'legacy_single_rank_dist_sampler'],
        'convnext_align_legacy_disable_worker_init': ['convnext_align', 'legacy_disable_worker_init'],
        'convnext_align_legacy_temporal_window_sampling': ['convnext_align', 'legacy_temporal_window_sampling'],
        'convnext_align_raw_yuv_backend': ['convnext_align', 'raw_yuv_backend'],
        'convnext_align_raw_yuv_matrix': ['convnext_align', 'raw_yuv_matrix'],
        'convnext_align_container_yuv_matrix': ['convnext_align', 'container_yuv_matrix'],
        'convnext_align_container_yuv_direct': ['convnext_align', 'container_yuv_direct'],
    }
    bool_int_keys = {
        'resize_antialias',
        'mse_normalize',
        'semantic_gms_legacy_pe',
        'semantic_gms_adaptive_crop',
        'patch_embed_cond',
        'resolution_token',
        'gradient_topk_sampling',
        'feature_whitening',
        'curriculum_learning',
        'sem_adaptive_patches',
        'aug_hflip',
        'aug_tflip',
        'tta_hflip',
        'vif_branch_enable',
        'vif_align_with_other_branches',
        'vif_use_grad_ratio',
        'vif_use_lap_ratio',
        'vif_use_ti',
        'vif_use_adm',
        'vif_use_motion',
        'vif_feature_norm_affine',
        'vif_use_in_fusion',
        'vif_residual_fusion',
        'vif_cache_features',
        'vif_cache_force_rebuild',
        'vif_cache_prebuild',
        'vif_cache_partition_on_remaining',
        'vif_cache_require_vmaf_score',
        'vif_libvmaf_use_cuda',
        'vif_learnable_prior_residual_scale',
        'vif_learnable_prior_blend',
        'use_cvqm_npy',
        # ---- semantic_branch ----
        'sem_apply_renorm',
        'sem_score_sigmoid',
        'sem_multipatch_input',
        'sem_topiq_multilayer',
        'sem_layer_decay_enable',
        'sem_legacy_no_decay_rule',
        'sem_legacy_layer_id_rule',
        'sem_legacy_layer_id_prefix_bug',
        'sem_lr_scale_like_pe',
        'sem_legacy_iter_lr',
        'sem_legacy_spatial_train_step',
        'sem_legacy_single_rank_dist_sampler',
        'sem_legacy_disable_worker_init',
        'sem_legacy_temporal_window_sampling',
        # ---- Round 9: temporal adaptor ----
        'freeze_backbone',
        'freeze_fr_modules',
        'freeze_head',
        # Note: cross_clip_temporal is a string-type parameter and is not included in bool_int_keys
        # ---- pe_align ----
        'pe_align_enable',
        'pe_align_apply_renorm',
        'pe_align_lr_scale_like_pe',
        'pe_align_layer_decay_enable',
        'pe_align_use_cvqm_npy',
        'pe_align_semantic_legacy_grid',
        'pe_align_legacy_spatial_train_step',
        'pe_align_legacy_iter_lr',
        'pe_align_resize_antialias',
        'pe_align_legacy_no_decay_rule',
        'pe_align_legacy_layer_id_rule',
        'pe_align_legacy_layer_id_prefix_bug',
        'pe_align_legacy_single_rank_dist_sampler',
        'pe_align_legacy_disable_worker_init',
        'pe_align_legacy_temporal_window_sampling',
        'pe_align_container_yuv_direct',
        'train_use_only_vif_branch',
        'eval_use_only_vif_branch',
        'save_only_best_cvqm_inference',
        # ---- swin_align ----
        'swin_align_enable',
        'swin_align_lr_scale_like_pe',
        'swin_align_use_cvqm_npy',
        'swin_align_semantic_legacy_grid',
        'swin_align_legacy_spatial_train_step',
        'swin_align_legacy_iter_lr',
        'swin_align_resize_antialias',
        'swin_align_legacy_single_rank_dist_sampler',
        'swin_align_legacy_disable_worker_init',
        'swin_align_legacy_temporal_window_sampling',
        'swin_align_container_yuv_direct',
        # ---- convnext_align ----
        'convnext_align_enable',
        'convnext_align_lr_scale_like_pe',
        'convnext_align_use_cvqm_npy',
        'convnext_align_semantic_legacy_grid',
        'convnext_align_legacy_spatial_train_step',
        'convnext_align_legacy_iter_lr',
        'convnext_align_resize_antialias',
        'convnext_align_legacy_single_rank_dist_sampler',
        'convnext_align_legacy_disable_worker_init',
        'convnext_align_legacy_temporal_window_sampling',
        'convnext_align_container_yuv_direct',
    }

    # Override with CLI (only non-None values) — also track which keys were set
    # We compare against argparse defaults to distinguish "user typed it" from "default".
    cli_overrides = {}  # {flat_key: original_value} for short run-signature
    parser_defaults = {k: parser.get_default(k) for k in vars(args)}
    args_dict = vars(args)
    for k, v in args_dict.items():
        if k == 'config':
            continue
        if v is None:
            continue
        if k == 'no_amp':
            if v:
                cfg['amp'] = False
                cli_overrides['amp'] = False
            continue
        # Only record as CLI override if value differs from argparse default
        # (i.e., user explicitly passed this flag)
        if v != parser_defaults.get(k):
            cli_overrides[k] = v
        if k in nested_override_paths:
            if k in bool_int_keys:
                v = bool(v)
            # Special: convert comma-separated string to list of ints
            if k == 'sem_topiq_layer_ids' and isinstance(v, str):
                v = [int(x.strip()) for x in v.split(',') if x.strip()]
            _set_nested(cfg, nested_override_paths[k], v)
            continue
        if k in bool_int_keys:
            v = bool(v)
        cfg[k] = v

    # Auto-set semantic_branch.mss when semantic_sampler starts with 'mss'
    sem_sampler = str(cfg.get('semantic_sampler', '')).lower()
    if sem_sampler.startswith('mss'):
        sem_branch = cfg.get('semantic_branch', {})
        if isinstance(sem_branch, dict):
            sem_branch['mss'] = True
            cfg['semantic_branch'] = sem_branch

    _validate_config_keys(cfg)

    # Generate run_signature (new: short mode with CLI overrides)
    cfg['run_signature'] = build_run_signature(cfg, cli_overrides=cli_overrides)

    # Generate parent directory name for hierarchical output organization
    cfg['parent_dir_name'] = build_parent_dir_name(cfg)

    # Store CLI overrides for reference (will be dumped alongside config.json)
    cfg['_cli_overrides'] = cli_overrides

    # Return as DotDict so both cfg['key'] and cfg.key work everywhere
    return cfg_to_dotdict(cfg)


class DotDict(dict):
    """Dict subclass allowing attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def cfg_to_dotdict(cfg: dict) -> DotDict:
    """Convert nested dict to DotDict."""
    dd = DotDict(cfg)
    for k, v in dd.items():
        if isinstance(v, dict):
            dd[k] = cfg_to_dotdict(v)
    return dd
