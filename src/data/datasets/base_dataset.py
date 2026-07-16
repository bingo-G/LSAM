"""
SampleMeta and OldLabelProviderAdapter:
Unified sample metadata structure and adapters for legacy label files.
"""

import os
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# ============================================================================
# Static path profiles
# ============================================================================
# This inference-only release ships with EMPTY profiles.  Dataset paths can
# still be supplied via env vars (HMF_VQA_CVQM_PHASE1_ROOT /
# HMF_VQA_CVQM_PHASE2_ROOT / HMF_VQA_CVQM_REF_ROOT / ...) when driving the
# internal engine directly; the shipped ``infer.py`` CLI does not need any
# of these environment variables.
PATH_PROFILES: Dict[str, Dict[str, str]] = {
    'default': {},
    'rtx4090': {},   # populate on demand
    'hpc':     {},   # populate on demand
}


def _detect_gpu_names() -> List[str]:
    names: List[str] = []
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        for line in out.splitlines():
            line = line.strip()
            if line:
                names.append(line)
        if names:
            return names
    except Exception:
        pass
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            has_cuda = torch.cuda.is_available()
        if has_cuda:
            for i in range(torch.cuda.device_count()):
                names.append(str(torch.cuda.get_device_name(i)))
    except Exception:
        pass
    return names


def _detect_data_profile() -> tuple[str, str]:
    """
    Resolve active data-profile.
    Priority:
      1) HMF_VQA_DATA_PROFILE (manual)
      2) auto detect by GPU model
    """
    raw = os.getenv('HMF_VQA_DATA_PROFILE', 'auto').strip().lower()
    if raw and raw != 'auto':
        return raw, 'env'

    gpu_names = [n.lower() for n in _detect_gpu_names()]
    if any('4090' in n for n in gpu_names):
        return 'rtx4090', 'gpu_auto'
    if any(('a100' in n) or ('v100' in n) for n in gpu_names):
        return 'hpc', 'gpu_auto'

    # CPU-only / hidden-GPU shells: fall back to the profile that has the most
    # actually existing dataset paths on this machine. This is more robust than
    # checking only HMF_VQA_DATA_ROOT, because some machines keep datasets on
    # multiple disks while the aggregate root may not exist.
    best_prof = ''
    best_score = 0
    for prof in ('rtx4090', 'hpc'):
        score = 0
        for path in PATH_PROFILES.get(prof, {}).values():
            path = str(path or '').strip()
            if not path:
                continue
            if os.path.exists(os.path.abspath(os.path.expanduser(path))):
                score += 1
        if score > best_score:
            best_prof = prof
            best_score = score
    if best_prof:
        return best_prof, 'path_auto'

    return 'default', 'gpu_auto'


_ACTIVE_DATA_PROFILE, _DATA_PROFILE_SOURCE = _detect_data_profile()
_PROFILE_ENV_SUFFIX = _ACTIVE_DATA_PROFILE.upper().replace('-', '_')


def _static_profile_value(env_key: str) -> str:
    """
    Read static path value from in-file PATH_PROFILES.
    """
    prof_cfg = PATH_PROFILES.get(_ACTIVE_DATA_PROFILE, {})
    v = str(prof_cfg.get(env_key, '') or '').strip()
    if v:
        return os.path.abspath(os.path.expanduser(v))
    default_cfg = PATH_PROFILES.get('default', {})
    v = str(default_cfg.get(env_key, '') or '').strip()
    if v:
        return os.path.abspath(os.path.expanduser(v))
    return ''


def _resolve_data_root() -> str:
    """
    Resolve base data root with profile-aware fallback.
    Supported env keys (in priority order):
      - HMF_VQA_DATA_ROOT_<PROFILE>
      - HMF_VQA_DATA_ROOT
      - ./datasets (repo-relative)
    """
    if _PROFILE_ENV_SUFFIX:
        v = os.getenv(f'HMF_VQA_DATA_ROOT_{_PROFILE_ENV_SUFFIX}', '').strip()
        if v:
            return os.path.abspath(os.path.expanduser(v))
    v = os.getenv('HMF_VQA_DATA_ROOT', '').strip()
    if v:
        return os.path.abspath(os.path.expanduser(v))
    v = _static_profile_value('HMF_VQA_DATA_ROOT')
    if v:
        return v
    return os.path.join(_REPO_ROOT, 'datasets')


_DEFAULT_DATA_ROOT = _resolve_data_root()


def _env_path(env_key: str, default_rel: str) -> str:
    """
    Resolve dataset path from env var, with profile-aware override:
      1) <ENV_KEY>_<PROFILE>
      2) <ENV_KEY>
      3) <DATA_ROOT>/<default_rel>
    """
    prof_key = f'{env_key}_{_PROFILE_ENV_SUFFIX}' if _PROFILE_ENV_SUFFIX else ''
    if prof_key:
        val = os.getenv(prof_key, '').strip()
        if val:
            return os.path.abspath(os.path.expanduser(val))
    val = os.getenv(env_key, '').strip()
    if val:
        return os.path.abspath(os.path.expanduser(val))
    val = _static_profile_value(env_key)
    if val:
        return val
    return os.path.abspath(os.path.join(_DEFAULT_DATA_ROOT, default_rel))


def resolve_sampled_clip_cache_root(base: Optional[str], dataset_name: str = '') -> Optional[str]:
    """
    Normalize a sampled-clip cache root.

    Supported layouts:
      1) nested: <root>/<dataset>/{ref,dis}/...
      2) flat legacy/downloaded cache: <root>/{ref,dis}/...
    """
    base = str(base or '').strip()
    if not base:
        return None
    base = os.path.abspath(os.path.expanduser(base)).rstrip(os.sep)
    if not base:
        return None

    base_name = os.path.basename(base).lower()
    if base_name in ('sampled_clip_cache_v1', 'sampled_clip_cache'):
        return base
    if base_name.endswith('_frame'):
        return base
    if os.path.isdir(os.path.join(base, 'ref')) or os.path.isdir(os.path.join(base, 'dis')):
        return base

    ds = str(dataset_name or '').strip()
    if ds and os.path.isdir(os.path.join(base, ds)):
        return base
    return os.path.join(base, 'sampled_clip_cache_v1')


def resolve_cvqad_cache_manifest_path(
    cache_root: Optional[str],
    manifest_path: Optional[str] = None,
    dataset_name: str = 'CVQAD',
) -> Optional[str]:
    """Resolve the exported CVQAC/CVQAD label manifest path."""
    manifest = str(manifest_path or '').strip()
    if manifest:
        return os.path.abspath(os.path.expanduser(manifest))
    resolved_root = resolve_sampled_clip_cache_root(cache_root, dataset_name)
    if not resolved_root:
        return None
    return os.path.join(resolved_root, 'cvqac_labels_all_splits.csv')


def resolve_cvqad_cache_index_path(
    cache_root: Optional[str],
    index_path: Optional[str] = None,
    dataset_name: str = 'CVQAD',
) -> Optional[str]:
    """Resolve the exported CVQAC/CVQAD offline cache index path."""
    index = str(index_path or '').strip()
    if index:
        return os.path.abspath(os.path.expanduser(index))
    resolved_root = resolve_sampled_clip_cache_root(cache_root, dataset_name)
    if not resolved_root:
        return None
    return os.path.join(resolved_root, 'cvqac_cache_index_all_splits.csv')


def get_dataset_path_runtime_info() -> dict:
    """Return resolved runtime info for debugging dataset path/profile selection."""
    return {
        'profile': _ACTIVE_DATA_PROFILE,
        'profile_source': _DATA_PROFILE_SOURCE,
        'profile_env_suffix': _PROFILE_ENV_SUFFIX,
        'data_root': _DEFAULT_DATA_ROOT,
        'gpu_names': _detect_gpu_names(),
        'static_cfg_file': 'src/data/datasets/base_dataset.py:PATH_PROFILES',
    }


def _ann_keys_for_split(split: str) -> List[str]:
    split_norm = str(split or 'test').strip().lower()
    if split_norm == 'train':
        return ['train_ann', 'val_ann', 'test_ann']
    if split_norm in ('val', 'validation'):
        return ['val_ann', 'test_ann', 'train_ann']
    return ['test_ann', 'val_ann', 'train_ann']


def _cvqad_split_layouts(cfg: dict, split: str) -> List[dict]:
    split_norm = str(split or 'test').strip().lower()
    base_root = str(cfg.get('root', '') or '').strip()
    test_root = str(cfg.get('test_root', '') or '').strip()
    if not test_root:
        test_root = os.path.join(base_root, 'Test', 'videos') if base_root else ''

    layouts = {
        'train': {
            'name': 'train',
            'dir_name': 'Train',
            'root': os.path.join(base_root, 'Train') if base_root else '',
            'ann_key': 'train_ann',
            'ann_rel': str(cfg.get('train_ann', '') or '').strip(),
        },
        'validation': {
            'name': 'validation',
            'dir_name': 'Validation',
            'root': os.path.join(base_root, 'Validation') if base_root else '',
            'ann_key': 'val_ann',
            'ann_rel': str(cfg.get('val_ann', '') or '').strip(),
        },
        'test': {
            'name': 'test',
            'dir_name': 'Test',
            'root': test_root,
            'ann_key': 'test_ann',
            'ann_rel': str(cfg.get('test_ann', '') or '').strip(),
        },
    }

    if split_norm in ('all', 'all_splits', 'whole', 'full'):
        order = ['train', 'validation', 'test']
    elif split_norm in ('val', 'validation'):
        order = ['validation']
    elif split_norm == 'train':
        order = ['train']
    else:
        order = ['test']

    result = []
    for key in order:
        item = dict(layouts[key])
        root = str(item.get('root', '') or '').strip()
        ann_rel = str(item.get('ann_rel', '') or '').strip()
        item['root_exists'] = bool(root) and os.path.isdir(root)
        item['ann_path'] = os.path.join(root, ann_rel) if root and ann_rel else ''
        item['ann_exists'] = bool(item['ann_path']) and os.path.isfile(item['ann_path'])
        item['ref_root'] = root
        item['ref_root_exists'] = item['root_exists']
        result.append(item)
    return result


def inspect_dataset_layout(name: str, split: str = 'test') -> dict:
    """Collect resolved dataset paths for the active machine/profile."""
    cfg = get_dataset_config(name)
    runtime = get_dataset_path_runtime_info()
    root = str(cfg.get('root', '') or '').strip()
    ref_root = str(cfg.get('ref_root', '') or '').strip()

    ann_key = ''
    ann_rel = ''
    ann_path = ''
    for key in _ann_keys_for_split(split):
        rel = str(cfg.get(key, '') or '').strip()
        if rel:
            ann_key = key
            ann_rel = rel
            ann_path = os.path.join(root, rel) if root else rel
            break

    info = {
        'dataset': name,
        'split': str(split or 'test'),
        'profile': runtime.get('profile'),
        'profile_source': runtime.get('profile_source'),
        'root': root,
        'root_exists': bool(root) and os.path.isdir(root),
        'ref_root': ref_root,
        'ref_root_exists': bool(ref_root) and os.path.isdir(ref_root),
        'ann_key': ann_key,
        'ann_rel': ann_rel,
        'ann_path': ann_path,
        'ann_exists': bool(ann_path) and os.path.isfile(ann_path),
        'phase_roots': {},
        'avt_csv_checks': [],
        'cvqad_split_checks': [],
        'cache_root': '',
        'cache_root_exists': False,
        'cache_manifest': '',
        'cache_manifest_exists': False,
        'cache_index': '',
        'cache_index_exists': False,
    }

    if name in ('CVQAD', 'CVQAC'):
        cvqad_layouts = _cvqad_split_layouts(cfg, split)
        info['cvqad_split_checks'] = cvqad_layouts
        cache_root = resolve_sampled_clip_cache_root(
            cfg.get('cache_root', cfg.get('root', '')),
            'CVQAD',
        )
        cache_manifest = resolve_cvqad_cache_manifest_path(
            cache_root,
            cfg.get('cache_labels', ''),
            'CVQAD',
        )
        cache_index = resolve_cvqad_cache_index_path(
            cache_root,
            cfg.get('cache_index', ''),
            'CVQAD',
        )
        info['cache_root'] = str(cache_root or '')
        info['cache_root_exists'] = bool(cache_root) and os.path.isdir(str(cache_root))
        info['cache_manifest'] = str(cache_manifest or '')
        info['cache_manifest_exists'] = bool(cache_manifest) and os.path.isfile(str(cache_manifest))
        info['cache_index'] = str(cache_index or '')
        info['cache_index_exists'] = bool(cache_index) and os.path.isfile(str(cache_index))
        if cvqad_layouts:
            primary = cvqad_layouts[0]
            info['root'] = str(primary.get('root', '') or '')
            info['root_exists'] = bool(primary.get('root_exists', False))
            info['ref_root'] = str(primary.get('ref_root', '') or '')
            info['ref_root_exists'] = bool(primary.get('ref_root_exists', False))
            info['ann_key'] = str(primary.get('ann_key', '') or '')
            info['ann_rel'] = str(primary.get('ann_rel', '') or '')
            info['ann_path'] = str(primary.get('ann_path', '') or '')
            info['ann_exists'] = bool(primary.get('ann_exists', False))

    if name == 'CVQM':
        phase_roots = {}
        for stage, path in dict(cfg.get('phase_roots', {})).items():
            p = str(path or '').strip()
            phase_roots[int(stage)] = {
                'path': p,
                'exists': bool(p) and os.path.isdir(p),
            }
        info['phase_roots'] = phase_roots

    if name == 'AVT':
        split_norm = str(split or 'all').strip().lower()
        tests = ['1', '2', '3', '4']
        aliases = {
            '1': {'1', 'test1', 'test_1', 'avpvs1', 'avpvs_1'},
            '2': {'2', 'test2', 'test_2', 'avpvs2', 'avpvs_2'},
            '3': {'3', 'test3', 'test_3', 'avpvs3', 'avpvs_3'},
            '4': {'4', 'test4', 'test_4', 'avpvs4', 'avpvs_4'},
        }
        if split_norm not in ('all', 'test', 'val', 'validation'):
            for test_id, keys in aliases.items():
                if split_norm in keys:
                    tests = [test_id]
                    break
        csv_checks = []
        for test_id in tests:
            csv_path = os.path.join(root, f'test_{test_id}_avpvs', f'mos_ci_test{test_id}.csv')
            csv_checks.append({
                'test_id': test_id,
                'path': csv_path,
                'exists': os.path.isfile(csv_path),
            })
        info['avt_csv_checks'] = csv_checks

    return info


def validate_dataset_layout(name: str, split: str = 'test', require_ref: bool = False) -> dict:
    """
    Validate resolved dataset paths for the active machine/profile.
    Raises FileNotFoundError when required layout items are missing.
    """
    info = inspect_dataset_layout(name, split=split)
    errors: List[str] = []

    root = str(info.get('root', '') or '').strip()
    cvqad_has_offline_index = bool(
        name in ('CVQAD', 'CVQAC') and info.get('cache_index_exists', False)
    )
    cvqad_has_manifest = bool(
        name in ('CVQAD', 'CVQAC') and info.get('cache_manifest_exists', False)
    )
    if root and not info.get('root_exists', False) and not cvqad_has_offline_index:
        errors.append(f"dataset root missing: {root}")

    ann_path = str(info.get('ann_path', '') or '').strip()
    if ann_path and not info.get('ann_exists', False) and not cvqad_has_manifest and not cvqad_has_offline_index:
        errors.append(f"annotation missing ({info.get('ann_key', '?')}): {ann_path}")

    ref_root = str(info.get('ref_root', '') or '').strip()
    if require_ref and ref_root and not info.get('ref_root_exists', False) and not cvqad_has_offline_index:
        errors.append(f"reference root missing: {ref_root}")

    if name in ('CVQAD', 'CVQAC'):
        for item in list(info.get('cvqad_split_checks', [])):
            root_i = str(item.get('root', '') or '').strip()
            ann_path_i = str(item.get('ann_path', '') or '').strip()
            ref_root_i = str(item.get('ref_root', '') or '').strip()
            split_i = str(item.get('name', '?'))
            if root_i and not item.get('root_exists', False) and not cvqad_has_offline_index:
                errors.append(f"CVQAD split root missing ({split_i}): {root_i}")
            if ann_path_i and not item.get('ann_exists', False) and not cvqad_has_manifest and not cvqad_has_offline_index:
                errors.append(f"CVQAD split annotation missing ({split_i}): {ann_path_i}")
            if require_ref and ref_root_i and not item.get('ref_root_exists', False) and not cvqad_has_offline_index:
                errors.append(f"CVQAD split reference root missing ({split_i}): {ref_root_i}")

    if name == 'CVQM':
        for stage, stage_info in sorted(dict(info.get('phase_roots', {})).items()):
            if not stage_info.get('exists', False):
                errors.append(f"CVQM phase{stage} root missing: {stage_info.get('path', '')}")

    if name == 'AVT':
        csv_checks = list(info.get('avt_csv_checks', []))
        if not csv_checks:
            errors.append(f"AVT split has no expected AVPVS csv checks: split={split}")
        else:
            missing_csvs = [item['path'] for item in csv_checks if not item.get('exists', False)]
            if missing_csvs:
                errors.append(
                    "AVT AVPVS annotations missing: " + ', '.join(missing_csvs)
                )

    if errors:
        lines = [
            f"Dataset layout validation failed for {name} (split={split}).",
            f"Active profile: {info.get('profile')} (source={info.get('profile_source')})",
        ]
        if root:
            lines.append(f"Resolved root: {root}")
        if ref_root:
            lines.append(f"Resolved ref_root: {ref_root}")
        if ann_path:
            lines.append(f"Resolved ann_path: {ann_path}")
        cache_root = str(info.get('cache_root', '') or '').strip()
        cache_manifest = str(info.get('cache_manifest', '') or '').strip()
        if cache_root:
            lines.append(f"Resolved cache_root: {cache_root}")
        if cache_manifest:
            lines.append(f"Resolved cache_manifest: {cache_manifest}")
        cache_index = str(info.get('cache_index', '') or '').strip()
        if cache_index:
            lines.append(f"Resolved cache_index: {cache_index}")
        for item in list(info.get('cvqad_split_checks', [])):
            lines.append(
                "Resolved CVQAD split "
                f"{item.get('name')}: root={item.get('root', '')}, ann={item.get('ann_path', '')}"
            )
        for stage, stage_info in sorted(dict(info.get('phase_roots', {})).items()):
            lines.append(
                f"Resolved phase{stage}_root: {stage_info.get('path', '')}"
            )
        lines.append("Issues:")
        lines.extend(f"  - {msg}" for msg in errors)
        raise FileNotFoundError('\n'.join(lines))

    return info


@dataclass
class SampleMeta:
    """Unified sample metadata for all datasets."""
    dataset_name: str
    video_id: str
    ref_path: Optional[str]
    dis_path: str
    width: Optional[int]
    height: Optional[int]
    bitdepth: Optional[int]
    pix_fmt: Optional[str]
    mos: float
    # Optional reference-side raw spec (for FR datasets where ref/dis differ).
    ref_width: Optional[int] = None
    ref_height: Optional[int] = None
    ref_bitdepth: Optional[int] = None
    ref_pix_fmt: Optional[str] = None
    extra: dict = field(default_factory=dict)
    split: str = 'train'  # train/val/test
    stage: Optional[int] = None  # CVQM: 1/2

    @property
    def is_raw(self) -> bool:
        ext = os.path.splitext(self.dis_path)[1].lower()
        return ext in {'.yuv', '.y4m', '.nv12', '.raw'}

    @property
    def is_fr(self) -> bool:
        return self.ref_path is not None and self.ref_path != ''


# ============================================================================
# Dataset path configuration
# ============================================================================
# The inference-only release ships one built-in parser (``CVQM``). ``infer.py``
# additionally registers an ad-hoc ``Custom`` parser at runtime for user-
# supplied videos passed through ``--ref/--dis``, ``--ref_dir/--dis_dir`` or
# ``--manifest`` — that path does not consult ``DATASET_CONFIGS`` at all.

DATASET_CONFIGS = {
    'CVQM': {
        'root': _env_path('HMF_VQA_CVQM_ROOT', 'CVQM'),
        'val_ann': 'CVQM_label_all_phase.xlsx',
        'test_ann': 'CVQM_label_all_phase.xlsx',
        'phase_roots': {
            1: _env_path('HMF_VQA_CVQM_PHASE1_ROOT', 'CVQM/phase1'),
            2: _env_path('HMF_VQA_CVQM_PHASE2_ROOT', 'CVQM/decoded'),
        },
        'ref_root': _env_path('HMF_VQA_CVQM_REF_ROOT', 'CVQM/refs'),
        'npy_root': _env_path('HMF_VQA_CVQM_NPY_ROOT', 'CVQM/CVQM_npy_Phase12'),
        'frame_root': _env_path('HMF_VQA_CVQM_FRAME_ROOT', 'CVQM/CVQM_Image_Phase12'),
    },
}


def get_dataset_config(name: str) -> dict:
    """Get config for a dataset by name."""
    if name not in DATASET_CONFIGS:
        raise KeyError(f"Dataset '{name}' not found. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name]
