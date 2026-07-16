from .base_dataset import (
    SampleMeta,
    get_dataset_config,
    DATASET_CONFIGS,
    get_dataset_path_runtime_info,
    inspect_dataset_layout,
    resolve_sampled_clip_cache_root,
    validate_dataset_layout,
)
from .cvqm import parse_cvqm, split_cvqm_by_stage

# Parser registry. Only the CVQM benchmark parser (the dataset the released
# checkpoint was trained on) ships in the release; ``infer.py`` additionally
# registers an ad-hoc ``Custom`` parser at runtime for user-supplied videos
# passed through ``--ref/--dis``, ``--ref_dir/--dis_dir``, or ``--manifest``.
PARSERS = {
    'CVQM': parse_cvqm,
}


def get_parser(name: str):
    """Return the parser function registered for ``name``ed (case-insensitive)."""
    if name in PARSERS:
        return PARSERS[name]
    for k in PARSERS:
        if k.lower() == name.lower():
            return PARSERS[k]
    raise KeyError(
        f"No parser for dataset '{name}'. Available: {list(PARSERS.keys())}"
    )
