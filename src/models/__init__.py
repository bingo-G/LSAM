from .hmf_vqa import HMFVQA
from .adapters.colorspace_adapter import ColorSpaceAdapter
from .branches.semantic_branch import SemanticBranch
from .fusion import FusionHead, ScaleToken, RAPE, TemporalShift, Aggregator

__all__ = [
    'HMFVQA',
    'ColorSpaceAdapter',
    'SemanticBranch',
    'FusionHead', 'ScaleToken', 'RAPE', 'TemporalShift', 'Aggregator',
]


def build_model(cfg):
    """Build the HMFVQA model from config.

    This release only ships the FR PE-semantic path (SD3 / ZG2 / LSAM recipes).
    Earlier NR-only model variants and the VIF / Detail branches have been
    removed; all other code paths in the original codebase reduced to no-ops
    under the released configs, so behaviour is bit-identical.
    """
    return HMFVQA(cfg)
