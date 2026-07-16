"""
Backbone registry and factory.

This release only ships the **PE** backbone family (PE-Core B16 / L14 / etc.),
which is what the released LSAM configuration uses. SwinV2 and ConvNeXtV2
backbones were removed; the factory raises ``ValueError`` for any non-PE
variant string.

All PE backbones share a common interface:
  - __init__(variant, pretrained, weights_path, **kwargs)
  - forward(x: [B, 3, H, W]) -> [B, out_dim]
  - out_dim: int  (output feature dimension)

Usage:
    from src.models.backbones import build_backbone
    backbone = build_backbone('pe_b16', pretrained=True, pe_weights='...')
"""

import logging
from typing import Optional

from .pe_encoder import PEEncoder

logger = logging.getLogger(__name__)


def _detect_family(name: str) -> str:
    """Return the backbone family for ``name``. Always ``'pe'`` in this release.

    Kept as a function (rather than an inline string) so callers don't have to
    change when a future release adds more families.
    """
    n = name.lower().strip()
    if n.startswith('pe') or n.startswith('PE'):
        return 'pe'
    raise ValueError(
        f"Unsupported backbone variant '{name}': only the PE family "
        f"(pe_b16, pe_l14, ...) is shipped in this release."
    )


def build_backbone(
    variant: str,
    pretrained: bool = True,
    *,
    pe_weights: Optional[str] = None,
    img_size: int = 224,   # kept for signature compatibility; PE backbones ignore it
    **kwargs,
):
    """Unified backbone factory (PE family only).

    Args:
        variant: Backbone variant string, e.g. ``'pe_b16'``.
        pretrained: Whether to load pretrained weights.
        pe_weights: Path to a PE weights file.
        img_size: Unused for PE; kept for signature backward compatibility.
        **kwargs: Forwarded to ``PEEncoder``.

    Returns:
        nn.Module with an ``out_dim`` attribute.
    """
    family = _detect_family(variant)  # raises for non-PE
    assert family == 'pe', f'unexpected family {family}'

    backbone = PEEncoder(
        variant=variant,
        pretrained=pretrained,
        weights_path=pe_weights,
        **kwargs,
    )
    logger.info(
        "build_backbone: variant=%s family=%s out_dim=%d pretrained=%s",
        variant, family, backbone.out_dim, pretrained,
    )
    return backbone
