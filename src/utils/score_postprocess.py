"""
Score post-processing for LSAM.

Maps the raw model output into the reported score range using a
resolution-aware, five-parameter logistic mapping. Two coefficient sets
ship out of the box:

  - HD  : videos with height  < ``UHD_HEIGHT_THRESHOLD`` (1500)
  - UHD : videos with height >= ``UHD_HEIGHT_THRESHOLD``

The mapping is applied uniformly to every video (single-pair, ref/dis-dir
and manifest modes) so the reported scores are directly comparable across
input modes and resolutions.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger('lsam.postprocess')


# Mapping coefficients (b1..b5) — shipped with the release.
_Coeffs = Tuple[float, float, float, float, float]

HD_COEFFS: _Coeffs = (
    0.2155385522, 26.0085223087, 0.4305192813, 1.2321556541, -0.1325352893,
)
UHD_COEFFS: _Coeffs = (
    1.6101304553,  4.4159440347, 0.4479988469, 0.1754582140,  0.2548359324,
)

# Per-sample selector: height >= threshold → UHD coefficients.
UHD_HEIGHT_THRESHOLD: int = 1500


def _map(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    """Apply one shipped coefficient set (five-parameter logistic form)."""
    x = np.asarray(x, dtype=np.float64)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def apply_score_mapping(
    pred_raw: Iterable[float],
    *,
    heights: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Map raw model outputs to the reported score range.

    Per-sample selector: ``heights[i] >= UHD_HEIGHT_THRESHOLD`` picks the UHD
    coefficient set, otherwise HD. When ``heights`` is not provided the HD
    mapping is used for every sample (1080p is the most common input).
    """
    pred = np.asarray(list(pred_raw), dtype=np.float64)
    if pred.size == 0:
        return pred
    if heights is None:
        return _map(pred, *HD_COEFFS)
    h = np.asarray(list(heights), dtype=np.int64)
    out = pred.copy()
    is_uhd = h >= UHD_HEIGHT_THRESHOLD
    if is_uhd.any():
        out[is_uhd] = _map(pred[is_uhd], *UHD_COEFFS)
    if (~is_uhd).any():
        out[~is_uhd] = _map(pred[~is_uhd], *HD_COEFFS)
    return out


# ---------------------------------------------------------------------------
# Correlation helpers (used by infer.py when MOS labels are available)
# ---------------------------------------------------------------------------

def compute_correlations(pred: Sequence[float], target: Sequence[float]) -> dict:
    """SRCC / PLCC / KRCC / RMSE between two equal-length sequences."""
    try:
        from scipy.stats import pearsonr, spearmanr, kendalltau
    except Exception:
        logger.warning('scipy unavailable — correlation values will be NaN')
        return {'SRCC': float('nan'), 'PLCC': float('nan'),
                'KRCC': float('nan'), 'RMSE': float('nan'), 'n': len(pred)}
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    n = int(min(len(p), len(t)))
    if n < 2:
        return {'SRCC': float('nan'), 'PLCC': float('nan'),
                'KRCC': float('nan'), 'RMSE': float('nan'), 'n': n}
    p = p[:n]; t = t[:n]
    try:
        srcc = float(spearmanr(p, t)[0])
    except Exception:
        srcc = float('nan')
    try:
        plcc = float(pearsonr(p, t)[0])
    except Exception:
        plcc = float('nan')
    try:
        krcc = float(kendalltau(p, t)[0])
    except Exception:
        krcc = float('nan')
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return {'SRCC': srcc, 'PLCC': plcc, 'KRCC': krcc, 'RMSE': rmse, 'n': n}


__all__ = [
    'HD_COEFFS',
    'UHD_COEFFS',
    'UHD_HEIGHT_THRESHOLD',
    'apply_score_mapping',
    'compute_correlations',
]
