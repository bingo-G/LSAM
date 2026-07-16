"""
Evaluation metrics for VQA: PLCC, SRCC, KRCC, RMSE.
Non-differentiable; used only at eval time.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional


def compute_plcc(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson Linear Correlation Coefficient."""
    if len(pred) < 2:
        return 0.0
    if np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    val, _ = stats.pearsonr(pred, target)
    return float(val)


def compute_srcc(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman Rank-order Correlation Coefficient."""
    if len(pred) < 2:
        return 0.0
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    val, _ = stats.spearmanr(pred, target)
    return float(np.nan_to_num(val, nan=0.0))


def compute_krcc(pred: np.ndarray, target: np.ndarray) -> float:
    """Kendall Rank Correlation Coefficient."""
    if len(pred) < 2:
        return 0.0
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    val, _ = stats.kendalltau(pred, target)
    return float(np.nan_to_num(val, nan=0.0))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    prefix: str = '',
) -> Dict[str, float]:
    """
    Compute all VQA metrics.

    Args:
        pred: [N] predictions
        target: [N] ground truth
        prefix: optional prefix for metric keys

    Returns:
        dict with PLCC, SRCC, KRCC, RMSE
    """
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()

    p = prefix + '_' if prefix else ''
    return {
        f'{p}PLCC': compute_plcc(pred, target),
        f'{p}SRCC': compute_srcc(pred, target),
        f'{p}KRCC': compute_krcc(pred, target),
        f'{p}RMSE': compute_rmse(pred, target),
    }


def compute_cvqm_stage_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    stages: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per CVQM stage (1 and 2) and overall.

    Args:
        pred: [N] predictions
        target: [N] ground truth
        stages: [N] stage labels (1 or 2)

    Returns:
        dict with 'stage1', 'stage2', 'overall' metric dicts
    """
    results = {}
    results['overall'] = compute_all_metrics(pred, target)

    for stage in [1, 2]:
        mask = stages == stage
        if mask.sum() > 1:
            results[f'stage{stage}'] = compute_all_metrics(
                pred[mask], target[mask]
            )
        else:
            results[f'stage{stage}'] = {
                'PLCC': 0.0, 'SRCC': 0.0, 'KRCC': 0.0, 'RMSE': 0.0,
            }

    return results
