"""
Plotter: generate and save metric plots during training.

Produces:
  - From JSON-line log.txt (primary, compatible with old plot_curves.py):
    - loss_curve.png: Train + val + test losses
    - accuracy_curve.png: All datasets SRCC/PLCC
    - learning_rate.png: LR schedule
    - per-dataset val_{name}_curve.png / test_{name}_curve.png
  - Legacy single-dataset plots:
    - Loss curve (train)
    - SRCC/PLCC curves (val)
    - Scatter plots (pred vs MOS)

Only operates on rank 0. Gracefully degrades if matplotlib unavailable.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional

from ..utils.dist import is_main_process

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning('matplotlib not available; plotting disabled.')


# ============================================================================
# Main entry: plot from JSON-line log.txt (compatible with old plot_curves.py)
# ============================================================================

def plot_from_jsonlog(output_dir: str):
    """
    Read log.txt (JSON-line format) and generate all training curves.
    This is the primary plotting function called by Trainer every epoch.

    Generates:
      - loss_curve.png: train loss + all val/test losses
      - accuracy_curve.png: all datasets SRCC/PLCC on one chart
      - learning_rate.png: LR schedule
      - val_{name}_curve.png: per val-dataset SRCC+PLCC
      - test_{name}_curve.png: per test-dataset SRCC+PLCC (incl. phase1/phase2)
    """
    if not is_main_process() or not HAS_MPL:
        return

    log_path = os.path.join(output_dir, 'log.txt')
    if not os.path.exists(log_path):
        return

    # Read JSON lines
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue

    if not data:
        return

    def _extract(items, key):
        xs, ys = [], []
        for it in items:
            if 'epoch' not in it or key not in it:
                continue
            val = it[key]
            if val is None:
                continue
            xs.append(it['epoch'])
            ys.append(val)
        return xs, ys

    # ---- Discover val/test dataset names ----
    val_datasets = set()
    test_datasets = set()
    for item in data:
        for key in item.keys():
            if key.startswith('val_') and key.endswith('_srcc'):
                name = key[4:-5]  # strip 'val_' and '_srcc'
                val_datasets.add(name)
            elif key.startswith('test_') and key.endswith('_srcc') and '_s1_' not in key and '_s2_' not in key:
                name = key[5:-5]  # strip 'test_' and '_srcc'
                test_datasets.add(name)

    def _get_srcc_plcc(data, prefix, name):
        s_x, s_y = _extract(data, f'{prefix}_{name}_srcc')
        p_x, p_y = _extract(data, f'{prefix}_{name}_plcc')
        return s_x, s_y, p_x, p_y

    # ======== 1. Loss Curve ========
    try:
        plt.figure(figsize=(10, 6))
        train_x, train_y = _extract(data, 'train_loss')
        if train_y:
            plt.plot(train_x, train_y, label='Train Loss', linewidth=2)
        for name in sorted(val_datasets):
            vx, vy = _extract(data, f'val_{name}_loss')
            if vy:
                plt.plot(vx, vy, label=f'Val {name} Loss', linestyle='--')
        for name in sorted(test_datasets):
            tx, ty = _extract(data, f'test_{name}_loss')
            if ty:
                plt.plot(tx, ty, label=f'Test {name} Loss', linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f'Failed to plot loss curve: {e}')

    # ======== 2. Accuracy Curve (all datasets SRCC/PLCC) ========
    try:
        plt.figure(figsize=(12, 6))
        colors = plt.cm.tab10.colors
        color_idx = 0

        for name in sorted(val_datasets):
            s_x, s_y, p_x, p_y = _get_srcc_plcc(data, 'val', name)
            c = colors[color_idx % len(colors)]
            if s_y:
                plt.plot(s_x, s_y, label=f'Val {name} SRCC', color=c, linestyle='-')
            if p_y:
                plt.plot(p_x, p_y, label=f'Val {name} PLCC', color=c, linestyle='--')
            color_idx += 1

        for name in sorted(test_datasets):
            s_x, s_y, p_x, p_y = _get_srcc_plcc(data, 'test', name)
            c = colors[color_idx % len(colors)]
            if s_y:
                plt.plot(s_x, s_y, label=f'Test {name} SRCC', color=c,
                         linestyle='-', marker='o', markersize=3)
            if p_y:
                plt.plot(p_x, p_y, label=f'Test {name} PLCC', color=c,
                         linestyle='--', marker='s', markersize=3)
            color_idx += 1

        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.title('Correlation Curve (SRCC & PLCC)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f'Failed to plot accuracy curve: {e}')

    # ======== 3. Learning Rate ========
    try:
        lr_x, lr_y = _extract(data, 'train_lr')
        if lr_y:
            plt.figure(figsize=(10, 6))
            plt.plot(lr_x, lr_y, label='Learning Rate', color='red', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.warning(f'Failed to plot LR curve: {e}')

    # ======== 4. Per val-dataset curves ========
    for name in sorted(val_datasets):
        try:
            s_x, s_y, p_x, p_y = _get_srcc_plcc(data, 'val', name)
            if not s_y and not p_y:
                continue
            plt.figure(figsize=(10, 6))
            if s_y:
                plt.plot(s_x, s_y, label='SRCC', color='blue', linewidth=2)
            if p_y:
                plt.plot(p_x, p_y, label='PLCC', color='orange', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Correlation')
            plt.title(f'Validation Correlation - {name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'val_{name}_curve.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f'Failed to plot val_{name} curve: {e}')

    # ======== 5. Per test-dataset curves (with phase1/phase2 if CVQM) ========
    for name in sorted(test_datasets):
        try:
            s_x, s_y, p_x, p_y = _get_srcc_plcc(data, 'test', name)
            if not s_y and not p_y:
                continue

            plt.figure(figsize=(10, 6))
            if s_y:
                plt.plot(s_x, s_y, label='SRCC (All)', color='blue',
                         linewidth=2, marker='o', markersize=4)
            if p_y:
                plt.plot(p_x, p_y, label='PLCC (All)', color='orange',
                         linewidth=2, marker='s', markersize=4)

            # Phase1 / Phase2 curves (if present, e.g. CVQM)
            for short, phase_label, ls in [('s1', 'Phase1', '-.'), ('s2', 'Phase2', ':')]:
                ps_x, ps_y = _extract(data, f'test_{name}_{short}_srcc')
                pp_x, pp_y = _extract(data, f'test_{name}_{short}_plcc')
                if ps_y:
                    plt.plot(ps_x, ps_y, label=f'SRCC ({phase_label})',
                             color='green' if short == 's1' else 'red',
                             linewidth=1.5, linestyle=ls)
                if pp_y:
                    plt.plot(pp_x, pp_y, label=f'PLCC ({phase_label})',
                             color='green' if short == 's1' else 'red',
                             linewidth=1.5, linestyle=ls, alpha=0.6)

            plt.xlabel('Epoch')
            plt.ylabel('Correlation')
            plt.title(f'Test Correlation - {name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'test_{name}_curve.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f'Failed to plot test_{name} curve: {e}')


# ============================================================================
# Legacy single-dataset plotting functions (kept for backward compatibility)
# ============================================================================

def plot_loss_curve(
    train_losses: List[float],
    output_dir: str,
    filename: str = 'loss_curve.png',
):
    """Plot training loss over epochs."""
    if not is_main_process() or not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_metric_curves(
    metrics_history: Dict[str, List[float]],
    output_dir: str,
    filename: str = 'metric_curves.png',
):
    """Plot SRCC/PLCC/KRCC curves over epochs."""
    if not is_main_process() or not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'SRCC': 'blue', 'PLCC': 'orange', 'KRCC': 'green', 'RMSE': 'red'}

    for name, values in metrics_history.items():
        if name == 'RMSE':
            continue  # separate scale
        color = colors.get(name, 'gray')
        ax.plot(range(1, len(values) + 1), values, '-o', color=color,
                label=name, markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    ax.set_title('Validation Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_scatter(
    pred: np.ndarray,
    target: np.ndarray,
    output_dir: str,
    filename: str = 'scatter.png',
    title: str = 'Pred vs MOS',
):
    """Plot scatter of predictions vs ground truth."""
    if not is_main_process() or not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target, pred, alpha=0.4, s=15, edgecolors='none')

    # Diagonal line
    mn = min(target.min(), pred.min())
    mx = max(target.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5)

    ax.set_xlabel('Ground Truth (MOS)')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
