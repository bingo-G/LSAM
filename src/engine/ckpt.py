"""
Checkpoint manager: save/load model + optimizer + scheduler + epoch.

Features:
  - Save best + latest checkpoints
  - Resume from checkpoint
  - DDP-safe (only rank 0 saves, all ranks load)
"""

import os
import torch
import logging
from typing import Optional, Dict, Any

from ..utils.dist import is_main_process

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    output_dir: str,
    filename: str = 'checkpoint.pth',
) -> Optional[str]:
    """
    Save checkpoint (rank 0 only).

    Args:
        state: dict with 'model', 'optimizer', 'scheduler', 'epoch', etc.
        output_dir: directory to save to
        filename: checkpoint filename

    Returns:
        filepath if saved, None otherwise
    """
    if not is_main_process():
        return None

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved: {filepath}')
    return filepath


def save_best_checkpoint(
    state: Dict[str, Any],
    output_dir: str,
    metric_name: str = 'SRCC',
    metric_value: float = 0.0,
):
    """Save best model: full state (for resume) + lightweight (inference only)."""
    state['best_metric_name'] = metric_name
    state['best_metric_value'] = metric_value
    # Full state (can resume training from best)
    save_checkpoint(state, output_dir, filename='best_model.pth')
    # Lightweight inference-only weights (no optimizer/scheduler, smaller file)
    if is_main_process():
        inference_state = {
            'model': state['model'],
            'epoch': state.get('epoch', 0),
            'best_metric_name': metric_name,
            'best_metric_value': metric_value,
        }
        filepath = os.path.join(output_dir, 'best_model_inference.pth')
        torch.save(inference_state, filepath)
        logger.info(f'Best inference weights saved: {filepath} '
                    f'({metric_name}={metric_value:.4f})')


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint.

    All ranks load (each reads from disk). For DDP, the model should be
    the unwrapped module (before wrapping with DDP).

    Args:
        filepath: path to checkpoint
        model: model to load weights into
        optimizer: optional optimizer to restore
        scheduler: optional scheduler to restore
        strict: strict loading for state_dict

    Returns:
        checkpoint dict (for accessing epoch, metrics, etc.)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Checkpoint not found: {filepath}')

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    # Handle DDP state_dict prefix
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', {}))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    if is_main_process():
        n_missing = len(missing)
        n_unexpected = len(unexpected)
        if n_missing or n_unexpected:
            logger.warning(
                'Checkpoint load mismatch (strict=%s): missing=%d, unexpected=%d',
                strict, n_missing, n_unexpected,
            )
            if n_missing:
                logger.warning('  Missing keys (head): %s', missing[:8])
            if n_unexpected:
                logger.warning('  Unexpected keys (head): %s', unexpected[:8])

    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            if is_main_process():
                logger.warning(f'Failed to load optimizer state: {e}')

    if scheduler is not None and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception as e:
            if is_main_process():
                logger.warning(f'Failed to load scheduler state: {e}')

    epoch = checkpoint.get('epoch', 0)
    checkpoint['_load_report'] = {
        'strict': bool(strict),
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
    }
    if is_main_process():
        logger.info(f'Loaded checkpoint from {filepath}, epoch={epoch}')
    return checkpoint


def get_resume_path(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in output_dir for resuming."""
    latest = os.path.join(output_dir, 'checkpoint.pth')
    if os.path.isfile(latest):
        return latest
    return None
