"""Loss functions for HMF-VQA."""

from .mse import MSELoss
from .rank import PairwiseRankLoss
from .plcc import PLCCLoss
from .fidelity import PairwiseFidelityLoss
from .metrics import compute_all_metrics, compute_cvqm_stage_metrics

__all__ = [
    'MSELoss', 'PairwiseRankLoss', 'PLCCLoss', 'PairwiseFidelityLoss',
    'compute_all_metrics', 'compute_cvqm_stage_metrics',
]


def build_criterion(cfg):
    """
    Build combined loss from config.

    Reads `cfg.losses` (str like 'mse,rank,plcc,fidelity') and
    `cfg.lambda_*` for weights.  Pairwise / correlation losses receive
    ``buffer_size = grad_accum`` so that they accumulate samples across
    gradient-accumulation steps (critical when batch_size=1).

    Returns a dict of {name: (loss_fn, weight)}.
    """
    import logging
    _log = logging.getLogger('hmf_vqa')

    # Parse which losses are enabled
    loss_names_str = getattr(cfg, 'losses', 'mse,rank,plcc')
    if isinstance(loss_names_str, str):
        enabled = [s.strip().lower() for s in loss_names_str.split(',') if s.strip()]
    elif isinstance(loss_names_str, (list, tuple)):
        enabled = [str(s).strip().lower() for s in loss_names_str]
    else:
        enabled = ['mse']

    # Buffer is primarily needed for tiny effective-batch regimes where
    # pairwise/correlation losses are extremely noisy (e.g. B=1~2).
    # Heuristic: enable buffering while effective batch is small.
    grad_accum = int(getattr(cfg, 'grad_accum', 4))
    batch_size = int(getattr(cfg, 'batch_size', 1))
    effective_batch = int(max(1, batch_size) * max(1, grad_accum))
    buf = grad_accum if (grad_accum > 1 and effective_batch <= 8) else 0

    # fidelity_buffer_mode: buffering policy for the Fidelity Loss.
    #   'auto'  — use the heuristic above (default, backwards-compatible).
    #   'force' — when grad_accum > 1, always enable the buffer so the pairwise
    #             loss is computed over the accumulated big batch, which
    #             greatly increases the pair count (B=6, ga=8 → 48 samples →
    #             C(48,2) = 1128 pairs vs 15 pairs per step). Note that only
    #             the last mini-batch's gradients are live.
    fidelity_buf_mode = str(getattr(cfg, 'fidelity_buffer_mode', 'auto')).lower()
    if fidelity_buf_mode == 'force' and grad_accum > 1:
        fid_buf = grad_accum
    else:
        fid_buf = buf

    losses = {}

    # MSE  (pointwise — no buffer needed)
    if 'mse' in enabled:
        mse_w = float(getattr(cfg, 'lambda_mse', 1.0))
        mse_norm = getattr(cfg, 'mse_normalize', False)
        losses['mse'] = (MSELoss(normalize=mse_norm), mse_w)

    # Rank loss  (pairwise)
    if 'rank' in enabled:
        rank_w = float(getattr(cfg, 'lambda_rank', 0.1))
        margin = float(getattr(cfg, 'rank_margin', 0.05))
        losses['rank'] = (PairwiseRankLoss(margin=margin, buffer_size=buf), rank_w)

    # PLCC loss  (correlation)
    if 'plcc' in enabled:
        plcc_w = float(getattr(cfg, 'lambda_plcc', 0.1))
        losses['plcc'] = (PLCCLoss(buffer_size=buf), plcc_w)

    # Fidelity loss  (pairwise Bhattacharyya)
    if 'fidelity' in enabled:
        fid_w = float(getattr(cfg, 'lambda_fidelity', 0.1))
        fid_alpha = float(getattr(cfg, 'fidelity_alpha', 10.0))
        losses['fidelity'] = (
            PairwiseFidelityLoss(alpha=fid_alpha, buffer_size=fid_buf), fid_w,
        )

    # Fallback: always have at least MSE
    if not losses:
        losses['mse'] = (MSELoss(), 1.0)

    weight_str = ', '.join(f'{k}={v[1]}' for k, v in losses.items())
    buf_str = (
        f'(pairwise buffer={buf}, fidelity_buffer={fid_buf}, batch_size={batch_size}, '
        f'grad_accum={grad_accum}, effective_batch={effective_batch})'
    )
    _log.info(f'Losses: {list(losses.keys())} (weights: {weight_str}) {buf_str}')
    return losses


def reset_all_loss_buffers(criterion: dict):
    """Reset internal buffers of every pairwise / correlation loss.

    Must be called once per optimizer step so that old detached samples
    do not leak across optimisation boundaries.
    """
    for _name, (loss_fn, _weight) in criterion.items():
        if hasattr(loss_fn, 'reset_buffer'):
            loss_fn.reset_buffer()
