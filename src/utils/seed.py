"""
Seed utilities for reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int, cudnn_deterministic: bool = False):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int):
    """DataLoader worker init function for reproducibility."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
