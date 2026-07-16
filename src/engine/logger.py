"""
Logger: JSON-line + CSV + console logging for training metrics.

Writes:
  - log.txt: JSON-line format (one JSON dict per epoch), compatible with plot_curves.py
  - train_log.csv / val_log.csv: CSV format
  - Console output via Python logging module.

JSON-line format per epoch:
  {"epoch": 0, "train_loss": 0.123, "train_lr": 1e-4,
   "val_CVQM_self_srcc": 0.85, "val_CVQM_self_plcc": 0.87, "val_CVQM_self_loss": 0.05,
   "test_CVQM_srcc": 0.80, "test_CVQM_plcc": 0.82,
   "test_CVQM_s1_srcc": 0.78, "test_CVQM_s1_plcc": 0.80,
   "test_CVQM_s2_srcc": 0.75, "test_CVQM_s2_plcc": 0.77, ...}
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from ..utils.dist import is_main_process


def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Create a logger with console + optional file handler.
    Console handler only added on rank 0 to avoid duplicate prints in DDP."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler: only rank 0
    if is_main_process():
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        # Non-rank-0: set to WARNING to suppress INFO/DEBUG console output
        logger.setLevel(logging.WARNING)

    if log_file and is_main_process():
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class JSONLineLogger:
    """
    JSON-line logger: writes one JSON dict per epoch to log.txt.
    Compatible with plot_curves.py from the old codebase.
    Only writes on rank 0.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._initialized = False

    def _init_file(self):
        if not is_main_process():
            return
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
        self._initialized = True

    def log(self, row: Dict):
        """Write a single JSON line to log.txt."""
        if not is_main_process():
            return
        if not self._initialized:
            self._init_file()

        with open(self.filepath, 'a') as f:
            f.write(json.dumps(row) + '\n')


class CSVLogger:
    """
    Append-mode CSV logger for metrics.
    Only writes on rank 0.
    """

    def __init__(self, filepath: str, fieldnames: list):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._initialized = False

    def _init_file(self):
        if not is_main_process():
            return
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        self._initialized = True

    def log(self, row: Dict):
        if not is_main_process():
            return
        if not self._initialized:
            self._init_file()

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            writer.writerow(row)


class MetricTracker:
    """Track best metrics for early stopping / checkpointing."""

    def __init__(self, patience: int = 20, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = -1
        self.counter = 0

    def update(self, value: float, epoch: int) -> bool:
        """
        Update tracker. Returns True if this is a new best.
        """
        improved = False
        if self.mode == 'max' and value > self.best_value:
            improved = True
        elif self.mode == 'min' and value < self.best_value:
            improved = True

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return improved

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience

