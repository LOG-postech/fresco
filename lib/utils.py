"""Training helpers: seeding and running averages."""

import functools
import os
import random
from collections.abc import Callable

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seeds Python hashing, NumPy and PyTorch (CPU and every GPU).

    Args:
        seed: The RNG seed.
        deterministic: Use deterministic cuDNN kernels (reproducible, slower)
            instead of autotuned ones.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _seed_worker(seed: int, worker_id: int) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def worker_init_fn(seed: int) -> Callable[[int], None]:
    """Returns a DataLoader worker_init_fn seeding worker i with seed + i."""
    return functools.partial(_seed_worker, seed)


class AverageMeter:
    """Tracks the latest value and the running average of a metric.

    Attributes:
        val: The latest value.
        avg: The average, weighted by each update's n.
        sum: The weighted sum.
        count: The total weight.
    """

    val: float
    avg: float
    sum: float
    count: float

    def __init__(self) -> None:
        self.val = self.avg = self.sum = self.count = 0

    def reset(self) -> None:
        """Clears the meter."""
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val: float, n: float = 1) -> None:
        """Records a value with weight n (e.g. a batch loss and batch size)."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
