"""Per-stage learning-rate schedule, in optimizer steps.

Linear warmup, then cosine decay to ``min_lr_ratio * lr`` at the final
optimizer step, times the learning-rate damper's factor
(lib/correctors/dampers.py, PipeMare's T1).
"""

from __future__ import annotations

import logging
import math
import typing

import torch

if typing.TYPE_CHECKING:
    import configs
    from ..correctors.base import AsyncPipelineOptimizer


def get_learning_rate(
    sched: configs.ScheduleConfig, base_lr: float, step: int, total_steps: int
) -> float:
    """Returns the learning rate of one stage at one optimizer step.

    Args:
        sched: The schedule settings (a configs.ScheduleConfig).
        base_lr: The stage's peak learning rate (sched.peak_value, adjusted for the
            stage's replicas by the runtime).
        step: The optimizer step, 0-based.
        total_steps: The number of optimizer steps in the run.

    Returns:
        The learning rate.
    """
    warmup = min(sched.warmup_steps, total_steps)

    if step < warmup:
        lr = (
            sched.warmup_init
            + (base_lr - sched.warmup_init) * (step + 1) / warmup
        )
    else:
        decay_steps = max(1, total_steps - 1 - warmup)
        progress = min(1.0, (step - warmup) / decay_steps)  # 0 -> 1 at the end
        min_lr = sched.min_lr_ratio * base_lr
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
    return lr


def adjust_learning_rate(
    optimizer: AsyncPipelineOptimizer,
    sched: configs.ScheduleConfig,
    base_lr: float,
    step: int,
    total_steps: int,
) -> float:
    """Sets the learning rate of one stage for one optimizer step.

    Args:
        optimizer: The stage's staleness method (its param_groups get the LR).
        sched: The schedule settings (a configs.ScheduleConfig).
        base_lr: The stage's peak learning rate (see get_learning_rate).
        step: The optimizer step, 0-based.
        total_steps: The number of optimizer steps in the run.

    Returns:
        The learning rate set.
    """
    lr = get_learning_rate(sched, base_lr, step, total_steps)
    lr *= optimizer.lr_scale(step)
    if step % 1000 == 0:
        logging.info('Step %d \tLearning rate: %f', step, lr)
    for param_group in optimizer.param_groups:
        if torch.is_tensor(param_group['lr']):  # a captured step reads it
            param_group['lr'].fill_(lr)
        else:
            param_group['lr'] = lr
    return lr
