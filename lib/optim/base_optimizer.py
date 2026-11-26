"""Builds the base optimizer that a staleness method wraps.

The base optimizer (a stock torch.optim optimizer) computes the update; the
asynchronous-pipeline staleness method around it (lib/correctors/) decides which
weights the backward and the update see. Every method uses the same base
optimizer, from cfg.optimizer.
"""

from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

if typing.TYPE_CHECKING:
    import configs


def build_base_optimizer(
    optimizer_cfg: configs.OptimizerConfig,
    lr: float,
    params: Iterable[nn.Parameter],
) -> Optional[torch.optim.Optimizer]:
    """Returns the NAdamW base optimizer over `params`.

    Args:
        optimizer_cfg: configs.OptimizerConfig (momentum = beta1,
            momentum_decay, weight_decay).
        lr: The initial learning rate (the schedule sets it every step).
        params: The parameters to optimize.

    Returns:
        A torch.optim.NAdam (decoupled weight decay), or None when `params` is
        empty.
    """
    params = list(params)
    if not params:
        return None
    return torch.optim.NAdam(
        params,
        lr=lr,
        betas=(optimizer_cfg.momentum, 0.999),
        weight_decay=optimizer_cfg.weight_decay,
        decoupled_weight_decay=True,
        momentum_decay=optimizer_cfg.momentum_decay,
    )
