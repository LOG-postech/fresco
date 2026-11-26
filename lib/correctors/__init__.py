"""Weight consistency: three slots, one mechanism each, around a base optimizer.

``forward`` (the weights the forward reads), ``backward`` (the weights the
backward reads) and ``update`` (how the base update is applied), plus
learning-rate damping; see configs.py for the paper's methods as slot choices.
AsyncPipelineOptimizer composes them.
"""

import dataclasses
from typing import Any, Optional

import torch
from torch import nn

from .base import AsyncPipelineCorrector, Current, AsyncPipelineOptimizer
from .fresco import FrescoProjection
from .prediction import Predict
from .recovery import Recover
from .stashing import Stash
from .dampers import StalenessAwareLRDamper

# {configs.py mechanism dataclass name: its AsyncPipelineCorrector}; the dataclass's fields
# are exactly the AsyncPipelineCorrector's arguments.
CORRECTORS = {
    'Current': Current,
    'Base': Current,
    'Predict': Predict,
    'Stash': Stash,
    'Recover': Recover,
    'Fresco': FrescoProjection,
}


def corrector(mechanism: Optional[Any]) -> AsyncPipelineCorrector:
    """The AsyncPipelineCorrector of a configs.py mechanism dataclass (None: Current)."""
    if mechanism is None:
        return Current()
    cls = CORRECTORS[type(mechanism).__name__]
    return cls(**dataclasses.asdict(mechanism))


def num_weight_versions(warmup_minibatches: int) -> int:
    """Returns the stage's in-flight micro-batches + 1 (its delay + 1).

    Args:
        warmup_minibatches: The stage's in-flight minibatches at steady state
            (StageRuntime.num_warmup_minibatches).
    """
    return warmup_minibatches + 1


def build(
    base_optimizer: Optional[torch.optim.Optimizer],
    modules: list[nn.Module],
    num_versions: int,
    forward: Optional[Any] = None,
    backward: Optional[Any] = None,
    update: Optional[Any] = None,
    staleness_lr_steps: int = 0,
    **kwargs: Any,
) -> AsyncPipelineOptimizer:
    """Builds a staleness method around a base optimizer.

    Args:
        base_optimizer: The base optimizer over the stage's parameters, or None
            when the stage has none.
        modules: The stage's modules.
        num_versions: num_weight_versions().
        forward, backward, update: The slots' configs.py mechanisms (None:
            the current weights / the base update).
        staleness_lr_steps: The staleness-aware LR damper's K (0: off).
        **kwargs: AsyncPipelineOptimizer's other keyword arguments.

    Returns:
        The staleness method (a AsyncPipelineOptimizer).
    """
    return AsyncPipelineOptimizer(
        base_optimizer,
        modules,
        num_versions,
        forward=corrector(forward),
        backward=corrector(backward),
        update=corrector(update),
        dampers=(
            [StalenessAwareLRDamper(staleness_lr_steps)]
            if staleness_lr_steps
            else []
        ),
        **kwargs,
    )


__all__ = [
    'CORRECTORS',
    'AsyncPipelineCorrector',
    'Current',
    'FrescoProjection',
    'Predict',
    'Recover',
    'StalenessAwareLRDamper',
    'AsyncPipelineOptimizer',
    'Stash',
    'build',
    'corrector',
    'num_weight_versions',
]
