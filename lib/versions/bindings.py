"""Weight bindings: a realization's graph form (lib/versions/parameters.py).

Which tensor stands in for each weight in a pass a GraphStage traces:
``BindLive`` (the current weights), ``BindSnapshot`` (the weights the
micro-batch's forward saw, kept in its slot) or ``BindExpr`` (computed from the
current weights, a prediction or a recovery, without modifying them). Binding
matters where autograd saved a parameter itself (LayerNorm gains, matmuls).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Union


@dataclasses.dataclass(frozen=True)
class BindLive:
    """The current weights."""


@dataclasses.dataclass(frozen=True)
class BindSnapshot:
    """The weights the micro-batch's forward saw, kept in its slot."""


@dataclasses.dataclass(frozen=True)
class BindExpr:
    """Weights computed from the current ones, without changing them.

    ``fn()`` returns ``{param: tensor}`` for the parameters that differ (a
    predicted or recovered weight, e.g. W + s * direction); the others read
    their current values. It runs eagerly, right before the pass.
    """

    fn: Any


WeightBinding = Union[BindLive, BindSnapshot, BindExpr]
