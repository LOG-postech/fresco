"""The weights each compute phase reads, and how they are obtained.

A corrector declares, per compute phase, the realization that gives its weights
(``Snapshot``, ``Extrapolate`` or ``Backtrack``). The ``ParameterVersionManager``
collects every corrector's reads (one per phase), owns the realizations and
their state, and serves them to both engines: the weights, eagerly
(``weights``), and a binding for a GraphStage (``binding``).

Each realization implements both forms: ``weights`` (new tensors, the current
weights untouched) and ``binding`` (lib/versions/bindings.py), and, if it keeps
state across steps, its update eagerly (``before_step`` / ``after_step``) and
in the step graph (``transform``).
"""

from __future__ import annotations

import abc
import collections
import enum
import logging
from collections.abc import Sequence
from typing import Any, Literal, Optional

import torch
from torch import nn

from .bindings import BindExpr, BindLive, BindSnapshot, WeightBinding

# The weights a pass reads: {parameter: tensor}; parameters not listed read
# their current value.
Weights = dict[nn.Parameter, torch.Tensor]


class ComputePhase(enum.Enum):
    """A pass of a micro-batch on a stage whose weights a corrector decides."""

    FORWARD = 'forward'
    BACKWARD = 'backward'


class Realization(abc.ABC):
    """How a phase's weights are obtained, in both engines' forms.

    Bound by the ParameterVersionManager (``bind``) to the phase it serves;
    reads the stage (parameters, delay, base optimizer) through the manager.
    """

    manager: ParameterVersionManager
    phase: ComputePhase

    def bind(
        self, manager: ParameterVersionManager, phase: ComputePhase
    ) -> None:
        self.manager, self.phase = manager, phase

    @property
    def tau(self) -> int:
        """The stage's delay: optimizer steps between a forward and its
        update."""
        return self.manager.tau

    def initialize(self) -> None:
        """Before training (allocate state)."""

    def attach(self) -> None:
        """A GraphStage takes over: drop eager-only state."""

    def state_dict(self) -> dict:
        """The persistent state (tensors keyed by parameter index)."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restores state_dict()'s state (after initialize)."""
        del state

    # ---- eager form ----
    @abc.abstractmethod
    def weights(self) -> Weights:
        """The weights the phase reads (new tensors; the current untouched)."""

    def record(self, read: Weights) -> None:
        """At a forward, before it runs: the weights it reads (``read``), the
        parameters still holding the current ones."""

    def before_step(self) -> None:
        """Before the base optimizer step."""

    def after_step(self) -> None:
        """After the base optimizer step."""

    # ---- graph form ----
    @abc.abstractmethod
    def binding(self) -> WeightBinding:
        """The phase's weight binding for a GraphStage."""

    @property
    def transform(self) -> Any:
        """An UpdateTransform (lib/graph/primitives.py) the step graph runs:
        the graph form of before_step / after_step, or None."""
        return None


class Snapshot(Realization):
    """The weights kept at the micro-batch's forward [PipeDream's stashing].

    Each forward keeps one copy of the weights on the CPU (FIFO; 1F1B backwards
    run in forward order); its phase reads it (consumed). Under a GraphStage
    the slots keep the forward-time weights instead (BindSnapshot).
    """

    @property
    def active(self) -> bool:
        return self.tau > 0  # one weight version: nothing in flight

    def initialize(self) -> None:
        # Per in-flight micro-batch: its weights.
        self.queue: Optional[collections.deque[Weights]] = collections.deque()

    def state_dict(self) -> dict:
        """The in-flight micro-batches' copies (none after a drain)."""
        return {
            'queue': [self.manager.indexed(s) for s in self.queue or ()]
        }

    def load_state_dict(self, state: dict) -> None:
        for stashed in state['queue']:
            weights = self.manager.by_param(stashed)
            self.queue.append({p: w.cpu() for p, w in weights.items()})

    def attach(self) -> None:
        self.queue = None  # the stage's slots keep the weights

    def binding(self) -> WeightBinding:
        return BindSnapshot() if self.active else BindLive()

    def record(self, read: Weights) -> None:
        if not self.active or self.queue is None:
            return
        self.queue.append(
            {
                p: read.get(p, p.data).detach().to('cpu', copy=True)
                for p in self.manager.params()
            }
        )

    def weights(self) -> Weights:
        """The oldest in-flight micro-batch's copy (consumed)."""
        if not self.active or not self.queue:
            return {}
        return {
            p: w.to(p.device, non_blocking=True)
            for p, w in self.queue.popleft().items()
        }


def _adamw_update_direction(
    state: dict, lr: float, betas: tuple[float, float], eps: float
) -> Optional[torch.Tensor]:
    """Compute -lr * m_hat / (sqrt(v_hat) + eps) for a single param.

    Returns None if state not initialized (no prior step).
    """
    if 'exp_avg' not in state:
        return None
    m = state['exp_avg']
    v = state['exp_avg_sq']
    step = state.get('step', 1)
    # step can be a 0-d tensor in newer PyTorch
    if torch.is_tensor(step):
        step = step.item()
    beta1, beta2 = betas
    bias_corr1 = 1.0 - beta1**step
    bias_corr2 = 1.0 - beta2**step
    if bias_corr1 == 0 or bias_corr2 == 0:
        return None
    m_hat = m / bias_corr1
    v_hat = v / bias_corr2
    # AdamW direction: -lr * m_hat / (sqrt(v_hat) + eps)
    return -lr * m_hat / (v_hat.sqrt() + eps)


class Extrapolate(Realization):
    """The weights extrapolated along an update direction: W + s * dW, s
    optimizer steps ahead of the weights current at the phase.

    New tensors each time (the current weights untouched): the eager hooks
    point ``p.data`` at them and back, so restoring is exact (adding and then
    subtracting the offset is not: it left ~10% of the weights one ulp off per
    micro-batch); a GraphStage binds them as a BindExpr.

    Args:
        rule: 'optimizer': dW from the base optimizer's state; 'weight_diff':
            dW = W_t - W_{t-1} (forward only: each forward advances W_{t-1}).
        horizon: s as a multiple of tau.
        offset: 'half_stage' adds stage // 2 to s (SpecTrain).
    """

    def __init__(
        self,
        rule: Literal['optimizer', 'weight_diff'] = 'optimizer',
        horizon: float = 1.0,
        offset: Literal['none', 'half_stage'] = 'none',
    ):
        self.rule, self.horizon, self.offset = rule, horizon, offset

    def bind(
        self, manager: ParameterVersionManager, phase: ComputePhase
    ) -> None:
        super().bind(manager, phase)
        if self.rule == 'weight_diff' and phase is not ComputePhase.FORWARD:
            raise ValueError(
                'weight_diff prediction is forward-only: each forward '
                'advances W_{t-1}'
            )

    @property
    def s(self) -> float:
        """The extrapolation distance, in optimizer steps."""
        half = self.manager.stage // 2 if self.offset == 'half_stage' else 0
        return self.horizon * self.tau + half

    def initialize(self) -> None:
        self._prev: Weights = {}  # weight_diff: W_{t-1}, as of the last forward
        logging.info(
            '[predict] stage=%d tau=%d rule=%s s=%g',
            self.manager.stage,
            self.tau,
            self.rule,
            self.s,
        )

    def state_dict(self) -> dict:
        return {
            'prev': self.manager.indexed(self._prev)
        }  # weight_diff's W_{t-1}

    def load_state_dict(self, state: dict) -> None:
        self._prev = {
            p: w.clone()
            for p, w in self.manager.by_param(state['prev']).items()
        }

    def binding(self) -> WeightBinding:
        return BindExpr(self.weights) if self.s else BindLive()

    def weights(self) -> Weights:
        """``{p: W + s * dW}``, as new tensors."""
        if not self.s:
            return {}
        if self.rule == 'weight_diff':
            return self._weight_diff()
        base = self.manager.base_optimizer
        if base is None:
            return {}
        out = {}
        with torch.no_grad():
            for group in base.param_groups:
                lr = group['lr']
                betas = group.get('betas', (0.9, 0.999))
                eps = group.get('eps', 1e-8)
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = base.state.get(p, {})
                    delta = _adamw_update_direction(state, lr, betas, eps)
                    if delta is not None:
                        out[p] = torch.add(p.data, delta, alpha=float(self.s))
        return out

    def _weight_diff(self) -> Weights:
        """``{p: W_t + s * (W_t - W_{t-1})}``; records W_t as the new W_{t-1}.

        Called once per forward. Training is one continuous pipeline run, so
        consecutive forwards are one optimizer step apart and W_t - W_{t-1}
        is the last applied update; on a parameter's first forward W_{t-1} is
        unknown: it is seeded and the weight is unchanged.
        """
        out = {}
        with torch.no_grad():
            for p in self.manager.params():
                prev = self._prev.get(p)
                if prev is None:
                    self._prev[p] = p.data.detach().clone()
                    continue
                delta = p.data - prev  # W_t - W_{t-1}
                prev.copy_(p.data)  # the buffer now holds W_t
                out[p] = torch.add(p.data, delta, alpha=float(self.s))
        return out


class Backtrack(Realization):
    """The weights backtracked along the update EMA [PipeMare's T2]: the
    weights ``W - tau * delta``, with ``delta`` an EMA of the applied updates
    (recovering the weights the micro-batch's forward saw, tau steps back).

    The EMA is updated around every step, eagerly or in the stage's step graph
    (``transform``).

    Args:
        discrepancy_correction: T2's D; the EMA's gamma = D ** (1 / tau).
    """

    def __init__(self, discrepancy_correction: float = 0.1):
        self.discrepancy_correction = discrepancy_correction

    def bind(
        self, manager: ParameterVersionManager, phase: ComputePhase
    ) -> None:
        super().bind(manager, phase)
        self.gamma = (
            self.discrepancy_correction ** (1.0 / self.tau)
            if self.tau > 0
            else 0.0
        )

    def initialize(self) -> None:
        # Per-parameter EMA of the true update.
        self.delta = {
            p: torch.zeros_like(p.data, dtype=torch.float32)
            for p in self.manager.params()
        }
        self.before: Optional[Weights] = None  # the weights before the step

    def state_dict(self) -> dict:
        return {'delta': self.manager.indexed(self.delta)}

    def load_state_dict(self, state: dict) -> None:
        for p, d in self.manager.by_param(state['delta']).items():
            self.delta[p].copy_(d)

    def binding(self) -> WeightBinding:
        return BindExpr(self.weights) if self.tau > 0 else BindLive()

    def weights(self) -> Weights:
        """``{p: W - tau * EMA(update)}``, as new tensors (W untouched)."""
        if self.tau <= 0:
            return {}
        with torch.no_grad():
            return {
                p: (p.data.float() - self.tau * delta).to(p.dtype)
                for p, delta in self.delta.items()
            }

    # The EMA update: eagerly, the weights before the step are a transient
    # copy; in the step graph (prepare / apply) they need a fixed buffer,
    # allocated only then.
    def before_step(self) -> None:
        if self.tau > 0:
            self._old = {p: p.data.clone() for p in self.manager.params()}

    def after_step(self) -> None:
        if self.tau > 0:
            self._track(self._old)
            self._old = None

    @property
    def transform(self) -> Optional[Backtrack]:
        return self if self.tau > 0 else None

    def init_state(self, ctx: Any) -> None:
        del ctx
        if self.before is None:
            self.before = {
                p: torch.empty_like(p.data) for p in self.manager.params()
            }

    def prepare(self, ctx: Any) -> None:
        """Keeps the weights before the step."""
        del ctx
        for p, w in self.before.items():
            w.copy_(p.data)

    def apply(self, ctx: Any) -> None:
        del ctx
        self._track(self.before)

    def _track(self, before: Weights) -> None:
        """delta = gamma * delta + (1 - gamma) * (applied update)."""
        if self.tau <= 0:
            return
        for p, w in before.items():
            self.delta[p].mul_(self.gamma).add_(
                p.data.float() - w.float(), alpha=1.0 - self.gamma
            )


class ParameterVersionManager:
    """Carries out the correctors' weight reads, one per compute phase.

    Owns the declared realizations (binds them, runs their lifecycle and step
    hooks) and serves each phase's weights: eagerly (``weights``) and as a
    GraphStage binding (``binding``). A phase's reader is a Realization, or an
    imperative corrector (references.py) with the same ``weights`` / ``record``
    / ``binding`` hooks, whose lifecycle its optimizer runs.

    Args:
        tau: The stage's delay: optimizer steps between a forward and its
            update.
        modules: The stage's modules (their trainable parameters are read).
        base_optimizer: The base optimizer (Extrapolate reads its state).
        stage: The stage's index in the pipeline.
    """

    def __init__(
        self,
        tau: int,
        modules: Sequence[nn.Module],
        base_optimizer: Optional[torch.optim.Optimizer] = None,
        stage: int = 0,
    ):
        self.tau, self.modules = tau, modules
        self.base_optimizer, self.stage = base_optimizer, stage
        self._readers: dict[ComputePhase, Any] = {}
        self._owners: dict[ComputePhase, str] = {}
        self.realizations: list[Realization] = []

    def params(self) -> list[nn.Parameter]:
        return [
            p for m in self.modules for p in m.parameters() if p.requires_grad
        ]

    def indexed(self, weights: Weights) -> dict[int, torch.Tensor]:
        """``weights`` keyed by parameter index (loads into a new model)."""
        index = {p: i for i, p in enumerate(self.params())}
        return {index[p]: w for p, w in weights.items()}

    def by_param(self, indexed: dict[int, torch.Tensor]) -> Weights:
        params = self.params()
        return {params[i]: w.to(params[i].device) for i, w in indexed.items()}

    def declare(self, phase: ComputePhase, reader: Any, owner: str) -> None:
        """``owner``'s reader of ``phase`` (one declaration per phase).

        Raises:
            ValueError: If another corrector already declared ``phase``.
        """
        if phase in self._owners:
            raise ValueError(
                f'{owner} and {self._owners[phase]} both declare the '
                f'{phase.value} weights'
            )
        self._owners[phase], self._readers[phase] = owner, reader
        if isinstance(reader, Realization):
            reader.bind(self, phase)
            self.realizations.append(reader)

    # ---- the realizations' lifecycle ----
    def initialize(self) -> None:
        for r in self.realizations:
            r.initialize()

    def attach(self) -> None:
        for r in self.realizations:
            r.attach()

    def before_step(self) -> None:
        for r in self.realizations:
            r.before_step()

    def after_step(self) -> None:
        for r in self.realizations:
            r.after_step()

    @property
    def transforms(self) -> list[Any]:
        """The realizations' update transforms, for the step graph."""
        return [
            r.transform for r in self.realizations if r.transform is not None
        ]

    # ---- serving the reads ----
    def weights(self, phase: ComputePhase) -> Weights:
        """The weights ``phase`` reads; at the forward, every reader records
        them."""
        reader = self._readers.get(phase)
        read = reader.weights() if reader is not None else {}
        if phase is ComputePhase.FORWARD:
            for r in self._readers.values():
                r.record(read)
        return read

    def binding(self, phase: ComputePhase) -> WeightBinding:
        """``phase``'s binding for a GraphStage."""
        reader = self._readers.get(phase)
        return (reader.binding() if reader is not None else None) or BindLive()
