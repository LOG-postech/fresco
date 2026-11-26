# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Weight consistency as three slots around a base optimizer.

A staleness method decides, per micro-batch, three things (configs.py):

    forward   the weights the forward reads          (Current / Predict)
    backward  the weights the backward reads          (Current / Stash /
                                                        Recover / Predict)
    update    how the base update is applied          (Base / Fresco)

plus learning-rate damping (dampers.py), which combines with any slot choice. Each mechanism is an AsyncPipelineCorrector; AsyncPipelineOptimizer
composes one per slot, plus the dampers. Per micro-batch the training loop
calls::

    before_forward(); forward; after_forward()
    zero_grad(); load_old_params(); backward; load_new_params(); step()

plus ``initialize_queue`` once before training, ``train`` / ``eval``, and
``state_dict`` / ``load_state_dict`` (the base optimizer's state and every
corrector's).
A GraphStage reads the same slots as primitives (``spec``) instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional

import torch
from torch import nn

from ..graph import primitives
from ..versions import ComputePhase, ParameterVersionManager, Realization, Weights


class Version:
    """Monotonic weight-version counter (one per optimizer step)."""

    __slots__ = ('version',)

    def __init__(self, version: int = 0):
        self.version = version

    def __repr__(self) -> str:
        return 'v%d' % self.version

    def incr(self) -> Version:
        return Version(self.version + 1)


class AsyncPipelineCorrector:
    """A mechanism that corrects one aspect of asynchronous-pipeline
    staleness: the weights a pass reads (a forward / backward slot), how the
    update is applied (the update slot), or a damper of the stale update.

    It declares the weights compute phases read (``reads``: a realization
    per phase, carried out by the optimizer's ParameterVersionManager in both
    engines), and implements its change to the update as hooks
    (``before_step`` / ``after_step`` eagerly, ``transform`` in the step
    graph; ``lr_scale`` for the learning rate).

    A corrector may instead give its slot's weights imperatively (``weights``
    / ``record`` eagerly, ``binding`` for a GraphStage), as the readable
    references in references.py do. A hook it does not override does nothing,
    but each eager hook needs its graph form: one that overrides ``weights``
    without ``binding``, or a step hook without ``transform``, runs eagerly
    only, and a GraphStage gets NotImplementedError instead of silently
    skipping its work. Bound to its AsyncPipelineOptimizer (``bind``), whose
    stage geometry and base optimizer it reads through the properties below.
    """

    method: AsyncPipelineOptimizer
    # The phase of the slot it fills (FORWARD / BACKWARD), or None.
    phase: Optional[ComputePhase] = None

    def bind(
        self,
        method: AsyncPipelineOptimizer,
        phase: Optional[ComputePhase] = None,
    ) -> None:
        self.method, self.phase = method, phase

    @property
    def reads(self) -> dict[ComputePhase, Realization]:
        """The weights it declares, per compute phase (after bind)."""
        return {}

    @property
    def modules(self) -> Sequence[nn.Module]:
        return self.method.modules

    @property
    def base_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.method.base_optimizer

    @property
    def num_versions(self) -> int:
        return self.method.num_versions

    @property
    def tau(self) -> int:
        """The stage's delay: optimizer steps between a forward and its
        update."""
        return self.num_versions - 1

    @property
    def stage(self) -> int:
        return self.method.stage

    @property
    def latest_version(self) -> Version:
        return self.method.latest_version

    def params(self) -> list[nn.Parameter]:
        return [
            p for m in self.modules for p in m.parameters() if p.requires_grad
        ]

    def initialize(self) -> None:
        """Before training (allocate state)."""

    def attach(self) -> None:
        """A GraphStage takes over: drop eager-only state and hooks."""

    # ---- checkpointing: the state the corrector carries across steps ----
    def state_dict(self) -> dict:
        """The corrector's persistent state (tensors keyed by parameter
        index, so it loads into a new model)."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restores state_dict()'s state (after initialize)."""
        del state

    def _indexed(self, weights: Weights) -> dict[int, torch.Tensor]:
        index = {p: i for i, p in enumerate(self.params())}
        return {index[p]: w for p, w in weights.items()}

    def _by_param(self, indexed: dict[int, torch.Tensor]) -> Weights:
        params = self.params()
        return {params[i]: w.to(params[i].device) for i, w in indexed.items()}

    # ---- eager hooks ----
    def weights(self) -> Weights:
        """Imperatively, the weights this slot's pass reads (new tensors; the
        current weights untouched)."""
        return {}

    def record(self, read: Weights) -> None:
        """At a forward, before it runs: the weights it reads (``read``, as
        weights()), the parameters still holding the current ones."""

    def before_step(self) -> None:
        """Before the base optimizer step."""

    def after_step(self) -> None:
        """After the base optimizer step."""

    def finish_step(self) -> None:
        """After a step a GraphStage ran as a graph."""

    def lr_scale(self, step: int) -> float:
        """A factor on the scheduled learning rate at an optimizer step."""
        del step
        return 1.0

    # ---- graph: this corrector as primitives ----
    def binding(self) -> Optional[primitives.WeightBinding]:
        """Imperatively, the slot's weight binding (None: the current).

        Raises:
            NotImplementedError: If ``weights`` is overridden but this is not
                (an eager-only corrector).
        """
        if self._overrides('weights'):
            raise NotImplementedError(
                f'{type(self).__name__} reads other weights eagerly '
                '(weights) but has no binding for a GraphStage'
            )
        return None

    @property
    def transform(self) -> Optional[primitives.UpdateTransform]:
        """An UpdateTransform the step graph runs, or None.

        Raises:
            NotImplementedError: If a step hook is overridden but this is not
                (an eager-only corrector).
        """
        if self._overrides('before_step') or self._overrides('after_step'):
            raise NotImplementedError(
                f'{type(self).__name__} changes the update eagerly '
                '(before_step / after_step) but has no transform for a step '
                'graph'
            )
        return None

    def imperative(self) -> bool:
        """Whether it gives its slot's weights by hooks, not ``reads``."""
        return self._overrides('weights') or self._overrides('record')

    def _overrides(self, hook: str) -> bool:
        """Whether this corrector's class overrides ``hook``."""
        return getattr(type(self), hook) is not getattr(
            AsyncPipelineCorrector, hook
        )

    @property
    def tap(self) -> Optional[primitives.LinearInputTap]:
        """A LinearInputTap traced into the forward graph, or None."""
        return None


class Current(AsyncPipelineCorrector):
    """The current weights: no correction in this slot."""


class AsyncPipelineOptimizer(torch.optim.Optimizer):
    """The three slots composed around a base ``torch.optim`` optimizer.

    The base optimizer (NAdamW, built by ``lib.optim.base_optimizer``) computes
    the update; this class owns the per-step scaffolding (gradient clipping,
    the version counter) and calls the slots' hooks.

    Args:
        base_optimizer: The base optimizer over the stage's parameters, or None
            when the stage has no parameters to optimize.
        modules: The stage's modules.
        num_versions: In-flight micro-batches + 1 (the stage's delay + 1).
        forward: The forward slot's AsyncPipelineCorrector (default Current).
        backward: The backward slot's AsyncPipelineCorrector (default Current).
        update: The update slot's AsyncPipelineCorrector (default Current: the base
            update).
        dampers: Learning-rate damping (dampers.py): an LR factor.
        stage: The stage's index in the pipeline (from the runtime).
        clip_grad: Gradient-norm clip, or None.
        verbose_freq: Verbose-logging frequency (0: off).
    """

    def __init__(
        self,
        base_optimizer: Optional[torch.optim.Optimizer],
        modules: Sequence[nn.Module],
        num_versions: int,
        forward: Optional[AsyncPipelineCorrector] = None,
        backward: Optional[AsyncPipelineCorrector] = None,
        update: Optional[AsyncPipelineCorrector] = None,
        dampers: Iterable[AsyncPipelineCorrector] = (),
        stage: int = 0,
        clip_grad: Optional[float] = None,
        verbose_freq: int = 0,
    ):
        self.modules = modules
        self.stage = stage
        self.attached = False  # True once attach() hands over to a GraphStage
        self.clip_grad = clip_grad
        self.num_versions = max(1, num_versions)
        self.base_optimizer = base_optimizer
        if self.base_optimizer is None:
            logging.info('Warning: no parameter groups to optimize')
        self.latest_version = Version()
        self.batch_counter = 0
        self.forward = forward or Current()
        self.backward = backward or Current()
        self.update = update or Current()
        self.dampers = list(dampers)
        # After the base step: the update slot projects the update, the read
        # realizations' state (e.g. Backtrack's EMA) tracks the result.
        self._step_order = [*self.dampers, self.update, self.backward]
        self._correctors = [self.forward, *self._step_order]
        self.forward.bind(self, ComputePhase.FORWARD)
        self.backward.bind(self, ComputePhase.BACKWARD)
        for c in [self.update, *self.dampers]:
            c.bind(self)
        # The weights each compute phase reads, from every corrector.
        self.versions = ParameterVersionManager(
            self.num_versions - 1,
            modules,
            base_optimizer,
            stage=stage,
        )
        for name, c in self._named_correctors().items():
            owner = f'{name} ({type(c).__name__})'
            for phase, realization in c.reads.items():
                self.versions.declare(phase, realization, owner)
            if c.phase is not None and c.imperative():
                self.versions.declare(c.phase, c, owner)
        self._swapped: Weights = {}
        self.initialize_queue()

    # ---- graph ----
    @property
    def spec(self) -> primitives.Spec:
        """The slots as primitives (lib.graph.primitives), for a GraphStage."""
        forward = self.versions.binding(ComputePhase.FORWARD)
        transforms = [
            c.transform for c in self._step_order if c.transform is not None
        ] + self.versions.transforms
        update = transforms[0] if transforms else None  # at most one per method
        return primitives.Spec(
            forward_weights=(
                None if isinstance(forward, primitives.BindLive) else forward
            ),
            backward_weights=self.versions.binding(ComputePhase.BACKWARD),
            linear_tap=self.update.tap,
            update=update,
        )

    def attach(self) -> None:
        """Hands the method's weight / activation handling to a GraphStage.

        After this, the stage provides ``spec`` inside its graphs and the
        method keeps no weight copies and swaps no weights itself.
        """
        self.attached = True
        for c in self._correctors:
            c.attach()
        self.versions.attach()

    def finish_step(self) -> None:
        """Bookkeeping after a step the stage ran as a graph."""
        for c in self._correctors:
            c.finish_step()
        self.latest_version = self.latest_version.incr()
        self.batch_counter += 1

    # ---- checkpointing: the base optimizer's state and every corrector's ----
    def _named_correctors(self) -> dict[str, AsyncPipelineCorrector]:
        named = {'forward': self.forward, 'backward': self.backward}
        named['update'] = self.update
        named.update({f'damper{i}': d for i, d in enumerate(self.dampers)})
        return named

    def state_dict(self) -> dict:
        """The base optimizer's state, every corrector's, and the counters."""
        return {
            'base': (
                None
                if self.base_optimizer is None
                else self.base_optimizer.state_dict()
            ),
            'correctors': {
                name: {'type': type(c).__name__, 'state': c.state_dict()}
                for name, c in self._named_correctors().items()
            },
            'version': self.latest_version.version,
            'batch_counter': self.batch_counter,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores state_dict(); a checkpoint of only the base optimizer's
        state (before the correctors were saved) restores just that.

        Raises:
            ValueError: If the checkpoint's correctors are not this method's.
        """
        if 'correctors' not in state:  # an older checkpoint
            if self.base_optimizer is not None:
                self.base_optimizer.load_state_dict(state)
            return
        saved = state['correctors']
        mine = self._named_correctors()
        kinds = {name: c['type'] for name, c in saved.items()}
        if kinds != {name: type(c).__name__ for name, c in mine.items()}:
            raise ValueError(
                f"checkpoint correctors {kinds} are not this method's "
                f'{ {n: type(c).__name__ for n, c in mine.items()} }'
            )
        if self.base_optimizer is not None:
            self.base_optimizer.load_state_dict(state['base'])
        for name, c in mine.items():
            c.load_state_dict(saved[name]['state'])
        self.latest_version = Version(state['version'])
        self.batch_counter = state['batch_counter']

    def __getattr__(self, key: str) -> Any:
        """Relays unknown attributes to the base optimizer.

        E.g. ``param_groups``, ``state``.
        """
        opt = self.__dict__.get('base_optimizer')
        if opt is None:  # empty parameter list
            return (
                {}
                if key == 'state'
                else ([{'params': []}] if key == 'param_groups' else None)
            )
        return getattr(opt, key)

    def zero_grad(self) -> None:
        if self.base_optimizer is not None:
            self.base_optimizer.zero_grad()

    def lr_scale(self, step: int) -> float:
        """The dampers' factor on the scheduled learning rate."""
        scale = 1.0
        for c in self._correctors:
            scale *= c.lr_scale(step)
        return scale

    def train(self) -> None:
        train = getattr(self.base_optimizer, 'train', None)
        if train is not None:
            train()

    def eval(self) -> None:
        evaluate = getattr(self.base_optimizer, 'eval', None)
        if evaluate is not None:
            evaluate()

    # ---- eager hooks: point the parameters at each pass's weights ----
    def initialize_queue(self) -> None:
        for c in self._correctors:
            c.initialize()
        self.versions.initialize()

    def _swap_in(self, weights: Weights) -> None:
        for p, w in weights.items():
            self._swapped[p] = p.data
            p.data = w

    def _swap_out(self) -> None:
        """Points the parameters back (exactly: no add-then-subtract)."""
        for p, w in self._swapped.items():
            p.data = w
        self._swapped = {}

    def before_forward(self) -> None:
        self._swap_in(self.versions.weights(ComputePhase.FORWARD))

    def after_forward(self) -> None:
        self._swap_out()

    def load_old_params(self) -> None:
        self._swap_in(self.versions.weights(ComputePhase.BACKWARD))

    def load_new_params(self) -> None:
        self._swap_out()

    def _do_step(self) -> Optional[float]:
        with torch.no_grad():
            for c in self._step_order:
                c.before_step()
            self.versions.before_step()
        loss = (
            self.base_optimizer.step()
            if self.base_optimizer is not None
            else None
        )
        with torch.no_grad():
            for c in self._step_order:
                c.after_step()
            self.versions.after_step()
        self.latest_version = self.latest_version.incr()
        return loss

    # ---- shared per-step scaffolding ----
    def _scale_and_clip_grads(self) -> None:
        if self.clip_grad is not None:
            for group in self.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        self._scale_and_clip_grads()
        loss = self._do_step()
        self.batch_counter += 1
        return loss
