"""The primitives a staleness method is built from on a GraphStage.

A method states what it needs as a ``Spec`` of these primitives; the stage
compiles them into its forward, backward and step graphs, so the method keeps
no weight copies, registers no hooks and swaps no ``param.data``.

1. Weight binding  Which tensor stands in for each weight in a pass:
                   ``BindLive``, ``BindSnapshot`` or ``BindExpr``, the graph
                   form of the parameter-version realizations
                   (lib/versions/bindings.py).
2. Tap             A computation at a node where a tensor already exists;
                   ``LinearInputTap`` runs on every nn.Linear input inside the
                   forward graph (FRESCO's input-feature basis).
3. State           Tap outputs live in the micro-batch's slot (one buffer per
                   slot, fixed addresses); a method's persistent tensors
                   are allocated once and updated in place.
4. Update transform  ``UpdateTransform.prepare`` / ``apply`` run in the step
                   graph before / after the base optimizer step (FRESCO's
                   projection); ``StepContext`` gives them the tap outputs of
                   the in-flight micro-batches.
"""

import dataclasses
from typing import Any, Optional, Protocol

import torch

# 1. weight binding: a realization's graph form, defined in lib/versions.
from ..versions.bindings import (  # noqa: F401 (re-exported)
    BindExpr,
    BindLive,
    BindSnapshot,
    WeightBinding,
)


# ---- 2. taps ----
class LinearInputTap(Protocol):
    """A computation on each nn.Linear's input, captured in the forward graph.

    Linears that read the same input tensor with the same ``key`` share one
    call (e.g. q / k / v). ``inputs`` and ``result`` run eagerly, between
    graph launches; ``compute`` is traced into the graph, so it must be a pure
    tensor function (no host syncs, no data-dependent control flow).
    """

    def key(self, linear: torch.nn.Linear) -> Any:
        """Linears with equal keys and the same input share one call."""
        ...

    def inputs(
        self,
        linear: torch.nn.Linear,
        x_shape: torch.Size,
        dry_run: bool = False,
    ) -> list[torch.Tensor]:
        """Per-forward inputs of ``compute`` for this Linear.

        With ``dry_run`` (graph building), returns tensors of the right shapes
        without drawing random numbers or changing state. The tensors must be
        contiguous: under CUDA graphs they are copied into contiguous static
        buffers, and a different layout would pick different kernels.
        """
        ...

    def compute(
        self, linear: torch.nn.Linear, x: torch.Tensor, *inputs: torch.Tensor
    ) -> torch.Tensor:
        """The traced computation on the Linear's input ``x``."""
        ...

    def result(
        self, linears: list[torch.nn.Linear], output: torch.Tensor
    ) -> None:
        """Receives the output for the Linears that share it."""
        ...


# ---- 3. state / 4. update transform ----
@dataclasses.dataclass
class StepContext:
    """What an update transform sees in the step graph.

    Attributes:
        groups: The tap groups: the Linears sharing one tap call.
        tap_outputs: ``tap_outputs[g]`` is group g's tap outputs in every slot,
            stacked: shape (num_slots, ...), fixed address.
        landing: The slot whose micro-batch's gradient is being applied; every
            other slot is in flight (the 1F1B steady state). A Python int, so
            a step graph is captured per landing slot.
    """

    groups: list[list[torch.nn.Linear]]
    tap_outputs: list[torch.Tensor]
    landing: int


class UpdateTransform(Protocol):
    """A change to the base optimizer's update, captured in the step graph."""

    def init_state(self, ctx: StepContext) -> None:
        """Allocates persistent state (eagerly, once, before capture)."""
        ...

    def prepare(self, ctx: StepContext) -> None:
        """Runs before the base optimizer step (capturable)."""
        ...

    def apply(self, ctx: StepContext) -> None:
        """Runs after the base optimizer step (capturable)."""
        ...


@dataclasses.dataclass(frozen=True)
class Spec:
    """A staleness method, as primitives (see module docstring)."""

    backward_weights: WeightBinding = BindLive()
    forward_weights: Optional[BindExpr] = None
    linear_tap: Optional[LinearInputTap] = None
    update: Optional[UpdateTransform] = None
