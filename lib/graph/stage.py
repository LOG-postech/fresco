"""A pipeline stage run as traced forward / backward graphs.

GraphStage compiles what the stage's staleness method declares (its
``primitives.Spec``) into the stage's graphs and runs them as CUDA graphs, one
micro-batch at a time (``cuda_graphs.CudaGraphRunner``):

    stage = GraphStage(block, method, (x,), loss_fn=..., num_slots=tau + 1)
    loss = stage.forward(x)          # micro-batch m -> slot m % num_slots
    stage.backward(torch.ones(()))   # oldest in-flight micro-batch (FIFO)
    stage.step()                     # clip + method's update + base step

With 1F1B and num_slots = (micro-batches in flight) + 1, a slot is reused only
after its micro-batch's backward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import nn

from . import primitives
from .compile import passes
from .cuda_graphs import CudaGraphRunner
from .compile import trace as trace_lib

if TYPE_CHECKING:
    from ..correctors.base import AsyncPipelineOptimizer


class GraphStage:
    """One pipeline stage, driven by forward() / backward() / step() calls.

    Args:
        module: The stage's module.
        method: The stage's staleness method (a AsyncPipelineOptimizer); its
            ``spec`` decides the graphs, and it is attached to the stage.
        example_inputs: Example stage inputs (fixed shapes).
        loss_fn: For the last stage, ``loss_fn(output, *targets)``.
        example_targets: Example targets for ``loss_fn``.
        num_slots: Ring size; at least the in-flight micro-batches + 1.
    """

    def __init__(
        self,
        module: nn.Module,
        method: AsyncPipelineOptimizer,
        example_inputs: Sequence[torch.Tensor],
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        example_targets: Sequence[torch.Tensor] = (),
        num_slots: int = 1,
    ):
        method.attach()
        self.method, self.spec = method, method.spec
        self.params: list[nn.Parameter] = [
            p for _, p in module.named_parameters()
        ]
        self.buffers: list[torch.Tensor] = [
            b for _, b in module.named_buffers()
        ]
        self.snapshot = isinstance(
            self.spec.backward_weights, primitives.BindSnapshot
        )
        self.graphs = trace_lib.trace(
            module, example_inputs, loss_fn, example_targets
        )
        self.tap = self.spec.linear_tap
        # Under CUDA graphs the taps whose input lives in the slot are
        # deferred: computed after the forward, on the runner's tap stream.
        self.groups: list[passes.TapGroup] = (
            passes.insert_linear_taps(self.graphs, module, self.tap)
            if self.tap is not None
            else []
        )
        # Expr bindings: weights computed from the current ones (predicted /
        # recovered), the current weights untouched.
        self.fw_expr = self.spec.forward_weights
        self.bw_expr: Optional[primitives.BindExpr] = (
            self.spec.backward_weights
            if isinstance(self.spec.backward_weights, primitives.BindExpr)
            else None
        )

        self.num_slots = num_slots
        self.num_forward = self.num_backward = 0
        # A capturable base optimizer: state on the device, a tensor LR.
        for group in method.param_groups:
            group['capturable'] = True
            group['lr'] = torch.tensor(
                float(group['lr']), device=self.params[0].device
            )
        self.runner = CudaGraphRunner(self, example_inputs, example_targets)
        self.num_steps = 0
        # The landing slot of a coming graph step.
        self._graph_step: Optional[int] = None
        self._state_ready = False  # the update transform's init_state ran
        self._prepared = False  # prepare_step ran for the coming backward

    # ---- the method's taps: inputs before a forward, results after ----
    def tap_inputs(self, dry_run: bool = False) -> list[torch.Tensor]:
        return [
            t
            for g in self.groups
            for t in self.tap.inputs(g.linears[0], g.x_shape, dry_run=dry_run)
        ]

    def tap_input_groups(
        self, flat: Sequence[torch.Tensor]
    ) -> list[Sequence[torch.Tensor]]:
        """``tap_inputs()`` split per tap group."""
        out, i = [], 0
        for g in self.groups:
            out.append(flat[i : i + g.num_inputs])
            i += g.num_inputs
        return out

    def _tap_results(self, outputs: Sequence[torch.Tensor]) -> None:
        for g, out in zip(self.groups, outputs):
            self.tap.result(g.linears, out)

    def split(
        self, outs: Sequence[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Splits forward outputs into (stage outputs, saved, tap outputs)."""
        n_out, n_saved = self.graphs.num_outputs, self.graphs.num_saved
        return (
            list(outs[:n_out]),
            list(outs[n_out : n_out + n_saved]),
            list(outs[n_out + n_saved :]),
        )

    # ---- forward / backward ----
    def forward(
        self, *inputs: torch.Tensor, targets: Sequence[torch.Tensor] = ()
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Runs the next micro-batch's forward; returns the stage output(s)."""
        assert (
            self.num_forward - self.num_backward < self.num_slots
        ), 'ring full'
        weights = self._weights(self.fw_expr)
        user, taps = self.runner.forward(
            self.num_forward, inputs, targets, weights
        )
        self._tap_results(taps)
        self.num_forward += 1
        return user[0] if len(user) == 1 else tuple(user)

    def prepare_step(self) -> None:
        """Starts the coming step's prepare, once per backward.

        It needs only the in-flight bases and the current weights, final once
        the newest forward has run. backward() calls it if the pipeline did
        not.
        """
        if self._prepared:
            return
        self._prepared = True
        self._graph_step = None
        if self._next_step_is_graph():
            landing = self.num_backward % self.num_slots
            if self.spec.update is not None and not self._state_ready:
                self.runner.join_taps()  # it reads the bases eagerly
                self.spec.update.init_state(self.runner.step_context(landing))
                self._state_ready = True
            self.runner.prepare(landing)
            self._graph_step = landing

    def backward(
        self, *grad_outputs: torch.Tensor
    ) -> Union[Optional[torch.Tensor], tuple[Optional[torch.Tensor], ...]]:
        """Runs the oldest in-flight micro-batch's backward.

        Sets the parameters' ``.grad`` and returns the gradients of the stage
        inputs (None where an input needs none).
        """
        self.prepare_step()
        self._prepared = False
        weights = self._weights(self.bw_expr)
        grads = self.runner.backward(
            self.num_backward, grad_outputs, weights=weights
        )
        self.num_backward += 1
        for p, g in zip(self.params, grads):
            p.grad = g
        first = len(self.params) + len(self.buffers)
        g_in = grads[first : first + self.graphs.num_inputs]
        return g_in[0] if len(g_in) == 1 else tuple(g_in)

    def _weights(
        self, expr: Optional[primitives.BindExpr]
    ) -> Sequence[torch.Tensor]:
        """The weights a pass reads: an Expr's tensors, else the current."""
        if expr is None:
            return self.params
        new = expr.fn()
        return [new.get(p, p) for p in self.params]

    # ---- step ----
    def _next_step_is_graph(self) -> bool:
        """Whether the step after the coming backward runs as a graph body.

        It needs every slot but the landing one in flight (the 1F1B steady
        state) and one eager step first (optimizer state); otherwise (the
        first step, the drain) the method's own step runs.
        """
        in_flight_after = self.num_forward - (self.num_backward + 1)
        return (
            self.num_steps >= 1
            and in_flight_after == self.num_slots - 1
        )

    def step(self) -> None:
        """Clips, applies the method's update transform and the base step."""
        if self._graph_step is None:
            self.runner.join_taps()  # the method reads the bases
            self.method.step()
        else:
            self.runner.step(self._graph_step)
            self.method.finish_step()
        self._graph_step = None
        self.num_steps += 1
