"""The graph engine: a rank's stage computed by GraphStage, moved by PipeDream.

PipeDream's runtime (lib.pipeline) keeps doing what it does well here --
process groups, send / recv threads, the data sampler -- and GraphEngine
replaces its eager compute: the stage's modules (and loss) are traced once
into a GraphStage. ``run_forward`` / ``run_backward`` mirror StageRuntime's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from ..graph.stage import GraphStage

if TYPE_CHECKING:
    from ..correctors.base import AsyncPipelineOptimizer
    from .runtime import StageRuntime


class _Chain(nn.Module):
    """A stage's modules applied in order (each feeds the next)."""

    def __init__(self, modules: Sequence[nn.Module]):
        super().__init__()
        self.body = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.body:
            x = m(x)
        return x


class GraphEngine:
    """Runs one rank's stage as a GraphStage over PipeDream's communication.

    Args:
        r: The rank's lib.pipeline.runtime.StageRuntime.
        method: The stage's staleness method.
    """

    def __init__(
        self,
        r: StageRuntime,
        method: AsyncPipelineOptimizer,
    ):
        self.r = r
        deps = r.modules_with_dependencies
        mods, ins, outs = (
            deps.modules(),
            deps.all_input_names(),
            deps.all_output_names(),
        )
        self.criterion: Optional[nn.Module] = (
            mods[-1] if r.is_criterion else None
        )
        body = mods[:-1] if r.is_criterion else mods
        for i in range(1, len(body)):  # a single chain of tensors
            assert ins[i] == outs[i - 1] and len(ins[i]) == 1, (ins, outs)
        self.in_name, self.out_name = ins[0][0], outs[len(body) - 1][0]
        dev = torch.device('cuda', torch.cuda.current_device())

        def example(name: str) -> torch.Tensor:
            x = torch.zeros(
                r.training_tensor_shapes[name],
                dtype=r.training_tensor_dtypes[name],
                device=dev,
            )
            return x.requires_grad_(x.is_floating_point())

        targets, loss_fn = (), None
        if self.criterion is not None:
            targets, loss_fn = (example('target'),), self._loss

        self.stage = GraphStage(
            _Chain(body),
            method,
            (example(self.in_name),),
            loss_fn=loss_fn,
            example_targets=targets,
            num_slots=r.num_warmup_minibatches + 1,
        )
        self.one = torch.ones((), device=dev)

    def _loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = target.numel()
        return self.criterion(out.view(n, -1), target.view(n))

    def run_forward(self) -> None:
        r = self.r
        r.receive_tensors_forward()
        t = r.tensors[-1]
        tg = (t['target'],) if self.criterion is not None else ()
        out = self.stage.forward(t[self.in_name].detach(), targets=tg)
        if self.criterion is not None:
            r.loss = out
        else:  # a copy: the send thread reads it after the next replay
            t[self.out_name] = out.clone()
        r.send_tensors_forward()
        r.forward_minibatch_id += 1

    def run_backward(self) -> None:
        r = self.r
        self.stage.prepare_step()  # overlaps the wait for the gradient
        r.receive_tensors_backward()
        grad = (
            self.one
            if self.criterion is not None
            else r.gradients[self.out_name]
        )
        g_in = self.stage.backward(grad)
        r.tensors.pop(0)
        if self.in_name not in ('input0', 'input1', 'input2'):
            r.gradients[self.in_name] = g_in.clone()
        r.send_tensors_backward()
        r.backward_minibatch_id += 1

    def step(self) -> None:
        self.stage.step()
