"""Traces a pipeline stage into separate forward and backward graphs.

AOTAutograd traces the stage once and partitions it into a forward graph, whose
outputs are the stage outputs followed by the tensors saved for the backward,
and a backward graph, whose inputs are those saved tensors followed by the
output gradients. The stage calls the two directly, so the pipeline schedule
(not autograd) decides when a micro-batch's backward runs. Common-subexpression
elimination runs first.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
from functorch.compile import aot_function, default_partition, make_boxed_func
from torch import nn
from torch._functorch.compile_utils import fx_graph_cse
from torch.func import functional_call


@dataclasses.dataclass
class StageGraphs:
    """A stage's forward and backward graphs and their calling convention.

    Forward inputs: params, buffers, stage inputs, targets (then any inputs a
    pass appends). Forward outputs: ``num_outputs`` stage outputs, then
    ``num_saved`` saved tensors (then any outputs a pass appends). Backward
    inputs: the saved tensors, then one gradient per stage output. Backward
    outputs: one gradient (or None) per forward input, in order.
    """

    fw: torch.fx.GraphModule
    bw: torch.fx.GraphModule
    param_names: list[str]
    num_inputs: int
    num_outputs: int
    num_saved: int


def _partition(
    joint: torch.fx.GraphModule,
    joint_inputs: Any,
    *,
    num_fwd_outputs: int,
    **_: Any,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    """Common-subexpression elimination, then the default partition."""
    joint.graph = fx_graph_cse(joint.graph)
    joint.recompile()
    return default_partition(
        joint, joint_inputs, num_fwd_outputs=num_fwd_outputs
    )


def trace(
    module: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    example_targets: Sequence[torch.Tensor] = (),
) -> StageGraphs:
    """Traces ``module`` (and ``loss_fn``) into forward and backward graphs.

    Args:
        module: The stage's module.
        example_inputs: Example stage inputs (their shapes are fixed); an input
            that requires grad gets a gradient from the backward.
        loss_fn: For the last stage, ``loss_fn(output, *targets)``; its scalar
            replaces the module output.
        example_targets: Example targets for ``loss_fn``.

    Returns:
        The StageGraphs. Tracing runs one forward and backward on the example
        inputs; the parameters' ``.grad`` is reset afterwards.
    """
    names, params = zip(*module.named_parameters())
    buffer_items = list(module.named_buffers())
    bnames = [n for n, _ in buffer_items]
    buffers = [b for _, b in buffer_items]
    n_p, n_b, n_in = len(params), len(buffers), len(example_inputs)

    def stage_fn(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = dict(zip(names, args[:n_p]))
        state.update(zip(bnames, args[n_p : n_p + n_b]))
        inputs = args[n_p + n_b : n_p + n_b + n_in]
        targets = args[n_p + n_b + n_in :]
        out = functional_call(module, state, inputs)
        if loss_fn is not None:
            out = loss_fn(out, *targets)
        return out if isinstance(out, tuple) else (out,)

    graphs: dict[str, torch.fx.GraphModule] = {}

    def keep(
        name: str,
    ) -> Callable[
        [torch.fx.GraphModule, Sequence[torch.Tensor]], Callable[..., Any]
    ]:
        def compiler(
            gm: torch.fx.GraphModule, _: Sequence[torch.Tensor]
        ) -> Callable[..., Any]:
            graphs[name] = gm
            return make_boxed_func(gm.forward)

        return compiler

    traced = aot_function(
        stage_fn,
        fw_compiler=keep('fw'),
        bw_compiler=keep('bw'),
        partition_fn=_partition,
    )
    outs = traced(*params, *buffers, *example_inputs, *example_targets)
    torch.autograd.backward(
        [o for o in outs if o.requires_grad],
        [torch.ones_like(o) for o in outs if o.requires_grad],
    )
    for p in params:
        p.grad = None
    for x in example_inputs:
        x.grad = None

    fw, bw = graphs['fw'], graphs['bw']
    num_fw_out = len(list(fw.graph.nodes)[-1].args[0])
    num_bw_in = sum(n.op == 'placeholder' for n in bw.graph.nodes)
    num_saved = num_fw_out - len(outs)
    assert num_bw_in == num_saved + len(outs), (num_bw_in, num_saved)
    return StageGraphs(
        fw=fw,
        bw=bw,
        param_names=list(names),
        num_inputs=n_in,
        num_outputs=len(outs),
        num_saved=num_saved,
    )
