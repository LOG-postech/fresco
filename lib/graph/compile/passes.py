"""Graph passes over a traced stage: Linears, taps, aliases, slot writes.

The passes read the aten-level forward graph: an ``nn.Linear`` is an ``mm`` or
``addmm`` whose weight operand is a parameter placeholder (through ``t``); its input is the
operand's source before the views, i.e. the tensor the Linear was called with.
"""

import dataclasses
from typing import Any, Optional

import torch
import torch.fx
from torch import nn

from .. import primitives
from .trace import StageGraphs

aten = torch.ops.aten
LINEAR_OPS = (aten.mm.default, aten.addmm.default)
VIEW_OPS = (
    aten.view.default,
    aten._unsafe_view.default,
    aten.t.default,
    aten.permute.default,
    aten.transpose.int,
)


def _source(
    node: torch.fx.Node, through: tuple[torch._ops.OpOverload, ...]
) -> torch.fx.Node:
    """Follows ``node``'s first argument back through the ops in ``through``."""
    while node.op == 'call_function' and node.target in through:
        node = node.args[0]
    return node


def _placeholders(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [n for n in gm.graph.nodes if n.op == 'placeholder']


@dataclasses.dataclass
class TapGroup:
    """Linears that read one input tensor with one tap key.

    deferred: the forward outputs the tap's input (it lives in the slot)
    instead of computing the tap, and the stage computes it after the forward
    (insert_linear_taps).
    """

    linears: list[nn.Linear]
    x_shape: torch.Size
    num_inputs: int
    deferred: bool = False


def linear_sites(
    graphs: StageGraphs, module: nn.Module
) -> list[tuple[nn.Linear, torch.fx.Node]]:
    """Returns ``[(linear, input node)]``."""
    by_weight = {
        f'{name}.weight' if name else 'weight': m
        for name, m in module.named_modules()
        if isinstance(m, nn.Linear)
    }
    ph = _placeholders(graphs.fw)
    weight_of = {
        ph[i]: graphs.param_names[i] for i in range(len(graphs.param_names))
    }
    sites: list[tuple[nn.Linear, torch.fx.Node]] = []
    for node in graphs.fw.graph.nodes:
        if node.op != 'call_function' or node.target not in LINEAR_OPS:
            continue
        x, w = node.args[-2], node.args[-1]
        name = weight_of.get(_source(w, VIEW_OPS))
        if name in by_weight:
            sites.append((by_weight[name], _source(x, VIEW_OPS[:2])))
    return sites


def insert_linear_taps(
    graphs: StageGraphs,
    module: nn.Module,
    tap: primitives.LinearInputTap,
) -> list[TapGroup]:
    """Runs ``tap`` on every Linear's input inside the forward graph.

    Linears that read the same input node with the same ``tap.key`` share
    one call. Each call gets new forward inputs (``tap.inputs``) appended
    after the existing ones, and its output appended after the existing
    outputs, so the backward graph is unchanged.

    A tap whose input lives in the slot is not called: the forward outputs that input in its place (no copy), and the stage computes
    the tap after the forward, off the forward's critical path (the
    CudaGraphRunner's tap graphs). Its inputs stay forward inputs (unused by
    the graph), so the layout of ``tap.inputs`` is the same either way.

    Returns:
        The TapGroups, in the order of their inputs and outputs.
    """
    output = list(graphs.fw.graph.nodes)[-1]
    # Nodes whose values live in the slot (out= writes).
    saved_bases: set[torch.fx.Node] = set()
    for node in output.args[0][
        graphs.num_outputs : graphs.num_outputs + graphs.num_saved
    ]:
        while node.op == 'call_function' and (
            node.target in VIEW_OPS or node.target.__name__ == 'getitem'
        ):
            node = node.args[0]
        saved_bases.add(node)
    groups: dict[tuple[torch.fx.Node, Any], TapGroup] = {}
    nodes: dict[tuple[torch.fx.Node, Any], torch.fx.Node] = {}
    for linear, x in linear_sites(graphs, module):
        k = (x, tap.key(linear))
        if k not in groups:
            groups[k] = TapGroup([], x.meta['val'].shape, 0)
            nodes[k] = x
        groups[k].linears.append(linear)

    g = graphs.fw.graph
    last_ph = _placeholders(graphs.fw)[-1]
    output = list(g.nodes)[-1]
    results: list[torch.fx.Node] = []
    for i, (k, group) in enumerate(groups.items()):
        lin = group.linears[0]
        group.num_inputs = len(tap.inputs(lin, group.x_shape, dry_run=True))
        with g.inserting_after(last_ph):
            ins = [
                g.placeholder(f'tap{i}_in{j}') for j in range(group.num_inputs)
            ]
        last_ph = ins[-1] if ins else last_ph

        if nodes[k] in saved_bases and torch.cuda.is_available():
            group.deferred = True
            results.append(nodes[k])
            continue

        def call(
            x: torch.Tensor,
            *inputs: torch.Tensor,
            _lin: nn.Linear = lin,
        ) -> torch.Tensor:
            return tap.compute(_lin, x, *inputs)

        with g.inserting_before(output):
            results.append(g.call_function(call, (nodes[k], *ins)))
    output.args = (tuple(output.args[0]) + tuple(results),)
    g.lint()
    graphs.fw.recompile()
    return list(groups.values())


def _out_overload(
    target: torch._ops.OpOverload,
) -> Optional[torch._ops.OpOverload]:
    """The ``out=`` overload of an aten op, or None."""
    packet = target.overloadpacket
    for name in ('out', f'{target._overloadname}_out'):
        if name in packet.overloads():
            op = getattr(packet, name)
            if any(a.is_out for a in op._schema.arguments):
                return op
    return None


def copy_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """A copy of ``gm`` with its own graph (node targets are shared)."""
    graph = torch.fx.Graph()
    graph.output(graph.graph_copy(gm.graph, {}))
    return torch.fx.GraphModule(gm, graph)


def write_saved_into_slots(
    graphs: StageGraphs,
) -> tuple[
    torch.fx.GraphModule,
    list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype]],
]:
    """Makes the forward graph write its saved tensors into slot buffers.

    Every op that produces a saved tensor (directly or through views) is
    rewritten to its ``out=`` variant, writing into a new forward input -- the
    slot's buffer -- so the saved tensors live in the slot rather than in the
    graph's memory pool, and need no copy. Saved tensors that are parameters,
    buffers or stage inputs are left alone, and so are ops without an
    ``out=`` variant (their outputs must be copied).

    Returns:
        A copy of the forward graph with the slot buffers as extra inputs
        (after all others), and each buffer's (shape, stride, dtype).
    """
    fw = copy_graph(graphs.fw)
    g = fw.graph
    output = list(g.nodes)[-1]
    saved = output.args[0][
        graphs.num_outputs : graphs.num_outputs + graphs.num_saved
    ]
    through = VIEW_OPS + (aten.expand.default, aten.alias.default)
    producers: dict[torch.fx.Node, None] = {}
    for node in saved:
        while node.op == 'call_function' and (
            node.target in through or node.target.__name__ == 'getitem'
        ):
            node = node.args[0]
        if node.op == 'call_function' and _out_overload(node.target):
            producers[node] = None
    last_ph = _placeholders(fw)[-1]
    specs: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype]] = []
    for node in producers:
        op = _out_overload(node.target)
        vals = node.meta['val']
        vals = list(vals) if isinstance(vals, (tuple, list)) else [vals]
        bufs: list[torch.fx.Node] = []
        for v in vals:
            with g.inserting_after(last_ph):
                last_ph = g.placeholder(f'slot{len(specs)}')
            specs.append((tuple(v.shape), tuple(v.stride()), v.dtype))
            bufs.append(last_ph)
        accepted = {a.name for a in op._schema.arguments}
        kwargs = {k: v for k, v in node.kwargs.items() if k in accepted}
        out_names = [a.name for a in op._schema.arguments if a.is_out]
        kwargs.update(zip(out_names, bufs))
        with g.inserting_after(node):
            new = g.call_function(op, node.args, kwargs)
        new.meta = dict(node.meta)
        node.replace_all_uses_with(new)
        g.erase_node(node)
    g.lint()
    fw.recompile()
    return fw, specs
