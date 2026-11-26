"""CUDA-graph capture and replay of a GraphStage over a ring of slots.

Micro-batch m belongs to slot m % num_slots, which holds what its backward
needs: the stage inputs, the saved tensors (the forward writes them straight
into the slot's buffers; compile.passes.write_saved_into_slots) and copies of
anything else saved (weight snapshots). A graph pair -- one forward graph, one
backward graph -- is captured per slot.

Everything else a graph produces (stage outputs, tap outputs, gradients) is
copied into static buffers outside the graphs' memory pool, so no graph output
outlives its replay, all graphs share one pool (one set of temporaries), and
replay order is free. The step graph captures gradient clipping, the
update transform and the base optimizer step as one launch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Optional, Union

import torch

from . import primitives
from .compile import passes

if TYPE_CHECKING:
    from .stage import GraphStage


def capture(
    body: Callable[[], None],
    pool: tuple[int, int],
    warmup: bool = True,
    warmups: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.cuda.CUDAGraph:
    """Records ``body`` as a CUDA graph (after ``warmups`` side-stream warm-up
    runs, which create cuBLAS handles and workspaces outside the capture).

    ``stream`` is the capture stream (default: torch's one, shared by every
    graph). cuBLAS / cuBLASLt workspaces are keyed by (handle, stream), and a
    graph bakes in its capture stream's: graphs captured on one stream share
    them, so a graph replayed concurrently with others needs its own.
    """
    if warmup:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmups):
                body()
        torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with (
        torch.no_grad(),
        torch.cuda.graph(
            graph, pool=pool, stream=stream, capture_error_mode='thread_local'
        ),
    ):
        body()
    return graph


class ReplayedCall:
    """``fn(*tensors) -> tensor`` replayed as a CUDA graph.

    Captured once per input signature (shapes, strides, dtypes), then replayed
    on copies of new inputs: the same kernels as calling ``fn``, one launch.
    The output is cloned out, so callers may share one memory pool (``pool``,
    created on first use) as long as they replay one at a time. ``fn`` must
    be capturable: no host syncs, and no random draws (pass them in).
    """

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        pool: Optional[tuple[int, int]] = None,
    ):
        self.fn, self.pool = fn, pool
        # signature -> (graph, static inputs, {'out': output})
        self._graphs: dict[
            tuple,
            tuple[
                torch.cuda.CUDAGraph,
                list[torch.Tensor],
                dict[str, torch.Tensor],
            ],
        ] = {}

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        sig = tuple((a.shape, a.stride(), a.dtype, a.device) for a in args)
        entry = self._graphs.get(sig)
        if entry is None:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            static = [a.clone() for a in args]
            out: dict[str, torch.Tensor] = {}

            def body() -> None:
                out['out'] = self.fn(*static)

            entry = (capture(body, self.pool, warmups=2), static, out)
            self._graphs[sig] = entry
        graph, static, out = entry
        for dst, src in zip(static, args):
            dst.copy_(src)
        graph.replay()
        return out['out'].clone()


def _view(
    storage: torch.UntypedStorage,
    dtype: torch.dtype,
    size: Sequence[int],
    stride: Sequence[int],
    offset: int,
    device: Union[torch.device, str],
) -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage, offset, size, stride
    )


class _Set:
    """One storage set: stage inputs, saved-tensor buffers, copy buffers."""

    def __init__(
        self,
        examples: Sequence[torch.Tensor],
        specs: Sequence[tuple[tuple[int, ...], tuple[int, ...], torch.dtype]],
        copy_nbytes: Sequence[int],
        device: Union[torch.device, str],
    ):
        def strided(
            shape: Sequence[int], stride: Sequence[int], dtype: torch.dtype
        ) -> torch.Tensor:
            return torch.empty_strided(shape, stride, dtype=dtype, device=device)

        # Zeroed: graph warm-ups run on them (zero is a valid token id).
        self.x = [
            strided(e.shape, e.stride(), e.dtype).zero_() for e in examples
        ]
        self.slot = [strided(sh, st, dt) for sh, st, dt in specs]
        self.copy = [
            torch.empty(nb, dtype=torch.uint8, device=device)
            for nb in copy_nbytes
        ]

    def tensors(self) -> list[torch.Tensor]:
        return [*self.x, *self.slot, *self.copy]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


class CudaGraphRunner:
    """The storage sets and the forward / backward / step graphs of a stage."""

    def __init__(
        self,
        stage: GraphStage,
        example_inputs: Sequence[torch.Tensor],
        example_targets: Sequence[torch.Tensor],
    ):
        self.stage = stage
        graphs, n = stage.graphs, stage.num_slots
        self.device = dev = stage.params[0].device
        self.fw, specs = passes.write_saved_into_slots(graphs)
        examples = [x.detach() for x in (*example_inputs, *example_targets)]
        # Expr bindings read static weight buffers, refilled before a replay:
        # the forward's (predicted weights) and the backward's (the saved
        # parameters' recovered / predicted values).
        self.fw_weights: Sequence[torch.Tensor] = (
            [torch.empty_like(p) for p in stage.params]
            if stage.fw_expr is not None
            else stage.params
        )
        self.bw_weights: Optional[list[torch.Tensor]] = (
            [torch.empty_like(p) for p in stage.params]
            if stage.bw_expr is not None
            else None
        )
        # Contiguous, like the tap inputs (kernels are chosen for the layout).
        self.tap_in_bufs = [
            torch.empty(t.shape, dtype=t.dtype, device=dev)
            for t in stage.tap_inputs(dry_run=True)
        ]

        # A warm-up forward / backward on a first set gives every shape, and
        # tells where each saved tensor lives.
        first = _Set(examples, specs, [], dev)
        for dst, src in zip(first.x, examples):
            dst.copy_(src)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            outs = self.fw(*self._fw_args(first))
            user, saved, taps = stage.split(outs)
            # A deferred tap's output is its input: compute it for the shape.
            taps = [
                stage.tap.compute(g.linears[0], t, *ins) if g.deferred else t
                for g, t, ins in zip(
                    stage.groups, taps, stage.tap_input_groups(self.tap_in_bufs)
                )
            ]
            self.tangent_bufs = [torch.ones_like(u) for u in user]
            grads = graphs.bw(*saved, *self.tangent_bufs)
        torch.cuda.current_stream().wait_stream(side)
        self._classify(saved, first)
        self.grad_bufs: list[Optional[torch.Tensor]] = [
            None if g is None else torch.empty_like(g) for g in grads
        ]
        del outs, saved, grads

        # Storage sets: one per slot.
        self.fw_sets = self.bw_sets = [
            _Set(examples, specs, self.copy_nbytes, dev) for _ in range(n)
        ]
        num_fw = len(self.fw_sets)
        self.out_bufs = [
            [torch.empty_like(u) for u in user] for _ in range(num_fw)
        ]
        # Tap outputs: one (num_slots, ...) tensor per tap group, so a step
        # treats every slot's outputs as one batch; each slot's graph writes
        # its own row.
        self.tap_store = [
            torch.empty((n, *t.shape), dtype=t.dtype, device=dev) for t in taps
        ]
        self.tap_out_bufs = [[t[s] for t in self.tap_store] for s in range(n)]
        del user, taps

        # Deferred taps: each set's forward graph outputs the inputs (static
        # views of the set's slot), and a tap graph per set computes them on
        # the tap stream after the forward (see forward).
        self.deferred = [i for i, g in enumerate(stage.groups) if g.deferred]
        self.tap_x: list[list[Optional[torch.Tensor]]] = [
            [None] * len(stage.groups) for _ in range(num_fw)
        ]

        self.pool = torch.cuda.graph_pool_handle()
        self.fw_graphs = [
            capture(self._fw_body(j), self.pool) for j in range(num_fw)
        ]
        self.bw_graphs = [
            capture(self._bw_body(k), self.pool)
            for k in range(len(self.bw_sets))
        ]
        self.step_graph: Optional[torch.cuda.CUDAGraph] = None
        self.tap_side: Optional[torch.cuda.Stream] = None
        if self.deferred:
            for j, s in enumerate(self.fw_sets):
                slot = {b.untyped_storage().data_ptr() for b in s.slot}
                for i in self.deferred:
                    ptr = self.tap_x[j][i].untyped_storage().data_ptr()
                    assert ptr in slot, 'a deferred tap input is not in the slot'
            self.tap_side = torch.cuda.Stream()
            self.tap_pool = torch.cuda.graph_pool_handle()
            self.tap_graphs = [
                capture(self._tap_body(j), self.tap_pool, stream=self.tap_side)
                for j in range(num_fw)
            ]

    def _classify(self, saved: Sequence[torch.Tensor], first: _Set) -> None:
        """Where each saved tensor lives: a set's buffers, or read in place."""
        st = self.stage
        owners: dict[int, tuple[str, Optional[int]]] = {}
        for j, b in enumerate(first.slot):
            owners[b.untyped_storage().data_ptr()] = ('slot', j)
        for j, x in enumerate(first.x):
            owners[x.untyped_storage().data_ptr()] = ('input', j)
        bound = 'expr' if st.bw_expr is not None else 'param'
        for j, p in enumerate(self.fw_weights):
            kind = 'copy' if st.snapshot else bound
            owners[p.untyped_storage().data_ptr()] = (kind, j)
        for j, b in enumerate(st.buffers):
            owners[b.untyped_storage().data_ptr()] = ('buffer', j)
        self.layout: list[
            tuple[str, int, torch.dtype, torch.Size, tuple[int, ...], int]
        ] = []
        copies: dict[int, tuple[int, int]] = {}
        for s in saved:
            ptr = s.untyped_storage().data_ptr()
            kind, j = owners.get(ptr, ('copy', None))
            if kind == 'copy':  # a snapshot parameter, or no out= variant
                if ptr not in copies:
                    copies[ptr] = (len(copies), s.untyped_storage().nbytes())
                j = copies[ptr][0]
            self.layout.append(
                (kind, j, s.dtype, s.size(), s.stride(), s.storage_offset())
            )
        self.copy_nbytes = [nb for _, nb in sorted(copies.values())]

    def slot_nbytes(self) -> int:
        """Bytes one slot keeps (stage inputs + saved tensors + copies)."""
        return self.fw_sets[0].nbytes()

    # ---- graph bodies ----
    def _fw_args(self, s: _Set) -> tuple[torch.Tensor, ...]:
        st = self.stage
        return (*self.fw_weights, *st.buffers, *s.x, *self.tap_in_bufs, *s.slot)

    def _saved_view(
        self,
        s: _Set,
        entry: tuple[str, int, torch.dtype, torch.Size, tuple[int, ...], int],
    ) -> torch.Tensor:
        kind, j, dtype, size, stride, offset = entry
        storage = {
            'slot': lambda: s.slot[j],
            'input': lambda: s.x[j],
            'param': lambda: self.stage.params[j],
            'expr': lambda: self.bw_weights[j],
            'buffer': lambda: self.stage.buffers[j],
            'copy': lambda: s.copy[j],
        }[kind]().untyped_storage()
        return _view(storage, dtype, size, stride, offset, self.device)

    def _fw_body(self, j: int) -> Callable[[], None]:
        s = self.fw_sets[j]

        def body() -> None:
            user, saved, taps = self.stage.split(self.fw(*self._fw_args(s)))
            done: set[int] = set()
            for t, (kind, i, *_) in zip(saved, self.layout):
                if kind == 'copy' and i not in done:  # whole storage, once
                    done.add(i)
                    src = _view(
                        t.untyped_storage(),
                        torch.uint8,
                        (self.copy_nbytes[i],),
                        (1,),
                        0,
                        self.device,
                    )
                    s.copy[i].copy_(src)
            for dst, u in zip(self.out_bufs[j], user):
                dst.copy_(u)
            for i, (dst, t) in enumerate(zip(self.tap_out_bufs[j], taps)):
                if self.stage.groups[i].deferred:
                    self.tap_x[j][i] = t  # the tap's input, in the slot
                else:
                    dst.copy_(t)

        return body

    def _tap_body(self, j: int) -> Callable[[], None]:
        st = self.stage
        ins = st.tap_input_groups(self.tap_in_bufs)

        def body() -> None:
            for i in self.deferred:
                lin = st.groups[i].linears[0]
                V = st.tap.compute(lin, self.tap_x[j][i], *ins[i])
                self.tap_out_bufs[j][i].copy_(V)

        return body

    def _bw_body(self, k: int) -> Callable[[], None]:
        saved = [self._saved_view(self.bw_sets[k], e) for e in self.layout]

        def body() -> None:
            grads = self.stage.graphs.bw(*saved, *self.tangent_bufs)
            for dst, g in zip(self.grad_bufs, grads):
                if dst is not None:
                    dst.copy_(g)

        return body

    # ---- forward / backward ----
    def forward(
        self,
        m: int,
        inputs: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        weights: Optional[Sequence[torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Runs micro-batch m's forward; returns (stage outputs, tap outputs).

        The stage outputs are static buffers: read them before the forward
        num_slots later.
        """
        compute = torch.cuda.current_stream()
        j = m % self.stage.num_slots
        # The previous taps read the slot and the tap inputs this forward
        # rewrites: wait for them (in the steady state the step already did,
        # through the prepare).
        self.join_taps()
        s = self.fw_sets[j]
        if self.fw_weights is not self.stage.params:
            for dst, w in zip(self.fw_weights, weights):
                dst.copy_(w)
        for dst, x in zip(s.x, (*inputs, *targets)):
            dst.copy_(x)
        for dst, t in zip(self.tap_in_bufs, self.stage.tap_inputs()):
            dst.copy_(t)
        self.fw_graphs[j].replay()
        if self.tap_side is not None:  # off the forward's critical path
            self.tap_side.wait_stream(compute)
            with torch.cuda.stream(self.tap_side):
                self.tap_graphs[j].replay()
        return self.out_bufs[j], self.tap_out_bufs[j]

    def backward(
        self,
        m: int,
        grad_outputs: Sequence[torch.Tensor],
        weights: Optional[Sequence[torch.Tensor]] = None,
    ) -> list[Optional[torch.Tensor]]:
        """Runs micro-batch m's backward."""
        k = m % self.stage.num_slots
        if self.bw_weights is not None:
            for dst, w in zip(self.bw_weights, weights):
                dst.copy_(w)
        for dst, g in zip(self.tangent_bufs, grad_outputs):
            dst.copy_(g)
        self.bw_graphs[k].replay()
        return self.grad_bufs

    # ---- step ----
    # The update transform's prepare (FRESCO: Q and its caches), then the step
    # graph (clip, base step, apply), both captured per landing slot.
    def step_context(self, landing: int) -> primitives.StepContext:
        st = self.stage
        return primitives.StepContext(
            groups=[g.linears for g in st.groups],
            tap_outputs=list(self.tap_store),
            landing=landing,
        )

    def _prep_body(self, landing: int) -> Callable[[], None]:
        update, ctx = self.stage.spec.update, self.step_context(landing)
        return lambda: update.prepare(ctx)

    def _step_body(self, landing: int) -> Callable[[], None]:
        method, update = self.stage.method, self.stage.spec.update
        ctx = self.step_context(landing)

        def body() -> None:
            method._scale_and_clip_grads()
            method.base_optimizer.step()
            if update is not None:
                update.apply(ctx)

        return body

    def prepare(self, landing: int) -> None:
        """The coming step's prepare, replayed as a CUDA graph."""
        if self.stage.spec.update is None:
            return
        if not hasattr(self, 'prep_graphs'):
            self.prep_graphs: dict[int, torch.cuda.CUDAGraph] = {}
            self.prep_pool = torch.cuda.graph_pool_handle()
        if landing not in self.prep_graphs:  # no warm-up: it would draw RNG
            self.prep_graphs[landing] = capture(
                self._prep_body(landing), self.prep_pool, warmup=False
            )
        self.join_taps()
        self.prep_graphs[landing].replay()

    def join_taps(self) -> None:
        """Makes the current stream wait for the deferred taps issued so far."""
        if self.tap_side is not None:
            torch.cuda.current_stream().wait_stream(self.tap_side)

    def step(self, landing: int) -> None:
        """The step after ``prepare``, as a CUDA graph."""
        if not hasattr(self, 'step_graphs'):
            self.step_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if landing not in self.step_graphs:
            # No warm-up run: it would apply a real step. The first step ran
            # eagerly, so the optimizer state and the handles exist.
            self.step_graphs[landing] = capture(
                self._step_body(landing), self.pool, warmup=False
            )
        self.step_graphs[landing].replay()
