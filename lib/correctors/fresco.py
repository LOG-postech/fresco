"""FRESCO, implemented: the update-slot corrector, fast, in both engines.

The algorithm alone, readable, is references.py (FrescoReference); this
file computes the same, independently:

    the projection   split around the base step (``Projection``: cache M_old Q
                     before it, project after it), with the bias absorbed
                     without concatenating it
    the corrector    ``FrescoProjection``: each micro-batch's spectral summary as a
                     Linear-input tap (traced into the forward graph, or a forward
                     pre-hook replayed as a CUDA graph eagerly), each Linear's
                     ranks planned once, Q and the projection eagerly or batched
                     in the stage's step graph

Q is the top-k2 subspace of the in-flight bases: eagerly by randomized SVD (or
QR), in the step graph by batched subspace iteration (lib/linalg), always
orthonormalised by Householder QR.
"""

from __future__ import annotations

import gc
import math
from collections import deque
from collections.abc import Callable
from typing import Any, NamedTuple, Optional

import torch
from torch import nn

from .. import linalg
from ..graph import primitives
from ..graph.cuda_graphs import ReplayedCall
from ..versions import ComputePhase
from .base import AsyncPipelineCorrector, AsyncPipelineOptimizer


class Basis(NamedTuple):
    """One micro-batch's spectral summary of one Linear input.

    Attributes:
        V: Orthonormal directions of the input's rows, (d, k1).
    """

    V: torch.Tensor


# ---------------------------------------------------------------------------
# The projection, split around the base step
# ---------------------------------------------------------------------------


def times_q(
    W: torch.Tensor, b: Optional[torch.Tensor], Q: torch.Tensor
) -> torch.Tensor:
    """[W, b] @ Q (b may be None), without concatenating the bias column."""
    d = W.shape[1]
    out = W @ Q[:d]
    if b is not None:
        out = out + torch.outer(b, Q[d])
    return out


def subtract_projection(
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    D: torch.Tensor,
    Q: torch.Tensor,
) -> None:
    """[W, b] -= D @ Q^T, in place."""
    d = W.shape[1]
    W.sub_(D @ Q[:d].mT)
    if b is not None:
        b.sub_(D @ Q[d])


class Projection:
    """One Linear's projection around the base step."""

    def __init__(
        self, W: nn.Parameter, b: Optional[nn.Parameter], k2: int
    ) -> None:
        self.W, self.b = W, b
        self.cache = torch.empty(
            W.shape[0], k2, device=W.device, dtype=torch.float32
        )

    def before(self, Q: torch.Tensor) -> None:
        """Before the base step: cache M_old Q."""
        self.cache.copy_(times_q(self.W.data, _data(self.b), Q))

    def after(self, Q: torch.Tensor) -> None:
        """After the base step: remove the update's component in col(Q)."""
        W, b = self.W.data, _data(self.b)
        D = times_q(W, b, Q) - self.cache
        subtract_projection(W, b, D, Q)


def _data(p: Optional[nn.Parameter]) -> Optional[torch.Tensor]:
    return None if p is None else p.data


# ---------------------------------------------------------------------------
# The corrector
# ---------------------------------------------------------------------------


class _Plan(NamedTuple):
    """One Linear's extraction, fixed by its shape and the micro-batch size.

    Attributes:
        augment: Bias augmentation: inputs gain a constant 1 coordinate.
        d: The (augmented) input dimension.
        n: Input rows per micro-batch.
        k2: The protected rank (protection_budget).
        k1: Directions extracted per micro-batch.
    """

    augment: bool
    d: int
    n: int
    k2: int
    k1: int


class FrescoProjection(AsyncPipelineCorrector):
    """FRESCO: keep each update out of the in-flight inputs' subspace.

    Per ``nn.Linear``, with ``M = [W, b]`` and inputs ``[x, 1]``: each forward
    stores a rank-k1 basis V of its inputs; each step takes Q (rank k2), the
    top subspace of the in-flight micro-batches' V, and applies
    ``dM (I - Q Q^T)`` instead of the update dM. The in-flight forwards'
    outputs are then unchanged by the step, with no weight copies.

    Ranks (paper Sec. 3.2): ``k2 = min(tau * n, ceil(alpha * min(d_out,
    d_in + 1)))``, n the micro-batch's tokens; ``k1 = ceil(k2 / tau)``.

    Both engines extract V through the Linear-input tap (a GraphStage traces
    it into the forward; eagerly a pre-hook replays it as a CUDA graph).
    Linears reading one input share its V (q / k / v).

    Args:
        proj_rank_ratio: alpha, the protection budget.
    """

    COMBINE_ITERS = 2  # subspace iterations for Q

    def __init__(self, proj_rank_ratio: float = 0.25) -> None:
        self.proj_rank_ratio = proj_rank_ratio
        # weight param -> bias param (or None)
        self._bias_of: dict[nn.Parameter, Optional[nn.Parameter]] = {}
        self._protected_k2: dict[nn.Parameter, int] = {}  # weight param -> k2
        # The per-micro-batch extractor: one block power iteration on
        # [x, 1]^T [x, 1] from a random start, one Newton-Schulz step.
        self.basis_iteration = linalg.SubspaceIteration(iters=1, ns_iters=1)
        # Eager engine: each Linear's extraction replayed as a CUDA graph.
        self._replayed: dict[nn.Parameter, ReplayedCall] = {}
        self._graph_pool: Optional[tuple[int, int]] = None  # shared by them
        self._plans: dict[tuple[nn.Parameter, int], _Plan] = {}
        self._last_plan: dict[nn.Parameter, _Plan] = {}  # the latest forward's
        # The previous capture (input, its version, key, basis, the Linear's weight):
        # Linears that read
        # the same input with the same rank share its basis.
        self._last_capture: Optional[
            tuple[torch.Tensor, int, tuple[bool, int], Basis, nn.Parameter]
        ] = None
        # weight -> its in-flight summaries, oldest (the landing one) first
        self._feature_queue: dict[nn.Parameter, deque[Basis]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._plan: list[tuple[torch.Tensor, list[Projection]]] = []

    def bind(
        self,
        method: AsyncPipelineOptimizer,
        phase: Optional[ComputePhase] = None,
    ) -> None:
        super().bind(method, phase)
        if (
            self.tau > 0
        ):  # nothing in flight (the last stage): nothing to protect
            self._register_hooks()

    # ---- graph stage: the basis is extracted inside the forward graph ----
    # This object is both the Linear-input tap (the basis) and the update
    # transform (the projection); neither runs without in-flight micro-batches.
    @property
    def tap(self) -> Optional["FrescoProjection"]:
        return self if self.tau > 0 else None

    @property
    def transform(self) -> Optional["FrescoProjection"]:
        return self if self.tau > 0 else None

    def attach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def key(self, linear: nn.Linear) -> tuple[bool, int]:
        """Linears sharing an input share a basis if their rank key matches."""
        return (
            self._bias_of.get(linear.weight) is not None,
            linear.out_features,
        )

    def _plan_of(self, linear: nn.Linear, n: int) -> _Plan:
        """The Linear's extraction plan for ``n`` input rows (cached)."""
        plan = self._plans.get((linear.weight, n))
        if plan is None:
            augment = self._bias_of.get(linear.weight) is not None
            d = linear.in_features + augment  # homogeneous coordinates [x, 1]
            k2, k1 = self.protection_budget(linear.out_features, d, n)
            plan = self._plans[(linear.weight, n)] = _Plan(
                augment, d, n, k2, k1
            )
        self._last_plan[linear.weight] = plan
        return plan

    def inputs(
        self, linear: nn.Linear, x_shape: torch.Size, dry_run: bool = False
    ) -> list[torch.Tensor]:
        """The tap's inputs: the extractor's start (drawn eagerly, so both
        engines use the same random numbers)."""
        plan = self._plan_of(linear, math.prod(x_shape) // linear.in_features)
        if dry_run:
            return [
                linear.weight.new_empty(
                    (plan.d, plan.k1), dtype=torch.float32
                )
            ]
        return [
            self.basis_iteration.start(
                plan.d, plan.k1, linear.weight.device
            )
        ]

    def compute(
        self, linear: nn.Linear, x: torch.Tensor, *start: torch.Tensor
    ) -> torch.Tensor:
        """V of the Linear's input ``x`` (traced into the forward graph, or
        replayed eagerly)."""
        plan = self._plan_of(linear, x.numel() // linear.in_features)
        return self._basis(
            x.reshape(-1, linear.in_features), plan.augment, start[0]
        )

    def result(self, linears: list[nn.Linear], V: torch.Tensor) -> Basis:
        """Queues the basis for every Linear that shares it."""
        k2 = self._last_plan[linears[0].weight].k2
        entry = Basis(V)
        for linear in linears:
            self._protected_k2[linear.weight] = k2
            self._feature_queue.setdefault(linear.weight, deque()).append(entry)
        return entry

    # ---- update transform: Q and the projection, in the stage's step graph
    # Tap groups with the same per-micro-batch rank k1 and protected rank k2
    # form one block: their in-flight bases are zero-padded to one tensor, so
    # each subspace iteration is one batched matmul pair and one batched
    # orthonormalisation for the whole block (cuSOLVER calls cost ~0.15 ms
    # each, whatever the size). The step graph is captured per landing slot,
    # so the in-flight slots are a static slice (no masked work).

    def init_state(self, ctx: primitives.StepContext) -> None:
        """Allocates the blocks, Q and the projections (once, eagerly)."""
        n_slots = ctx.tap_outputs[0].shape[0]
        blocks = {}
        for g, linears in enumerate(ctx.groups):
            _, d, k1 = ctx.tap_outputs[g].shape
            k2 = self._protected_k2[linears[0].weight]
            blocks.setdefault((k1, k2), []).append((g, d))
        self._blocks: list[dict[str, Any]] = []
        for (k1, k2), members in blocks.items():
            dmax = max(d for _, d in members)
            dev = ctx.tap_outputs[members[0][0]].device
            self._blocks.append(
                dict(
                    members=members,
                    k2=k2,
                    C=torch.zeros(
                        len(members), dmax, (n_slots - 1) * k1, device=dev
                    ),
                    Q=torch.zeros(len(members), dmax, k2, device=dev),
                    projections=[
                        [
                            Projection(
                                lin.weight, self._bias_of.get(lin.weight), k2
                            )
                            for lin in ctx.groups[g]
                        ]
                        for g, _ in members
                    ],
                )
            )

    @staticmethod
    def _fill(blk: dict[str, Any], ctx: primitives.StepContext) -> None:
        """Writes the in-flight bases (all slots but the landing one) into C."""
        L = ctx.landing
        for i, (g, d) in enumerate(blk["members"]):
            V = ctx.tap_outputs[g]
            live = torch.cat([V[:L], V[L + 1 :]])  # (slots - 1, d, k1)
            blk["C"][i, :d].copy_(live.permute(1, 0, 2).reshape(d, -1))

    def prepare(self, ctx: primitives.StepContext) -> None:
        """Computes Q for every block; runs each projection's ``before``."""
        for blk in self._blocks:
            self._fill(blk, ctx)
            blk["Q"].copy_(
                linalg.top_subspace(blk["C"], blk["k2"], self.COMBINE_ITERS)
            )
            for i, ((_, d), projs) in enumerate(
                zip(blk["members"], blk["projections"])
            ):
                for proj in projs:
                    proj.before(blk["Q"][i, :d])

    def apply(self, ctx: primitives.StepContext) -> None:
        """Runs each projection's ``after`` (after the base step)."""
        for blk in self._blocks:
            for i, ((_, d), projs) in enumerate(
                zip(blk["members"], blk["projections"])
            ):
                for proj in projs:
                    proj.after(blk["Q"][i, :d])

    def finish_step(self) -> None:
        """Retires the landed micro-batch's bases."""
        self._retire()

    def _retire(self) -> None:
        """Retires the landing micro-batch's basis: it leaves the protection
        set."""
        for q in self._feature_queue.values():
            if q:
                q.popleft()

    def initialize(self) -> None:
        # FRESCO stores no weight snapshots — only the per-layer feature bases.
        self._feature_queue = {}
        self._last_capture = None
        gc.collect()

    def state_dict(self) -> dict:
        """The in-flight bases (none after a drain) and the protected ranks,
        keyed by the weight's parameter index."""
        index = {p: i for i, p in enumerate(self.params())}
        return {
            "feature_queue": {
                index[p]: [b.V for b in q]
                for p, q in self._feature_queue.items()
            },
            "protected_k2": {
                index[p]: k for p, k in self._protected_k2.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        params = self.params()
        self._feature_queue = {
            params[i]: deque(Basis(V.to(params[i].device)) for V in q)
            for i, q in state["feature_queue"].items()
        }
        self._protected_k2 = {
            params[i]: k for i, k in state["protected_k2"].items()
        }

    # ---- eager engine: the tap protocol, from a forward pre-hook ----
    def _register_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        linears = [
            sub
            for module in self.modules
            for _n, sub in module.named_modules()
            if isinstance(sub, nn.Linear)
        ]
        self._bias_of = {sub.weight: sub.bias for sub in linears}
        self._hooks = [
            sub.register_forward_pre_hook(self._make_hook(sub))
            for sub in linears
        ]

    def _make_hook(
        self, linear: nn.Linear
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def hook(module: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
            if linear.weight.requires_grad and module.training:
                self._capture(linear, inp[0])

        return hook

    def protection_budget(
        self, d_out: int, d_in: int, n: int
    ) -> tuple[int, int]:
        """Returns the protected rank k2 and the per-micro-batch rank k1.

        ``k2 = min(tau*n, ceil(alpha * min(d_out, d_in)))`` with ``d_in``
        already including the bias coordinate (paper Sec. 3.2);
        ``k1 = ceil(k2 / tau)``.
        """
        tau = max(1, self.num_versions - 1)
        k2 = math.ceil(self.proj_rank_ratio * min(d_out, d_in) - 1e-9)
        k2 = max(1, min(k2, tau * n, d_in))
        k1 = max(1, min(-(-k2 // tau), n, d_in))  # oversample: tau*k1 >= k2
        return k2, k1

    def _capture(self, linear: nn.Linear, x: torch.Tensor) -> None:
        """The tap protocol on one Linear's input ``x``: inputs, compute
        (replayed as a CUDA graph on a GPU), result."""
        if x.numel() == 0:
            return
        key = self.key(linear)
        last = self._last_capture
        if (
            last is not None
            and last[0] is x
            and last[1] == x._version
            and last[2] == key
        ):
            # Same input tensor and the same rank as the previous Linear (q /
            # k / v): the same basis.
            self._plan_of(linear, x.numel() // linear.in_features)
            self._protected_k2[linear.weight] = self._protected_k2[last[4]]
            self._feature_queue.setdefault(linear.weight, deque()).append(
                last[3]
            )
            return
        x2 = x.detach()
        start = self.inputs(linear, x2.shape)
        if x2.is_cuda:
            call = self._replayed.get(linear.weight)
            if call is None:
                if self._graph_pool is None:
                    self._graph_pool = torch.cuda.graph_pool_handle()
                call = self._replayed[linear.weight] = ReplayedCall(
                    lambda xs, ss, lin=linear: self.compute(lin, xs, ss),
                    self._graph_pool,
                )
            V = call(x2, *start)
        else:
            V = self.compute(linear, x2, *start)
        entry = self.result([linear], V)
        # Holding x keeps its storage alive, so `is` cannot match a new tensor
        # at a recycled address.
        self._last_capture = (x, x._version, key, entry, linear.weight)

    def _basis(
        self, x: torch.Tensor, augment: bool, start: torch.Tensor
    ) -> torch.Tensor:
        """V of one Linear's 2-D input ``x``, on the features [x, 1]. Shared
        by both engines; ``start`` is from ``basis_iteration.start``."""
        xf = _features(x, augment)
        return self.basis_iteration.iterate(  # on [x, 1]^T [x, 1]
            lambda M: xf.T @ (xf @ M), start
        )

    # ---- projection ----
    def _eager_plan(
        self,
    ) -> list[tuple[torch.Tensor, list[Projection]]]:
        """The eager step's (Q, projections) per group of Linears.

        Linears whose in-flight bases are the same objects (q / k / v) share
        one Q. Q is the top-k2 subspace of the protection set (the in-flight
        bases; the landing micro-batch is not in it): svd_lowrank, or plain QR
        when they hold at most k2 directions.
        """
        groups = {}
        for module in self.modules:
            for sub in module.modules():
                if not isinstance(sub, nn.Linear):
                    continue
                q = self._feature_queue.get(sub.weight)
                if not q or len(q) <= 1:
                    continue
                inflight = list(q)[1:]  # q[0] is the landing micro-batch
                k2 = self._protected_k2[sub.weight]
                key = (k2, *map(id, inflight))  # q / k / v: the same objects
                groups.setdefault(key, (inflight, k2, []))[2].append(sub)
        plan = []
        for inflight, k2, linears in groups.values():
            cat = torch.cat([b.V for b in inflight], dim=1)
            if k2 >= cat.shape[1]:
                Q = torch.linalg.qr(cat)[0]
            else:
                Q = torch.svd_lowrank(cat, q=k2)[0]
            plan.append((Q, linears))
        return [
            (
                Q,
                [
                    Projection(
                        lin.weight, self._bias_of.get(lin.weight), Q.shape[1]
                    )
                    for lin in linears
                ],
            )
            for Q, linears in plan
        ]

    def before_step(self) -> None:
        self._plan = self._eager_plan() if self.num_versions > 1 else []
        for Q, projs in self._plan:
            for proj in projs:
                proj.before(Q)

    def after_step(self) -> None:
        for Q, projs in self._plan:
            for proj in projs:
                proj.after(Q)
        self._plan = []
        self._retire()  # the landed batch
        self._last_capture = None


def _features(x: torch.Tensor, augment: bool) -> torch.Tensor:
    """``x`` as features, with a constant 1 column if augmented."""
    if augment:
        x = torch.cat([x, x.new_ones(x.shape[0], 1)], dim=1)
    return x
