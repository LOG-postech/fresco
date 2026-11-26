"""Reference implementations: every weight-consistency mechanism, written to be
read (the learning-rate damper, already a formula, has none).

Each is an AsyncPipelineCorrector (lib/correctors/base.py) implementing only the
eager hooks, in plain fp32, with the paper's formulas spelled out: no graph
bindings, no batching, no CUDA graphs. Any of them can
stand in for its optimized counterpart in lib/correctors (the AsyncPipelineOptimizer
runs it eagerly; a GraphStage refuses it with NotImplementedError).

lib/ never imports this package: the references are independent of the
optimized code, so a bug in one is not silently shared by the other.

    PredictReference            W + s * dW (PipeOptim, SpecTrain, LWP)
    StashReference              the backward reads the forward's weights (PipeDream)
    RecoveryReference           W - tau * EMA(update) (PipeMare's T2)
    FrescoReference             null-space projection (this paper)
"""

import math
from collections import deque

import torch
from torch import nn

from lib.correctors.base import AsyncPipelineCorrector


# ---------------------------------------------------------------------------
# Weight prediction
# ---------------------------------------------------------------------------
# Weight prediction, readable: a pass reads W + s * dW, the weights s steps
# ahead (PipeOptim, SpecTrain, Kosson et al.'s LWP).


class PredictReference(AsyncPipelineCorrector):
    """W + s * dW, with s = horizon * tau (+ stage // 2 for SpecTrain's offset).

    dW is either the base optimizer's next update direction (``optimizer``:
    -lr * m_hat / (sqrt(v_hat) + eps), PipeOptim / SpecTrain) or the
    last applied update W_t - W_{t-1} (``weight_diff``: LWP, forward only).

    Args:
        rule: 'optimizer' or 'weight_diff'.
        horizon: s as a multiple of tau.
        offset: 'half_stage' adds stage // 2 to s (SpecTrain); 'none'.
    """

    def __init__(
        self,
        rule: str = 'optimizer',
        horizon: float = 1.0,
        offset: str = 'none',
    ):
        self.rule, self.horizon, self.offset = rule, horizon, offset

    def initialize(self) -> None:
        self.s = self.horizon * self.tau + (
            self.stage // 2 if self.offset == 'half_stage' else 0
        )
        self.previous = {}  # weight_diff: the weights at the previous forward

    def weights(self) -> dict:
        if self.s == 0:
            return {}
        if self.rule == 'weight_diff':
            return self._weight_diff()
        return {
            p: p.data + self.s * dW
            for p, dW in self._optimizer_direction().items()
        }

    def _optimizer_direction(self) -> dict:
        """The update direction, from the optimizer's state (none before its first step)."""
        out = {}
        for group in self.base_optimizer.param_groups:
            lr, (beta1, beta2), eps = group['lr'], group['betas'], group['eps']
            for p in group['params']:
                state = self.base_optimizer.state.get(p)
                if not state:
                    continue
                t = float(state['step'])
                m_hat = state['exp_avg'] / (1 - beta1**t)
                v_hat = state['exp_avg_sq'] / (1 - beta2**t)
                out[p] = -lr * m_hat / (v_hat.sqrt() + eps)
        return out

    def _weight_diff(self) -> dict:
        out = {}
        for p in self.params():
            if p in self.previous:  # the first forward only records W
                out[p] = p.data + self.s * (p.data - self.previous[p])
            self.previous[p] = p.data.clone()
        return out


# ---------------------------------------------------------------------------
# Weight stashing
# ---------------------------------------------------------------------------
# Weight stashing [PipeDream], readable: the backward reads stashed weights.


class StashReference(AsyncPipelineCorrector):
    """At each forward, keep a copy of the weights; the micro-batch's backward
    reads that copy (backwards run in forward order).
    """

    def initialize(self) -> None:
        self.stash: deque[dict[torch.nn.Parameter, torch.Tensor]] = deque()

    def record(self, read: dict) -> None:
        if self.tau == 0:  # nothing in flight
            return
        copy = {}
        for p in self.params():
            copy[p] = read.get(p, p.data).clone()
        self.stash.append(copy)

    def weights(self) -> dict:
        return self.stash.popleft() if self.stash else {}


# ---------------------------------------------------------------------------
# Weight recovery
# ---------------------------------------------------------------------------
# Weight recovery (PipeMare's T2, discrepancy correction), readable.


class RecoveryReference(AsyncPipelineCorrector):
    """The backward reads W - tau * delta, where delta is an exponential
    moving average of the applied updates:

        delta <- gamma * delta + (1 - gamma) * (W_after_step - W_before_step)
        gamma  = D ** (1 / tau)

    Args:
        discrepancy_correction: D.
    """

    def __init__(self, discrepancy_correction: float = 0.1):
        self.D = discrepancy_correction

    def initialize(self) -> None:
        self.gamma = self.D ** (1.0 / self.tau) if self.tau > 0 else 0.0
        self.delta = {
            p: torch.zeros_like(p, dtype=torch.float32) for p in self.params()
        }

    def weights(self) -> dict:
        if self.tau == 0:
            return {}
        return {
            p: (p.data.float() - self.tau * self.delta[p]).to(p.dtype)
            for p in self.params()
        }

    def before_step(self) -> None:
        self.before = {p: p.data.clone() for p in self.params()}

    def after_step(self) -> None:
        if self.tau == 0:
            return
        for p in self.params():
            update = p.data.float() - self.before[p].float()
            self.delta[p] = (
                self.gamma * self.delta[p] + (1 - self.gamma) * update
            )


# ---------------------------------------------------------------------------
# FRESCO
# ---------------------------------------------------------------------------
# FRESCO (this paper), readable: project the stale update onto the null space
# of the in-flight micro-batches' inputs.
#
# For each Linear y = W x + b of a stage with delay tau, with M = [W, b] and the
# input rows X = [x, 1] (bias augmentation):
#
#     k2, k1 = protection_budget(d_out, d_in + 1, n)
#
#     at each micro-batch's forward:
#         V = Newton-Schulz(X^T X G)                    # its spectral summary: one
#                                                        # block power step on X^T X
#                                                        # from a random block G
#     at each step (micro-batch m lands; m+1 .. m+tau are in flight):
#         Q = top-k2 left singular vectors of [V_m+1 .. V_m+tau]
#         M_new = base_optimizer_step(M)
#         M = M_new - (M_new - M) Q Q^T
#
# so (M_final - M) X^T = 0 for every in-flight X in col(Q): the in-flight
# forwards' outputs, which their backwards differentiate, are unchanged.


class FrescoReference(AsyncPipelineCorrector):
    """FRESCO's canonical algorithm, eager and fp32.

    Args:
        proj_rank_ratio: alpha, the protection budget.
    """

    def __init__(self, proj_rank_ratio: float = 0.25):
        self.alpha = proj_rank_ratio

    # ---- the algorithm ----
    def protection_budget(self, d_out: int, d: int, n: int) -> tuple[int, int]:
        """k2 = min(tau * n, ceil(alpha * min(d_out, d))), k1 = ceil(k2 / tau)."""
        tau = max(1, self.tau)
        k2 = max(
            1, min(math.ceil(self.alpha * min(d_out, d) - 1e-9), tau * n, d)
        )
        k1 = max(1, min(-(-k2 // tau), n, d))
        return k2, k1

    def summarize(self, X: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """One block power step on X^T X from ``start``, orthonormalised by
        one Newton-Schulz step."""
        return self.newton_schulz(X.T @ (X @ start))

    def newton_schulz(self, G: torch.Tensor) -> torch.Tensor:
        """An orthonormal basis close to G's columns (matmuls only)."""
        H = G / (G.norm() + 1e-8)
        return 1.5 * H - 0.5 * H @ (H.T @ H)

    @staticmethod
    def protected_subspace(
        summaries: list[torch.Tensor], k2: int
    ) -> torch.Tensor:
        """Q: the top-k2 left singular vectors of the in-flight summaries."""
        U = torch.linalg.svd(torch.cat(summaries, dim=1), full_matrices=False)[
            0
        ]
        return U[:, :k2]

    @staticmethod
    def project(
        M_old: torch.Tensor, M_new: torch.Tensor, Q: torch.Tensor
    ) -> torch.Tensor:
        """The update with its component in col(Q) removed."""
        return M_new - (M_new - M_old) @ Q @ Q.T

    # ---- the corrector: run it in the pipeline ----
    def initialize(self) -> None:
        self.linears = [
            m
            for mod in self.modules
            for m in mod.modules()
            if isinstance(m, nn.Linear)
        ]
        self.summaries = {
            lin: deque() for lin in self.linears
        }  # in-flight, oldest first
        self.k2 = {}
        self.last = None  # (input, summary): Linears reading one input share it
        if (
            self.tau > 0
        ):  # nothing in flight on the last stage: nothing to protect
            for lin in self.linears:
                lin.register_forward_pre_hook(self._capture)

    def _capture(self, linear: nn.Linear, inputs: tuple) -> None:
        if not linear.training:
            return
        x = inputs[0]
        augment = linear.bias is not None
        n, d = x.numel() // linear.in_features, linear.in_features + augment
        k2, k1 = self.protection_budget(linear.out_features, d, n)
        self.k2[linear] = k2
        key = (augment, linear.out_features)
        if self.last is not None and self.last[0] is x and self.last[1] == key:
            V = self.last[2]  # q / k / v read the same x: the same summary
        else:
            X = x.detach().reshape(n, linear.in_features).float()
            if augment:
                X = torch.cat([X, X.new_ones(n, 1)], dim=1)
            start = torch.randn(d, k1, device=X.device)
            start = start / (start.norm() + 1e-8)
            V = self.summarize(X, start)
            self.last = (x, key, V)
        self.summaries[linear].append(V)

    def _M(self, linear: nn.Linear) -> torch.Tensor:
        """M = [W, b] under bias augmentation, else W."""
        if linear.bias is not None:
            return torch.cat(
                [linear.weight.data, linear.bias.data[:, None]], dim=1
            )
        return linear.weight.data

    def before_step(self) -> None:
        self.plan = {}
        for lin in self.linears:
            in_flight = list(self.summaries[lin])[
                1:
            ]  # [0] is the landing micro-batch
            if in_flight:
                Q = self.protected_subspace(in_flight, self.k2[lin])
                self.plan[lin] = (Q, self._M(lin).clone())

    def after_step(self) -> None:
        for lin, (Q, M_old) in self.plan.items():
            M = self.project(M_old, self._M(lin), Q)
            linear_W, linear_b = (
                M[:, : lin.in_features],
                M[:, lin.in_features :],
            )
            lin.weight.data.copy_(linear_W)
            if linear_b.shape[1]:
                lin.bias.data.copy_(linear_b[:, 0])
        for lin in self.linears:  # retire the landed micro-batch's summary
            if self.summaries[lin]:
                self.summaries[lin].popleft()
        self.last = None
