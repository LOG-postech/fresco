"""Randomized subspace iteration: the numerics behind FRESCO's bases.

The top-k subspace of a matrix (or of a PSD operator A^T A given only its
action) by block power / subspace iteration, with the orthogonalisations it
needs (Householder QR, Newton-Schulz polar).

Everything is batched over leading dimensions where noted and free of host
syncs (CUDA-graph capturable). Random starts are drawn eagerly
(``SubspaceIteration.start``), so a captured iteration replays the same
numbers as the eager one.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Operator = Callable[[torch.Tensor], torch.Tensor]


def orthonormalize(Y: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis of Y's columns (fp32 Householder QR), batched."""
    return torch.linalg.qr(Y)[0]


def orthogonalize(M: torch.Tensor, ns_iters: int = 1) -> torch.Tensor:
    """M's columns orthonormalised by ``ns_iters`` Newton-Schulz polar
    iterations from M / ||M|| (matmuls only: capturable)."""
    H = M / (M.norm() + 1e-8)
    for _ in range(ns_iters):
        H = 1.5 * H - 0.5 * (H @ (H.T @ H))
    return H


def power_iteration(
    op: Operator,
    G: torch.Tensor,
    iters: int,
    ortho: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``iters`` block power iterations ``G <- ortho(op(G))``."""
    for _ in range(iters):
        G = ortho(op(G))
    return G


def top_subspace(C: torch.Tensor, k: int, iters: int = 2) -> torch.Tensor:
    """The top-k left singular subspace of C: (..., d, m) -> (..., d, k).

    A fresh random combination of C's columns, then ``iters`` subspace
    iterations (like svd_lowrank).
    """
    R = torch.randn(*C.shape[:-2], C.shape[-1], k, device=C.device)
    Q = orthonormalize(C @ R)
    for _ in range(iters):
        Q = orthonormalize(C @ (C.mT @ Q))
    return Q


class SubspaceIteration:
    """Block power iteration for the top-k subspace of a PSD operator
    ``op(G) = A^T A G``.

    ``start`` gives the random starting block (drawn eagerly); ``iterate`` runs
    the iterations (capturable).

    Args:
        iters: Power iterations per call.
        ns_iters: Newton-Schulz iterations per orthogonalisation.
    """

    def __init__(self, iters: int = 1, ns_iters: int = 1):
        self.iters, self.ns_iters = iters, ns_iters

    def start(self, d: int, k: int, device: torch.device) -> torch.Tensor:
        """A random (d, k) block."""
        G = torch.randn(d, k, device=device, dtype=torch.float32)
        return G / (G.norm() + 1e-8)

    def orthogonalize(self, M: torch.Tensor) -> torch.Tensor:
        return orthogonalize(M, self.ns_iters)

    def iterate(self, op: Operator, start: torch.Tensor) -> torch.Tensor:
        """The top-k estimate from ``start`` (capturable)."""
        return power_iteration(op, start, self.iters, self.orthogonalize)
