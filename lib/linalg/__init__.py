"""Numerical linear algebra shared across the library (subspace.py)."""

from .subspace import (
    SubspaceIteration,
    orthogonalize,
    orthonormalize,
    power_iteration,
    top_subspace,
)

__all__ = [
    'SubspaceIteration',
    'orthogonalize',
    'orthonormalize',
    'power_iteration',
    'top_subspace',
]
