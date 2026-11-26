"""Builds the per-stage model of a model package (models.gptn)."""

import importlib
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    import configs


def build_model(
    spec: 'configs.ModelConfig', vocab_size: int
) -> tuple[list[tuple[nn.Module, list[str], list[str]]], str]:
    """Builds the model split into pipeline stages.

    Args:
        spec: The model preset (a configs.ModelConfig).
        vocab_size: The vocabulary size of the dataset.

    Returns:
        A (model, arch) pair: the list of (stage module, inputs, outputs)
        triples ending with the loss, and the package's architecture name.
    """
    module = importlib.import_module(spec.module)
    arch = module.arch()
    model = module.model(
        nn.CrossEntropyLoss(),
        vocab_size=vocab_size,
        block_size=spec.block_size,
        n_embd=spec.n_embd,
        n_head=spec.n_head,
        n_layer=spec.n_layer,
    )
    return model, arch


def count_parameters(
    model: list[tuple[nn.Module, list[str], list[str]]],
) -> float:
    """Returns the number of parameters of the stage modules, in millions."""
    return (
        float(
            sum(
                sum(p.numel() for p in s.parameters()) for s, _, _ in model[:-1]
            )
        )
        / 1e6
    )
