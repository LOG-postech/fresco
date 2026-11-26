"""Inputs of runtime.StageRuntime: tensor shapes and the stage maps."""

from __future__ import annotations

import json
from typing import Any, Optional

import torch
from torch import nn


def tensor_shapes(
    model: list[tuple[nn.Module, list[str], list[str]]],
    batch_size: int,
    eval_batch_size: int,
    block_size: int,
    n_embd: int,
    vocab_size: int,
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    dict[str, torch.dtype],
]:
    """Shapes and dtypes of every tensor passed between pipeline stages.

    Args:
        model: The list of (stage module, inputs, outputs) triples ending with
            the loss.
        batch_size: The training mini-batch size.
        eval_batch_size: The eval mini-batch size.
        block_size: The sequence length.
        n_embd: The embedding dimension (the hidden-state width).
        vocab_size: The vocabulary size (the logits width).

    Returns:
        A (training shapes, eval shapes, dtypes) tuple of {tensor name: ...}
        dicts.
    """
    input_size = [batch_size, block_size]
    training = {'input0': input_size, 'target': input_size}
    dtypes = {'input0': torch.int64, 'target': torch.int64}
    for _, _, outputs in model[:-1]:  # Skip the last layer (the loss).
        for output in outputs:
            width = vocab_size if output == 'output' else n_embd
            training[output] = input_size + [width]
            dtypes[output] = torch.float32
    evaluation = {
        key: tuple([eval_batch_size] + shape[1:])
        for key, shape in training.items()
    }
    training = {key: tuple(shape) for key, shape in training.items()}
    return training, evaluation, dtypes


def load_stage_maps(config_path: Optional[str], n_layer: int) -> dict[str, Any]:
    """Loads which module runs on which stage, and which stage on which rank.

    Args:
        config_path: The stage-layout JSON, or None for one layer per stage.
        n_layer: The number of transformer layers (used when config_path is
            None).

    Returns:
        The configuration_maps dict of runtime.StageRuntime:
        module_to_stage_map, stage_to_rank_map (keyed by int stage) and
        stage_to_depth_map.
    """
    if config_path is None:  # One layer per stage.
        return {
            'module_to_stage_map': list(range(n_layer)) + [n_layer - 1],
            'stage_to_rank_map': {i: [i] for i in range(n_layer)},
            'stage_to_depth_map': None,
        }
    with open(config_path, 'r', encoding='utf-8') as f:
        layout = json.load(f)
    return {
        'module_to_stage_map': layout.get('module_to_stage_map', None),
        'stage_to_rank_map': {
            int(k): v for k, v in layout.get('stage_to_rank_map', None).items()
        },
        'stage_to_depth_map': layout.get('stage_to_depth_map', None),
    }
