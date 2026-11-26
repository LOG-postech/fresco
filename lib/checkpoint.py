"""Per-stage checkpoints: one file per pipeline stage."""

import logging
import os
from typing import Any, Optional

import torch


def load_checkpoint(prefix: str, stage: Optional[int]) -> dict[str, Any]:
    """Loads the checkpoint of one stage.

    Args:
        prefix: The checkpoint path prefix (--checkpoint.resume); the file is
            ``<prefix>.<stage>.pth.tar``.
        stage: The pipeline stage.

    Returns:
        The checkpoint dict (step, arch, eval_loss, state_dict, optimizer).
    """
    path = f'{prefix}.{stage}.pth.tar'
    assert os.path.isfile(path)
    logging.info("=> loading checkpoint '%s'", path)
    checkpoint = torch.load(path)
    logging.info(
        "=> loaded checkpoint '%s' (step %s, eval_loss %s)",
        path,
        checkpoint['step'],
        checkpoint.get('eval_loss'),
    )
    return checkpoint


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: str,
    stage: Optional[int],
) -> None:
    """Saves the checkpoint of one stage.

    Args:
        state: The checkpoint dict.
        checkpoint_dir: The directory (created if missing).
        stage: The pipeline stage.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint.{stage}.pth.tar')
    torch.save(state, path)
    logging.info('Saved checkpoint to %s', path)
