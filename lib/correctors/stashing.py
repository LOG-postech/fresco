# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Weight stashing [PipeDream]: the backward slot reads stashed weights."""

from ..versions import ComputePhase, Snapshot
from .base import AsyncPipelineCorrector


class Stash(AsyncPipelineCorrector):
    """The backward reads the weights its micro-batch's forward saw, kept at
    the forward (a Snapshot: lib/versions/parameters.py)."""

    def __init__(self):
        self.realization = Snapshot()

    @property
    def reads(self) -> dict[ComputePhase, Snapshot]:
        return {ComputePhase.BACKWARD: self.realization}

    def state_dict(self) -> dict:
        """The in-flight micro-batches' stashes (none after a drain)."""
        return self.realization.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.realization.load_state_dict(state)
