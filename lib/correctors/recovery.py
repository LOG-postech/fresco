"""Weight recovery: PipeMare's T2 [Yang et al. 2021], a backward slot."""

from ..versions import Backtrack, ComputePhase
from .base import AsyncPipelineCorrector


class Recover(AsyncPipelineCorrector):
    """The backward reads the weights backtracked along the update EMA,
    ``W - tau * delta`` (recovering the weights its forward saw; a Backtrack:
    lib/versions/parameters.py).

    Args:
        discrepancy_correction: T2's D; the EMA's gamma = D ** (1 / tau).
    """

    def __init__(self, discrepancy_correction: float = 0.02):
        self.discrepancy_correction = discrepancy_correction
        self.realization = Backtrack(discrepancy_correction)

    @property
    def reads(self) -> dict[ComputePhase, Backtrack]:
        return {ComputePhase.BACKWARD: self.realization}

    def state_dict(self) -> dict:
        return self.realization.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.realization.load_state_dict(state)
