"""Learning-rate damping: scales the learning rate of a stale stage (PipeMare's
T1), for any slot choice.
"""

from .base import AsyncPipelineCorrector


class StalenessAwareLRDamper(AsyncPipelineCorrector):
    """PipeMare's T1 [Yang et al. 2021, Eq. 4]: a stage with delay tau steps at
    lr / tau ** p, p = 1 - min(step / K, 1), reaching the scheduled lr after K
    steps; a stage without staleness (tau <= 1) is not damped.

    Args:
        lr_steps: K.
    """

    def __init__(self, lr_steps: int):
        self.lr_steps = lr_steps

    def lr_scale(self, step: int) -> float:
        if self.tau <= 1:
            return 1.0
        return self.tau ** -(1.0 - min(step / self.lr_steps, 1.0))
