"""Weight prediction: a forward (or backward) slot reading W + s * dW.

One mechanism covers the published predictors (configs.PRESETS):

1. PipeOptim [Guan 2025 TKDE]: dW the base optimizer's update direction (for
   AdamW: ``-lr * m_hat / (sqrt(v_hat) + eps)``), forward s = tau (Eq. 4).
2. SpecTrain [Chen 2018 ICPP]: the same dW, forward s = stage//2 + tau and
   backward s = stage//2 (Eqs. 5-6). (Original SpecTrain is SGDM-only; we
   extend to AdamW following PipeOptim's generalization.)
3. Kosson LWP [Kosson 2021 MLSys]: dW = W_t - W_{t-1}, the last applied
   update, forward s = T = horizon * tau (LWP^w, Eq. 12).

Staleness: tau = num_versions - 1 = P - stage - 1 with one rank per stage,
the optimizer steps between a micro-batch's forward and the update that
consumes its gradient (PipeOptim's s = D - rank - 1). The published PipeOptim
code uses world_size - rank, one more; SpecTrain's rank//2 + world_size -
rank - 1 is stage//2 + tau.
"""

from typing import Literal

from ..versions import ComputePhase, Extrapolate
from .base import AsyncPipelineCorrector


class Predict(AsyncPipelineCorrector):
    """The slot's pass reads the predicted weights W + s * dW (an
    Extrapolate: lib/versions/parameters.py).

    Args:
        rule: 'optimizer': dW from the base optimizer's state; 'weight_diff':
            dW = W_t - W_{t-1} (forward only: each forward advances W_{t-1}).
        horizon: s as a multiple of tau.
        offset: 'half_stage' adds stage // 2 to s (SpecTrain).
    """

    def __init__(
        self,
        rule: Literal['optimizer', 'weight_diff'] = 'optimizer',
        horizon: float = 1.0,
        offset: Literal['none', 'half_stage'] = 'none',
    ):
        self.rule, self.horizon, self.offset = rule, horizon, offset
        self.realization = Extrapolate(rule, horizon, offset)

    @property
    def s(self) -> float:
        """The prediction distance, in optimizer steps."""
        return self.realization.s

    @property
    def reads(self) -> dict[ComputePhase, Extrapolate]:
        return {self.phase: self.realization}

    def state_dict(self) -> dict:
        return self.realization.state_dict()  # weight_diff's W_{t-1}

    def load_state_dict(self, state: dict) -> None:
        self.realization.load_state_dict(state)
