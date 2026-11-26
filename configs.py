"""Experiment config of main.py: every setting, as dataclasses.

Every setting is a field of a section dataclass below. lib/config.py turns
every field into one command-line flag ``--<section>.<field>`` (with tyro); its
help text is the field's entry in the class docstring's ``Attributes:``
section. This file only declares the settings; parsing lives in lib/config.py.

Typical usage:

    python main.py --model ngpt_95m_32layer \\
        --method.name fresco --method.proj_rank_ratio 0.25 --lr.schedule.peak_value 3e-4

``--method.name`` picks one of the paper's methods (PRESETS). ``--help`` lists
every flag, grouped by section, with its default.
"""

# Config dataclasses hold one attribute per setting.
# pylint: disable=too-many-instance-attributes

import copy
import dataclasses
import typing
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# Model presets (selected by name with --model; shapes are not flags)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """A model architecture.

    Stage layouts for a model live in ``lib/models/<family>/layers=<n_layer>/``.

    Attributes:
        module: Package defining the per-stage model and tensor shapes.
        n_layer: Transformer layers.
        n_embd: Embedding dimension.
        n_head: Attention heads.
        block_size: Sequence length.
    """

    module: str
    n_layer: int
    n_embd: int
    n_head: int
    block_size: int = 512


# To train a new model, add it here and to ModelName.
MODEL_PRESETS = {
    'ngpt_95m_32layer': ModelConfig(
        'lib.models.gptn', n_layer=32, n_embd=384, n_head=6
    ),
}
ModelName = Literal['ngpt_95m_32layer']
assert set(typing.get_args(ModelName)) == set(
    MODEL_PRESETS
), 'ModelName and MODEL_PRESETS must list the same presets'


# ---------------------------------------------------------------------------
# Config sections (one --help group each)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataConfig:
    """Data.

    Attributes:
        dir: Dataset root directory.
        name: Dataset name, e.g. openwebtext.
        workers: Data-loading worker processes.
    """

    dir: str = '~/data'
    name: Optional[str] = None
    workers: int = 0


@dataclasses.dataclass
class PipelineConfig:
    """Pipeline / distributed launch (the launcher sets rank and port).

    Attributes:
        n_stages: Pipeline stages; selects the stage layout
            layers=<n_layer>/stage<n_stages>.json (None: n_layer).
        rank: Global rank of this process, i.e. its pipeline stage.
        local_rank: CUDA device index (0 when CUDA_VISIBLE_DEVICES pins one
            GPU).
        num_ranks_in_server: GPUs per machine.
        backend: Process-group backend.
        master_addr: Address of the rank-0 machine.
        master_port: Port of the rank-0 machine.
    """

    n_stages: Optional[int] = None
    rank: Optional[int] = None
    local_rank: int = 0
    num_ranks_in_server: int = 1
    backend: Optional[Literal['gloo', 'nccl']] = None
    master_addr: Optional[str] = None
    master_port: str = '12345'


@dataclasses.dataclass
class TrainConfig:
    """Training length, batch, and evaluation.

    Training is one continuous pipeline run of `steps` optimizer steps (the
    pipeline fills once and drains once, at the end); the held-out loss is
    measured once, after the final step, on the same `eval_steps` batches for
    every method.

    Attributes:
        steps: Optimizer steps to train for (one micro-batch each).
        batch_size: Training micro-batch size.
        eval_batch_size: Eval micro-batch size.
        eval_steps: Held-out batches in the final evaluation.
        seed: RNG seed.
        deterministic: Deterministic cuDNN, for reproducible runs.
    """

    steps: int = 20000
    batch_size: int = 8
    eval_batch_size: int = 8
    eval_steps: int = 25
    seed: int = 0
    deterministic: bool = False


@dataclasses.dataclass
class OptimizerConfig:
    """Base optimizer, shared by FRESCO and every baseline.

    Every method wraps the same base optimizer, so comparisons differ only in
    the staleness method (NAdamW).

    Attributes:
        momentum: Beta1.
        momentum_decay: NAdam momentum decay.
        weight_decay: Weight decay.
        clip_grad: Gradient-norm clip (None: off).
    """

    momentum: float = 0.99
    momentum_decay: float = 0.004
    weight_decay: float = 0.01
    clip_grad: Optional[float] = 1.0


@dataclasses.dataclass
class ScheduleConfig:
    """Learning-rate schedule, in optimizer steps.

    Linear warmup from warmup_init to peak_value over warmup_steps, then cosine
    decay to min_lr_ratio * peak_value at the final optimizer step.

    Attributes:
        peak_value: Peak learning rate.
        warmup_steps: Linear-warmup length (0: no warmup).
        warmup_init: Learning rate at the start of warmup.
        min_lr_ratio: Final learning rate as a fraction of peak_value.
    """

    peak_value: float = 3e-4
    warmup_steps: int = 3000
    warmup_init: float = 1e-7
    min_lr_ratio: float = 0.1


@dataclasses.dataclass
class LrConfig:
    """Learning rate.

    Attributes:
        schedule: Learning-rate schedule.
    """

    schedule: ScheduleConfig = dataclasses.field(default_factory=ScheduleConfig)


@dataclasses.dataclass
class DamperConfig:
    """Learning-rate damping, for any method (lib/correctors/dampers.py).

    Attributes:
        staleness_lr_steps: Staleness-aware LR damper (PipeMare's T1): a stage
            with delay tau starts at lr / tau and reaches lr after this many
            steps (lr / tau ** (1 - k/K)); 0 is off.
    """

    staleness_lr_steps: int = 0


@dataclasses.dataclass
class RuntimeConfig:
    """Compute engine.

    Attributes:
        engine: graph (traced CUDA graphs) or eager (PyTorch autograd).
    """

    engine: Literal['eager', 'graph'] = 'graph'


# ---------------------------------------------------------------------------
# Weight consistency: what a micro-batch's passes read and how the update is
# applied, as three slots with one mechanism each:
#
#   forward   the weights the forward reads            Current | Predict
#   backward  the weights the backward differentiates  Current | Stash |
#             at                                       Recover | Predict
#   update    how the base update is applied           Base | Fresco
#
# PRESETS names the paper's methods as slot choices.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Current:
    """The current weights (no correction in this slot)."""


@dataclasses.dataclass
class Predict:
    """Weight prediction: W + s * dW, the weights s steps ahead.

    s = horizon * tau (+ stage // 2 with offset half_stage, SpecTrain's).
    PipeOptim [Guan et al. 2023] and SpecTrain [Chen et al. 2018] take dW
    from the base optimizer's state; Kosson et al.'s LWP [2021] from the last
    weight difference.

    Attributes:
        rule: dW: optimizer (the base optimizer's update direction) or
            weight_diff (W_t - W_{t-1}; forward only).
        horizon: s as a multiple of the stage's delay tau.
        offset: Add stage // 2 to s (SpecTrain) or not.
    """

    rule: Literal['optimizer', 'weight_diff'] = 'optimizer'
    horizon: float = 1.0
    offset: Literal['none', 'half_stage'] = 'none'


@dataclasses.dataclass
class Stash:
    """Weight stashing [PipeDream]: the backward reads the forward's weights."""


@dataclasses.dataclass
class Recover:
    """Weight recovery (PipeMare's T2, discrepancy correction [Yang et al.
    2021]): the backward reads W - tau * EMA(update), approximating the
    weights the forward saw.

    Attributes:
        discrepancy_correction: T2's D; the update EMA decays at gamma = D ** (1
            / tau).
    """

    discrepancy_correction: float = 0.02


@dataclasses.dataclass
class Base:
    """The base optimizer's update as is."""


@dataclasses.dataclass
class Fresco:
    """FRESCO: project the stale update onto the in-flight null space.

    Protected rank per Linear: k2 = min(tau*n, ceil(alpha * min(d_out,
    d_in + 1))); each micro-batch extracts k1 = ceil(k2 / tau) directions.

    Attributes:
        proj_rank_ratio: alpha, the protection budget.
    """

    proj_rank_ratio: float = 0.25


# The paper's methods, as slot choices.
PRESETS = {
    'inconsistent': dict(forward=Current(), backward=Current(), update=Base()),
    'stashing': dict(forward=Current(), backward=Stash(), update=Base()),
    'recovery': dict(forward=Current(), backward=Recover(), update=Base()),
    'pipeoptim_nostash': dict(
        forward=Predict(), backward=Current(), update=Base()
    ),
    'spectrain': dict(
        forward=Predict(offset='half_stage'),
        backward=Predict(horizon=0.0, offset='half_stage'),
        update=Base(),
    ),
    'lwp': dict(
        forward=Predict(rule='weight_diff'), backward=Current(), update=Base()
    ),
    'fresco': dict(forward=Current(), backward=Current(), update=Fresco()),
}
MethodName = Literal[
    'inconsistent',
    'stashing',
    'recovery',
    'pipeoptim_nostash',
    'spectrain',
    'lwp',
    'fresco',
]
assert set(typing.get_args(MethodName)) == set(
    PRESETS
), 'MethodName and PRESETS must list the same methods'


@dataclasses.dataclass
class MethodConfig:
    """The method, and its hyperparameters.

    Attributes:
        name: One of the paper's methods.
        proj_rank_ratio: FRESCO: alpha, the protection budget.
        discrepancy_correction: Parameter recovery: D.
    """

    name: MethodName
    proj_rank_ratio: float = 0.25
    discrepancy_correction: float = 0.02


def method_slots(method: MethodConfig) -> dict[str, Any]:
    """The method's forward, backward and update mechanisms."""
    slots = copy.deepcopy(PRESETS[method.name])
    if isinstance(slots['update'], Fresco):
        slots['update'].proj_rank_ratio = method.proj_rank_ratio
    if isinstance(slots['backward'], Recover):
        slots['backward'].discrepancy_correction = method.discrepancy_correction
    return slots


@dataclasses.dataclass
class CheckpointConfig:
    """Checkpointing.

    Attributes:
        resume: Checkpoint path prefix to resume from (continues at its step).
        dir: Directory to save checkpoints in.
        dir_not_nfs: The checkpoint dir is not on shared NFS.
    """

    resume: str = ''
    dir: str = ''
    dir_not_nfs: bool = False


@dataclasses.dataclass
class LoggingConfig:
    """Logging.

    Attributes:
        exp_name: Experiment name (the wandb run name, when set).
        print_freq: Log the training loss every N steps.
        loss_ema: Decay of the per-step exponential moving average of the
            training loss that is logged next to the instantaneous loss.
        verbose_frequency: Verbose runtime logging every N steps (0: off).
        wandb: Log to wandb (from the last stage).
        wandb_project: Wandb project name.
    """

    exp_name: str = ''
    print_freq: int = 100
    loss_ema: float = 0.97
    verbose_frequency: int = 0
    wandb: bool = False
    wandb_project: Optional[str] = None


@dataclasses.dataclass
class Config:
    """FRESCO / async pipeline-parallel training.

    Attributes:
        model: Model preset (see MODEL_PRESETS in configs.py).
        method: The method and its hyperparameters.
        data: Data.
        pipeline: Pipeline / distributed launch.
        train: Training length and batch.
        optimizer: Base optimizer.
        lr: Learning rate.
        damper: Learning-rate damping, for any method.
        runtime: Compute engine.
        checkpoint: Checkpointing.
        logging: Logging.
    """

    model: ModelName
    method: MethodConfig
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    pipeline: PipelineConfig = dataclasses.field(default_factory=PipelineConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=OptimizerConfig
    )
    lr: LrConfig = dataclasses.field(default_factory=LrConfig)
    damper: DamperConfig = dataclasses.field(default_factory=DamperConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = dataclasses.field(
        default_factory=CheckpointConfig
    )
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)
