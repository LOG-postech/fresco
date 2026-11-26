# CLI Arguments

Complete argument reference for [`train.py`](../train.py) and [`main.py`](../main.py), based on [`configs.py`](../configs.py). For examples, see [Advanced usage](advanced_usage.md); for method descriptions, see [Methods](methods.md).

## Syntax and Defaults

- Run `uv run train.py` once to launch all stages on one machine. Run `uv run train.py --help` for launcher help.
- The launcher forwards `--section.field VALUE` arguments to every worker. Field names use underscores, for example `--train.eval_batch_size 4`.
- `uv run main.py --help` lists trainer arguments. `main.py` starts one stage; its configuration defaults are listed below.
- Boolean flags take no value: `--train.deterministic` enables a setting and `--train.no_deterministic` disables it. Optional values accept the literal `None`, for example `--optimizer.clip_grad None`.
- Explicit CLI values override defaults. The common peak LR is `3e-4` and β₁ is `0.99`; baseline scripts may pass explicit overrides.
- Floating-point values accept decimals, scientific notation, or fractions: `0.25`, `2.5e-1`, and `96/384` are equivalent.

## Launcher Arguments

| Argument | Full alias | Default | Values / purpose |
|---|---|---|---|
| `-h`, `--help` | — | — | Show launcher help and exit |
| `--method` | `--method.name` | `fresco` | `fresco`, `stashing`, `recovery`, `pipeoptim_nostash`, `spectrain`, `lwp`, `inconsistent` |
| `--model` | — | `ngpt_95m_32layer` | Model preset |
| `--stages` | `--pipeline.n_stages` | `NSTAGES`, otherwise `32` | Stage count: `1`, `2`, `4`, `8`, `16`, `32` for the provided model |
| `--gpus` | `--pipeline.num_ranks_in_server` | `NGPUS`, otherwise `min(8, stages)` | Local GPUs; must divide stages, with `1 <= gpus <= stages` |
| `--lr` | `--lr.schedule.peak_value` | `3e-4` | Peak learning rate |
| `--steps` | `--train.steps` | `20000` | Positive number of optimizer steps |
| `--batch-size` | `--train.batch_size` | `8` | Positive training micro-batch size |
| `--engine` | `--runtime.engine` | `eager` for one stage; `graph` otherwise | `eager` or `graph` |
| `--wandb` | `--logging.wandb` | Off | Enable W&B logging |
| `--master-port` | `--pipeline.master_port` | A free local port | Rendezvous port, `1`–`65535` |
| `--log-dir` | — | `logs/<timestamp>` under the repository | Worker output directory, one `rank<N>.log` per stage |
| `--dry-run` | — | Off | Print worker commands and GPU assignments without training |

- The launcher sets `--data.name openwebtext`; override it with an explicit dotted flag.
- Other `--pipeline.*` flags are managed by the launcher and cannot be forwarded. Use the recognized aliases above to change stage count, GPU count, or port.
- `--log-dir` and `--dry-run` belong only to `train.py`; they are not trainer settings.

## Trainer Arguments

These are all leaf fields of `configs.Config`, accepted by `main.py`. Except for launcher-managed pipeline fields, they can also be passed through `train.py`.

### Model

| Argument | Default | Values / purpose |
|---|---|---|
| `--model` | Required | `ngpt_95m_32layer`; the launcher supplies this preset by default |

Architecture dimensions are defined in `MODEL_PRESETS`, not separate CLI flags. See the [model documentation](../lib/models/README.md) to add a preset or stage layout.

### Method

#### Common Options

| Argument | Default | Values / purpose |
|---|---|---|
| `--method.name` | Required | `fresco`, `stashing`, `recovery`, `pipeoptim_nostash`, `spectrain`, `lwp`, `inconsistent`; the launcher defaults to `fresco` |

- All methods share the base optimizer, LR schedule, and optional LR damping documented below.
- Forward, backward, and update mechanisms are selected by the preset; they are not separate CLI flags.
- `stashing`, `pipeoptim_nostash`, `spectrain`, `lwp`, and `inconsistent` have no additional method-specific CLI hyperparameters; use the shared optimizer and LR arguments to configure them.

#### FRESCO (`fresco`)

| Argument | Default | Values / purpose |
|---|---|---|
| `--method.proj_rank_ratio` | `96/384` | Protection budget α |

- Protected rank: `k = min(tau * n, ceil(alpha * min(d_out, d_in + 1)))`; each micro-batch contributes `ceil(k / tau)` directions.
- The paper's wall-clock comparison uses `48/384`: pass `--method.proj_rank_ratio 0.125`.
- Floating-point arguments accept fractions directly, e.g., `--method.proj_rank_ratio 96/384`.

#### Parameter Recovery (`recovery`)

| Argument | Default | Values / purpose |
|---|---|---|
| `--method.discrepancy_correction` | `0.02` | Coefficient D; update EMA decay is `D ** (1 / tau)` |

### Data

| Argument | Default | Values / purpose |
|---|---|---|
| `--data.dir` | `~/data` | Dataset root passed to the loader; the current OpenWebText loader uses the Hugging Face cache instead |
| `--data.name` | `None` | Dataset key; currently `openwebtext` is supported and supplied by the launcher |
| `--data.workers` | `0` | DataLoader worker processes |

Use `HF_HOME` to change the [dataset cache location](../README.md#dataset-cache).

### Pipeline

| Argument | Trainer default | Launcher value / purpose |
|---|---|---|
| `--pipeline.n_stages` | `None` | `--stages`; `None` selects the model's layer count when using `main.py` directly |
| `--pipeline.rank` | `None` | Global worker rank, assigned from `0` to `stages - 1` |
| `--pipeline.local_rank` | `0` | CUDA device index within the process; the launcher pins one visible GPU and uses `0` |
| `--pipeline.num_ranks_in_server` | `1` | Local communication rank-group size; the launcher supplies `--gpus` |
| `--pipeline.backend` | `None` | `gloo`, `nccl`, or `None`; the launcher supplies `gloo` |
| `--pipeline.master_addr` | `None` | Rank-0 host address; the launcher supplies `localhost` |
| `--pipeline.master_port` | `12345` | Rendezvous port as a string; the launcher chooses a free port unless specified |

`train.py` assigns GPU `rank % gpus` from the visible device list. Direct `main.py` workers need consistent distributed settings; the launcher currently supports a single node.

### Training and Evaluation

| Argument | Default | Values / purpose |
|---|---|---|
| `--train.steps` | `20000` | Optimizer steps; one micro-batch per step |
| `--train.batch_size` | `8` | Training micro-batch size |
| `--train.eval_batch_size` | `8` | Evaluation micro-batch size |
| `--train.eval_steps` | `25` | Held-out batches evaluated after training |
| `--train.seed` | `0` | Random seed |
| `--train.deterministic`, `--train.no_deterministic` | `False` | Enable / disable deterministic cuDNN settings |

### Base Optimizer

| Argument | Default | Values / purpose |
|---|---|---|
| `--optimizer.momentum` | `0.99` | NAdamW β₁ |
| `--optimizer.momentum_decay` | `0.004` | NAdam momentum decay |
| `--optimizer.weight_decay` | `0.01` | Decoupled weight decay |
| `--optimizer.clip_grad` | `1.0` | Gradient-norm clipping threshold; `None` disables clipping |

The base optimizer is NAdam with decoupled weight decay. β₂ is fixed at `0.999`; optimizer type and β₂ are not CLI arguments.

### Learning-Rate Schedule

| Argument | Default | Values / purpose |
|---|---|---|
| `--lr.schedule.peak_value` | `3e-4` | Peak learning rate |
| `--lr.schedule.warmup_steps` | `3000` | Linear-warmup steps; `0` disables warmup |
| `--lr.schedule.warmup_init` | `1e-7` | Initial warmup learning rate |
| `--lr.schedule.min_lr_ratio` | `0.1` | Final learning rate as a fraction of the peak, after cosine decay |

### Learning-Rate Damping

| Argument | Default | Values / purpose |
|---|---|---|
| `--damper.staleness_lr_steps` | `0` | PipeMare T1 duration K in optimizer steps; `0` disables damping. A stage with delay τ starts with scale `1 / tau` and reaches `1` after K steps |

### Runtime

| Argument | Default | Values / purpose |
|---|---|---|
| `--runtime.engine` | `graph` | `graph`: traced CUDA graphs; `eager`: PyTorch autograd. The launcher defaults to `eager` for one stage |

### Checkpointing

| Argument | Default | Values / purpose |
|---|---|---|
| `--checkpoint.resume` | `''` | Prefix of per-stage files `<prefix>.<stage>.pth.tar`; empty disables resume |
| `--checkpoint.dir` | `''` | Save final checkpoints as `<dir>/checkpoint.<stage>.pth.tar`; empty disables saving |
| `--checkpoint.dir_not_nfs`, `--checkpoint.no_dir_not_nfs` | `False` | Enable / disable saving on every stage replica for non-shared storage; otherwise only replica 0 saves |

### Logging

| Argument | Default | Values / purpose |
|---|---|---|
| `--logging.exp_name` | `''` | W&B run name; empty uses `stage<P>_<method>` |
| `--logging.print_freq` | `100` | Training-loss logging interval in steps |
| `--logging.loss_ema` | `0.97` | Decay of the logged training-loss EMA |
| `--logging.verbose_frequency` | `0` | Verbose runtime logging interval; `0` disables it |
| `--logging.wandb`, `--logging.no_wandb` | `False` | Enable / disable W&B logging from the last stage |
| `--logging.wandb_project` | `None` | W&B project name |

`main.py` also accepts `-h` / `--help` to show all trainer options and exit.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `NSTAGES` | `32` | Launcher stage count; explicit `--stages` takes precedence |
| `NGPUS` | `min(8, stages)` | Launcher GPU count; explicit `--gpus` takes precedence |
| `CUDA_VISIBLE_DEVICES` | First `gpus` devices | Visible GPU indices or identifiers; the launcher uses the first `gpus` entries |
| `HF_HOME` | Hugging Face default | Dataset and tokenizer cache location |

The launcher sets each worker's `CUDA_VISIBLE_DEVICES` to its assigned GPU. The communication code sets `MASTER_ADDR` and `MASTER_PORT` from the pipeline arguments.
