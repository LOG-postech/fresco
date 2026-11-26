# Advanced Usage

For installation and the basic training command, see the [README](../README.md).

## Running FRESCO

### Multi-GPU

Run the launcher once; it starts one worker per pipeline stage on a single machine.

```bash
uv run train.py \
  --method fresco \
  --stages 32 --gpus 8 \
  --lr 3e-4 --steps 20000 --batch-size 8 \
  --optimizer.momentum 0.99 \
  --method.proj_rank_ratio 96/384 \
  --train.seed 0
```

- Stage counts: `1, 2, 4, 8, 16, 32`; GPUs must divide the stage count.
- By default, stage `r` uses rank `r`; rank `r` runs on GPU `r % gpus`. No `torchrun` wrapper is needed.

### GPU Selection

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run train.py --stages 8 --gpus 2 --dry-run
```

- `--dry-run` previews worker commands and GPU assignments; omit it to train.

### Model-to-Rank Mapping

`--stages` selects the model's layout JSON. For example, `--model ngpt_95m_32layer --stages 4` loads [`lib/models/gptn/layers=32/stage4.json`](../lib/models/gptn/layers=32/stage4.json). Edit that file to change the partition or rank assignment.

| Field | Controls |
|---|---|
| `module_to_stage_map` | Pipeline stage for each model module, in model order |
| `stage_to_rank_map` | Worker rank assigned to each pipeline stage |

For example, this layout assigns 6, 10, 8, and 8 Transformer blocks to stages 0–3, and swaps the ranks of stages 1 and 2:

```json
{
  "module_to_stage_map": [
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3
  ],
  "stage_to_rank_map": {
    "0": [0], "1": [2], "2": [1], "3": [3]
  }
}
```

```bash
uv run train.py --method fresco --stages 4 --gpus 4
```

- The 32-layer model has 33 entries: one per block module, plus the criterion on the last stage. The first block module includes embeddings; the last includes the LM head.
- Keep stage IDs contiguous from `0` to `P-1` and block assignments in model order. The launcher supports one rank per stage; use every rank from `0` to `P-1` exactly once.
- Rank-to-GPU placement still follows `CUDA_VISIBLE_DEVICES[rank % gpus]`. With the example above and four GPUs, stage 1 runs on GPU 2 and stage 2 on GPU 1.
- `--dry-run` shows worker ranks and GPUs; it does not inspect a custom model partition.

## Custom Pipeline Integration

Wrap the base optimizer for each stage with FRESCO:

```python
from lib.correctors import AsyncPipelineOptimizer, FrescoProjection

optimizer = AsyncPipelineOptimizer(
    base_optimizer,
    modules=[stage],
    num_versions=num_warmup_microbatches + 1,
    update=FrescoProjection(proj_rank_ratio=0.25),
)
```

- `stage` is the local stage module; `base_optimizer` optimizes its parameters.
- `num_warmup_microbatches` is the stage's pipeline delay, supplied by the runtime. The scheduler must retain the forward/backward order of the in-flight micro-batches.
- FRESCO tracks trainable `nn.Linear` inputs and projects their weight and bias updates when the wrapped `optimizer.step()` runs. Other parameters receive the base optimizer's update.
- The launcher already handles this integration. For a custom scheduler, follow the training loop in [`main.py`](../main.py) and the hooks documented in [`lib/correctors/`](../lib/correctors/README.md).

## CLI Configuration

See [CLI Arguments](arguments.md) for every launcher option, alias, trainer setting, and default.

### CLI Overrides

Combine short options with full `--section.field value` flags:

```bash
uv run train.py --method fresco --stages 8 --gpus 4 \
  --lr 1e-3 --steps 2000 \
  --method.proj_rank_ratio 0.5 \
  --lr.schedule.warmup_steps 100 \
  --train.seed 42 \
  --checkpoint.dir checkpoints/fresco
```

- Overrides apply to every stage and take precedence over method defaults.
- Help: `uv run train.py --help` for the launcher; `uv run main.py --help` for all settings in [`configs.py`](../configs.py).

## P=32 Presets & Hyperparameter Search

- Best presets are provided only for `P=32`, through the method scripts in [`scripts/`](../scripts/).
- Use sweeps for other depths or to repeat the search at `P=32`.

```bash
# P=32 Recovery preset, with W&B logging
bash scripts/recovery.sh --wandb --logging.wandb_project fresco

# Create a sweep, then run the printed agent command
uv run wandb login
bash scripts/sweep.sh fresco
uv run wandb agent <ENTITY>/fresco/<SWEEP_ID>
```

| Search | Command | Parameters |
|---|---|---|
| FRESCO | `bash scripts/sweep.sh fresco` | P × LR × β₁ × α × LRD |
| Other baselines | `bash scripts/sweep.sh methods` | Method × P × LR × β₁ × LRD |
| Recovery | `bash scripts/sweep.sh recovery` | P × LR × β₁ × D × LRD |
| Synchronous reference | `bash scripts/sweep.sh sync` | P=1 × LR × β₁ |

- Search ranges are in [`scripts/sweep_*.yaml`](../scripts/); asynchronous searches cover `P=2,4,8,16,32`, and LRD searches `K=0` or `15000`.
- `methods` and `sync` select by training loss EMA (`0.97`), last logged at step `19,900` with the default 20k steps and logging interval of 100. `fresco` and `recovery` select by held-out `eval/loss` for α/D tuning.
- The P=32 presets keep LRD off. The [paper's wall-clock comparison](https://arxiv.org/html/2611.11238v1#S5.F5) uses α=`0.125`: pass `--method.proj_rank_ratio 0.125`.
- Shallow runs default to `min(8, P)` GPUs, differing from the paper's stated eight GPUs per run. Paper results have not been re-verified end-to-end with this code.

## Repository Layout

| Path | Contents |
|---|---|
| [`scripts/`](../scripts/) | P=32 method presets, sweep runner, and sweep YAML configs |
| [`train.py`](../train.py) | Single-machine launcher: starts and monitors every pipeline stage |
| [`scripts/train.sh`](../scripts/train.sh) | Shell entry point for the launcher, also used by sweeps |
| [`main.py`](../main.py) | Training entry point for one pipeline stage |
| [`configs.py`](../configs.py) | Every trainer setting and the model presets |
| [`lib/`](../lib/README.md) | Models, correctors, optimizer, pipeline runtime and graph engine |
| [`references.py`](../references.py) | Readable, eager reference implementations of each method |
