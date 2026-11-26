# `lib/optim/` — the base optimizer and its learning-rate schedule

The base optimizer computes each update; the asynchronous-pipeline staleness method
around it lives in [`../correctors/`](../correctors/README.md). Every staleness method wraps
the same base optimizer, so comparisons differ only in the method.

| Module | Contents |
| --- | --- |
| `base_optimizer.py` | `build_base_optimizer(cfg.optimizer, lr, params)` → `torch.optim.NAdam` (decoupled weight decay, `momentum_decay`); `None` for a stage with no parameters. |
| `lr_schedule.py` | Per-stage learning rate in optimizer steps: linear warmup, cosine to `min_lr_ratio` at the final step, times the staleness method's `lr_scale` (learning-rate damping, PipeMare's T1: `--damper.staleness_lr_steps`). Takes the schedule settings and the stage's peak LR, not the runtime. |

| Flag | Default | Description |
| --- | --- | --- |
| `--optimizer.momentum` | 0.99 | β₁ (β₂ = 0.999) |
| `--optimizer.momentum_decay` | 0.004 | NAdam momentum decay |
| `--optimizer.weight_decay` | 0.01 | decoupled weight decay |
| `--optimizer.clip_grad` | 1.0 | gradient-norm clip (applied by the staleness method's step) |
| `--lr.schedule.*` | — | see `lr_schedule.py` and `--help` |
