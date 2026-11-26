# `lib/correctors/` — weight consistency for asynchronous pipelines

In an asynchronous pipeline a micro-batch's forward, its backward and the update that
consumes its gradient happen at different weight versions. A staleness method decides
three things, one mechanism per **slot**, around a shared **base optimizer** (a stock
`torch.optim.NAdam`, [`lib/optim/base_optimizer.py`](../optim/README.md)):

| Slot | What it decides | Mechanisms (configs.py) |
| --- | --- | --- |
| `forward` | the weights the forward reads | `current`, `predict` |
| `backward` | the weights the backward differentiates at | `current`, `stash`, `recover`, `predict` |
| `update` | how the base update is applied | `base`, `fresco` |

**Learning-rate damping** (PipeMare's T1, `dampers.py`, `--damper.staleness_lr_steps`) is
orthogonal to all three. The paper's methods are presets of these choices
(`configs.PRESETS`, `--method.name <name>`):

| Preset | forward | backward | update |
| --- | --- | --- | --- |
| `inconsistent` | current | current | base |
| `stashing` (PipeDream) | current | stash | base |
| `recovery` (PipeMare's T2) | current | recover | base |
| `pipeoptim_nostash` (PipeOptim) | predict (optimizer, τ) | current | base |
| `spectrain` | predict (optimizer, ⌊stage/2⌋+τ) | predict (optimizer, ⌊stage/2⌋) | base |
| `lwp` (Kosson) | predict (weight diff, T = τ) | current | base |
| `fresco` (this paper) | current | current | fresco |

The read slots are declarations: `Stash`, `Recover` and `Predict` each declare a
realization (`Snapshot`, `Backtrack`, `Extrapolate`) for their slot's compute phase, and the optimizer's
`ParameterVersionManager` (`lib/versions/parameters.py`) carries them out in both
engines.

Every slot mechanism also has a readable, eager version in [`references.py`](../../references.py),
independent of this package.

## Modules

| Module | Contents |
| --- | --- |
| `__init__.py` | `build(base_optimizer, modules, num_versions, forward=, backward=, update=, staleness_lr_steps=, **kwargs)` from the configs.py mechanisms (`CORRECTORS` maps each to its corrector), and `num_weight_versions()`. |
| `base.py` | `AsyncPipelineCorrector` (one mechanism: the weights it declares, `reads`, and its update change, as hooks), `AsyncPipelineOptimizer` (the base optimizer with one corrector per slot plus learning-rate damping, and the `ParameterVersionManager` serving their reads), `Current`, `Version`. |
| `prediction.py` | `Predict` — W + s·dW (`Extrapolate`); dW from the optimizer state (PipeOptim, SpecTrain) or the last weight difference (LWP). |
| `stashing.py` | `Stash` — the backward reads the weights kept at its forward (`Snapshot`). |
| `recovery.py` | `Recover` — PipeMare's T2: backtrack the weights along an EMA of updates (`Backtrack`). |
| `fresco.py` | `FrescoProjection` — this paper: project the update onto the in-flight null space, fast, in both engines (the projection split around the step; the extractor and Q's subspace iteration are `lib/linalg`). The algorithm alone, readable: [`references.py`](../../references.py). |
| `dampers.py` | `StalenessAwareLRDamper`. |

## How the slots run

`AsyncPipelineOptimizer` owns the per-step scaffolding (clip → base step → version
counter) and calls each `AsyncPipelineCorrector`'s hooks:

| Training-loop call | What the slots do |
| --- | --- |
| `before_forward()` / `after_forward()` | point the parameters at the forward's weights (`versions.weights(FORWARD)`) and back (exactly: `p.data` swaps); every realization records them (a `Snapshot` keeps a copy) |
| `load_old_params()` / `load_new_params()` | the same for the backward's weights (`versions.weights(BACKWARD)`) |
| `lr_scale(step)` | the learning-rate damper's factor on the scheduled learning rate (`lib/optim/lr_schedule.py` applies it) |
| `step()` | every slot's `before_step`, the base step, then `after_step` in order: update slot (FRESCO projects it) → the realizations (`Backtrack`'s EMA tracks the result) |
| `initialize_queue()` | allocate each slot's and realization's state |
| `state_dict()` / `load_state_dict()` | the base optimizer's state, every corrector's (recovery's EMA, LWP's W_{t−1}, FRESCO's bases, in-flight stashes) and the counters; loading checks the checkpoint is this method's, and still reads a checkpoint of the base optimizer's state alone |
| `param_groups` / `state` / ... | relayed to the base optimizer |

Under the graph engine a `GraphStage` reads the same slots as primitives (`spec`): the
forward / backward reads as weight bindings (`BindExpr` / `BindSnapshot` / `BindLive`,
the realizations' graph form), FRESCO as
the Linear-input tap, and the step-time work as one update transform in the step graph (so e.g. recovery's EMA also advances under a captured step).

## FRESCO — how it works

The algorithm alone, readable, is [`references.py`](../../references.py)
(`FrescoReference`, with the paper's pseudocode in its docstring); below is how
`FrescoProjection` computes the same, fast.


1. **Capture** — per micro-batch, each `nn.Linear` input gets a rank-`k1` spectral
   summary `V` (a `Basis`), through one path for both engines: the Linear-input tap
   protocol `inputs` → `compute` → `result`, traced into the forward graph by a
   GraphStage and driven by a forward pre-hook (`_capture`) in the eager engine.
   `compute` is subspace iteration on `[x, 1]ᵀ[x, 1]` (`basis_iteration`, a
   `lib.linalg.SubspaceIteration`: one Newton–Schulz block power step from a random
   block).
   `_plan_of` fixes each Linear's ranks once (`protection_budget`): `k1 = ⌈k2/τ⌉`,
   where `k2 = min(τ·n, ⌈α·min(d_out, d_in + 1)⌉)` is the protected rank (paper
   Sec. 3.2), so the `τ` in-flight bases hold `τ·k1 ≥ k2` directions. With bias
   augmentation the input gains a constant 1 coordinate, so `V` lives in
   `(d_in + 1)`-space. `result` queues `V` for the Linear (its in-flight summaries; the oldest
   retires when its micro-batch lands). Linears that read the same input with the same rank
   (q / k / v) share one `V`. Eagerly, `compute` is replayed
   as one CUDA graph per Linear (`ReplayedCall`: the same kernels, one launch).
2. **Combine** — `_eager_plan` (or `prepare` in the step graph): take the protection set (the in-flight batches' `V`, not the landing one),
   truncate the union to the top-`k2` directions (`Q = svd_lowrank(cat(V_i), k2)`; plain
   `qr` when the union is already ≤ `k2`), and cache `W_old @ Q`. Fixing `k2` keeps the
   protected subspace uniform across stages and below saturation. With bias
   augmentation the projected matrix is `M = [W, b]`.
3. **Project** — `after_step` (or `apply`), via `Projection` (`fresco.py`), after the base step: subtract the component of
   the weight change lying in `col(Q)`:
   `W_final = W_new − (W_new @ Q − W_old @ Q) @ Qᵀ`.

Projecting *after* the step (not the gradient before it) matters — Adam's `m/√v` does
not commute with a linear projection. FRESCO stores **no** weight snapshots.

## Arguments

Each is a field of [`configs.py`](../../configs.py), set as `--<section>.<field> VALUE`.

| Flag | Default | Description |
| --- | --- | --- |
| `--method.name` | required | one of `configs.PRESETS` |
| `--method.proj_rank_ratio` | 96/384 | FRESCO's protection budget α: `k2 = min(τ·n, ⌈α·min(d_out, d_in+1)⌉)` per layer |
| `--method.discrepancy_correction` | 0.02 | Parameter recovery's D (EMA decay γ = D^(1/τ)) |
| `--damper.staleness_lr_steps` | 0 | learning-rate damping's K (0: off), any method |
| `--optimizer.momentum` | 0.99 | base-optimizer β₁ |

## Notes

| Behaviour | Detail |
| --- | --- |
| Project after the step | the `col(Q)` component is removed from the weight *change*, not the gradient. |
