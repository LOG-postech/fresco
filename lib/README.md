# `lib/` — everything `main.py` uses

| Path | Contents |
| --- | --- |
| [`models/`](models/README.md) | The model (`gptn`), their stage layouts, `build.py`. |
| [`correctors/`](correctors/README.md) | The weight-consistency correctors (FRESCO; stashing, recovery and prediction baselines) composed around a base optimizer. |
| [`optim/`](optim/README.md) | The base optimizer (stock `torch.optim`) and the LR schedule. |
| `versions/` | `parameters.py`: the realizations (`Snapshot`, `Extrapolate`, `Backtrack`) and the `ParameterVersionManager` that serves each phase's weights in both engines; `bindings.py`: their graph form. |
| `linalg/` | `subspace.py`: randomized subspace iteration (`SubspaceIteration`), QR / Newton–Schulz orthogonalisation, `top_subspace`. Generic: FRESCO's extractor and combine use it. |
| [`pipeline/`](pipeline/README.md) | `runtime.py` (PipeDream's stage runtime: send / recv threads, eager compute) and `graph_engine.py` (the same stage computed by a `GraphStage`). |
| `graph/` | The graph stage: `primitives.py` (what a method is made of), `compile/` (tracing, FX passes), `cuda_graphs.py` (slot ring, CUDA graphs), `stage.py` (`GraphStage`). |
| `config.py`, `data.py`, `checkpoint.py`, `utils.py` | Command line (tyro), datasets and loaders, per-stage checkpoints, seeding / meters. |

`--runtime.engine graph` (default) computes stages with the graph stage below;
`--runtime.engine eager` with PyTorch autograd.

## The graph stage

In the eager engine, methods select forward/backward weights through hooks,
capture layer inputs when needed, and maintain their required state. Only methods
that need weight copies keep them.

On a graph stage the method instead declares a **`Spec`** built from four primitives, and
the stage compiles that spec into graphs it traces once:

| Primitive | Meaning | Used by |
| --- | --- | --- |
| **Weight binding** | Which tensor a pass reads for each weight (a realization's graph form, `lib/versions/bindings.py`): `BindLive` (current), `BindSnapshot` (the forward's, kept in the micro-batch's slot), or `BindExpr` (computed from the current weights, which stay untouched) | stashing: backward `BindSnapshot`; none, FRESCO: `BindLive`; Recovery: backward `BindExpr` (recovered); PipeOptim, LWP: forward `BindExpr` (predicted); SpecTrain: forward and backward `BindExpr` |
| **Tap** | `LinearInputTap`: a computation inserted at each `nn.Linear`'s input node in the forward graph. Linears reading one tensor share one call | FRESCO's input-feature basis |
| **State** | Tap outputs live in the slot; a method's persistent tensors are allocated once and updated in place | FRESCO's bases |
| **Update transform** | `prepare` / `apply` around the base optimizer step, inside the step graph | FRESCO's projection |

```python
import torch
import configs
from lib import correctors
from lib.graph.stage import GraphStage

opt = correctors.build(base_optimizer, [block], num_versions,
                       update=configs.Fresco(proj_rank_ratio=0.25))
stage = GraphStage(block, opt, (example_x,), loss_fn=loss_fn, num_slots=num_versions)
loss = stage.forward(x)              # micro-batch m runs in slot m % num_slots
stage.backward(torch.ones(()))       # the oldest in-flight micro-batch (1F1B)
stage.step()                         # clip + update transform + base step
```

**Tracing** (`compile/trace.py`). AOTAutograd traces the stage once, with parameters,
buffers, inputs and targets as graph inputs. Common-subexpression elimination runs first.
Then the joint graph is partitioned into a forward graph and a backward graph:

- **Forward** returns the stage output, the tensors saved for the backward, and the tap
  outputs.
- **Backward** takes the saved tensors and the output gradient.

The stage calls both directly, so the schedule decides when each backward runs.

**CUDA graphs** (`cuda_graphs.py`).

- **Graphs per slot:** each slot has a forward graph and a backward graph.
- **No copies of saved tensors:** a pass rewrites every op that produces a saved tensor
  to its `out=` variant, so the forward writes those tensors straight into the slot's
  buffers.
- **One shared pool:** stage outputs, tap outputs and gradients are copied into static
  buffers. So no graph output outlives its replay, all graphs share one memory pool (one
  set of temporaries), and replay order doesn't matter.
- **The step graph** captures clipping, the update transform and a capturable base
  optimizer (tensor LR) as one launch. The first step and the pipeline drain run
  eagerly.

**FRESCO on a graph stage.**

- **Basis:** extracted by the tap, in the forward graph.
- **Q:** the top-k2 subspace of the in-flight bases, from 2 subspace iterations with QR
  from a fresh random start (`svd_lowrank` can't be captured). Linears sharing bases
  share one Q.
- **Projection:** `M ← M − (M Q − M_old Q) Qᵀ`, applied without concatenating the bias
  column.

## Limits

- Static shapes.
- The step graph assumes one micro-batch per step.
