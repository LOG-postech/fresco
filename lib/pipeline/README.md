# `lib/pipeline/` — API reference

The asynchronous pipeline engine, inherited from
[PipeDream](https://github.com/msr-fiddle/pipedream). **Method-agnostic** — it moves
tensors between stages and runs forward/backward; staleness handling lives entirely in
[`../correctors/`](../correctors/README.md). One rank runs one pipeline stage.

## Modules

| Module | Contents |
| --- | --- |
| `runtime.py` | `StageRuntime` (this rank's stage + its forward/backward), `ModulesWithDependencies`. |
| `communication.py` | `CommunicationHandler` — point-to-point tensor transfer between stages. |
| `threadsafe_queue.py` | `Queue` — the send/recv helper-thread queue. |
| `threadsafe_counter.py` | `Counter` — helper-thread synchronisation. |

## `StageRuntime`

### Members the optimizer reads (the coupling surface)

| Member | Type | Meaning |
| --- | --- | --- |
| `modules()` | `list[nn.Module]` | The modules in this stage. |
| `master_parameters` | list | Parameters the optimizer updates. |
| `num_warmup_minibatches` | int | Pipeline depth ahead of this stage → `num_versions = num_warmup_minibatches + 1`. |

### Methods

| Method | Purpose |
| --- | --- |
| `initialize(model, ...)` | Build this stage's slice from the model + stage map. |
| `run_forward()` | One microbatch forward (recv → compute → send). |
| `run_backward()` | One microbatch backward (recv grad → backward → send grad). |
| `train(num_iterations)` / `eval(num_iterations)` | Steady-state 1F1B loop / forward-only eval. |
| `zero_grad()` | Zero this stage's grads. |
| `parameters()` / `modules()` / `state_dict()` / `load_state_dict(d)` | Standard accessors. |
| `receive_tensors_forward()` / `send_tensors_forward()` | Forward-activation P2P (first stage has no recv; last has no send). |
| `receive_tensors_backward()` / `send_tensors_backward()` | Backward-gradient P2P. |
| `get_adjusted_learning_rate(base_lr)` | Stage-local LR adjustment. |
| `wait()` | Join the communication helper threads. |

## `CommunicationHandler`

| Method | Purpose |
| --- | --- |
| `initialize(receive_ranks, send_ranks, ...)` | Wire this rank's neighbour channels. |
| `start_helper_threads(num_iterations, forward_only)` | Launch background send/recv threads (overlap transfer with compute). |
| `send(tensor_name, tensor, ...)` / `recv(tensor_name, ...)` | Enqueue a send / block for a recv. |
| `create_process_groups()` | Build the P2P process groups. |
| `set_tensor_shapes(shapes)` / `set_counter(counter)` | Configure buffers / synchronisation. |
| `wait()` | Join helper threads. |

## Notes

| Behaviour | Detail |
| --- | --- |
| Backend | The single-node launcher uses gloo and starts each rank as a background process pinned through `CUDA_VISIBLE_DEVICES`. |
| Stale gradient | The backward reuses the forward's graph, and `backward_weight_hooks()` (a `saved_tensors_hooks` pack/unpack pair) clones each saved leaf parameter and refreshes it from the current `param.data` with the version counter reset — that is how PyTorch ≥1.5 tensor-versioning is satisfied across the optimizer steps that land between a forward and its backward. |
| Staleness source | `num_versions = num_warmup_minibatches + 1` — the only value the runtime hands the optimizer. |
