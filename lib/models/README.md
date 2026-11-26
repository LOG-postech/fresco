# `lib/models/` — API reference

One model family, selected through a `--model` preset. It exposes `arch()` and
`model(...)` (the pipeline module list). The runtime and FRESCO are model-agnostic
(they act on `nn.Linear` by `isinstance`).

## Configuration

- Presets: `ModelName` and `MODEL_PRESETS` in [`configs.py`](../../configs.py); select with `--model ngpt_95m_32layer`.
- Model code: `gptn/`; layouts: `gptn/layers=<N>/stage<K>.json`, selected by `--stages`.
- New model: add a preset and its stage layouts, then launch with `--model <preset>`.

## Modules

| Package | `arch()` | Family | Files |
| --- | --- | --- | --- |
| `lib.models.gptn` | `"gptn"` | NanoGPT (learned pos-emb) | `stage_first/mid/last.py`, `util.py`, `__init__.py` |

`StageFirst` (embed + blocks), `StageMid` (blocks), `StageLast`
(blocks + final norm + LM head); `util.py` holds the blocks. `model(...)` appends
the criterion as the last module.

## `model()` arguments

| Arg | Default | Notes |
| --- | --- | --- |
| `criterion, vocab_size, block_size` | — | Required positionals. |
| `n_layer` | 32 | Transformer layers; the layout determines pipeline stages. |
| `n_embd` | 384 | |
| `n_head` | 6 | |
| `dropout` | 0.0 | |

## Scales (paper)

| Scale | Module | `n_layer` | `n_embd` | `n_head` | Stage config |
| --- | --- | --- | --- | --- | --- |
| 95M | `gptn` | 32 | 384 | 6 | `gptn/layers=32/stage32.json` |

The preset uses `block_size=512`; `stage32.json` assigns one transformer layer per stage.

## Stage-split JSON

| Key | Meaning |
| --- | --- |
| `module_to_stage_map` | One entry per submodule → its stage. Length `n_layer + 1`; the trailing entry is the loss/criterion module (shares the last stage). |
| `stage_to_rank_map` | `{stage: [rank, ...]}`. |

Available: `gptn/layers=32/stage{1,2,4,8,16,32}.json` (95M P-sweep).

## Gotchas

| Behaviour | Detail |
| --- | --- |
| Q/K/V are three `nn.Linear` | Attention uses separate `c_q`, `c_k`, `c_v` (not a fused `c_attn`). FRESCO captures a subspace **per** `nn.Linear`; fusing changes the memory cost and subspace geometry. Do not re-fuse without re-sweeping. |
| Trailing criterion entry | `module_to_stage_map` is one longer than `n_layer` — keep the criterion entry when writing a new split. |
