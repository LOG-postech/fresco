# Methods

For setup and the basic training command, see the [README](../README.md).

Select a method with `--method <keyword>`.

## No corrections

| Method | Description |
|---|---|
| Naive asynchronous training (`inconsistent`) | Uses current weights for forward and backward passes with unmodified optimizer updates. |

## Retrospective gradient correction

| Method | Description |
|---|---|
| PipeDream (`stashing`) | Saves forward-time weights and reuses them for backward passes. [Narayanan et al., SOSP 2019](https://arxiv.org/abs/1806.03377). |
| PipeMare T2 (`recovery`) | Approximates forward-time weights for backward passes using an EMA of updates. [Yang et al., MLSys 2021](https://arxiv.org/abs/1910.05124). |

## Prospective gradient correction

| Method | Description |
|---|---|
| SpecTrain (`spectrain`) | Predicts weights for both forward and backward passes. [Chen et al., arXiv 2018](https://arxiv.org/abs/1809.02839). |
| Linear Weight Prediction (`lwp`) | Predicts forward-pass weights from the difference between consecutive weight versions. [Kosson et al., MLSys 2021](https://arxiv.org/abs/2003.11666). |
| PipeOptim (`pipeoptim_nostash`) | Predicts forward-pass weights from the base optimizer's update direction. [Guan et al., IEEE TKDE 2025](https://arxiv.org/abs/2312.00839). |

## Concurrent update correction

| Method | Description |
|---|---|
| **FRESCO** (`fresco`) | Projects optimizer updates away from protected in-flight activation directions. [Lee and Lee, NeurIPS 2026](https://arxiv.org/abs/2611.11238). |
