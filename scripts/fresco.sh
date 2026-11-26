#!/bin/bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_dir/scripts/train.sh" \
  --method.name fresco \
  --method.proj_rank_ratio 0.25 \
  --lr.schedule.peak_value 3e-4 \
  "$@"
