#!/bin/bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_dir/scripts/train.sh" \
  --method.name spectrain \
  --optimizer.momentum 0.9 \
  --lr.schedule.peak_value 3e-4 \
  "$@"
