#!/bin/bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_dir/scripts/train.sh" \
  --method.name recovery \
  --method.discrepancy_correction 0.02 \
  --lr.schedule.peak_value 1e-4 \
  "$@"
