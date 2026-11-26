#!/bin/bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_dir/scripts/train.sh" \
  --method.name inconsistent \
  --lr.schedule.peak_value 3e-4 \
  "$@"
