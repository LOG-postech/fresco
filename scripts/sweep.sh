#!/bin/bash
# Create a W&B hyperparameter search; W&B prints the agent command.
set -euo pipefail

search=${1:-fresco}
if [[ $# -gt 0 ]]; then
  shift
fi
case "$search" in
  fresco|methods|recovery|sync) ;;
  -h|--help)
    echo "Usage: bash scripts/sweep.sh [fresco|methods|recovery|sync] [wandb sweep options]"
    exit 0
    ;;
  *)
    echo "Unknown search: $search. Choose fresco, methods, recovery, or sync." >&2
    exit 2
    ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
exec uv run wandb sweep --project fresco "$repo_dir/scripts/sweep_${search}.yaml" "$@"
