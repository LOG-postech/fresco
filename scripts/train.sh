#!/bin/bash
# Compatibility entry point; train.py launches every stage.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec uv run --project "$repo_dir" python "$repo_dir/train.py" "$@"
