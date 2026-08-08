#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
echo 'The NileMini tokenizer is frozen; validating the checked-in artifact instead of retraining it.'
exec uv run nilemini doctor
