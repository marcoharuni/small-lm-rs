#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec uv run smalllm pretrain --profile configs/training/pilot_l4.json "$@"
