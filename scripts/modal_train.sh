#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec uvx --from 'modal==1.5.3' modal run --detach infra/modal_train.py "$@"
