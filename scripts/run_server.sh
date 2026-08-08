#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: Cargo is required; install a stable Rust toolchain" >&2
  exit 127
fi

exec cargo run --package nilemini-server -- "$@"
