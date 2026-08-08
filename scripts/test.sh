#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
uv sync --locked --all-groups
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy
JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --frozen pytest
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
