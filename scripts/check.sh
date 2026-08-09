#!/usr/bin/env bash

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

run() {
  echo "==> $*"
  "$@" || exit $?
}

artifact="${1:-artifacts/small-lm-8m}"

[ -f "$artifact/model.safetensors" ] || { echo "missing model artifact" >&2; exit 1; }

(
  cd "$artifact" || exit 1
  sha256sum -c SHA256SUMS
) || exit $?

run uv run --frozen pytest -q
run cargo fmt --all -- --check
run cargo clippy --workspace --all-targets -- -D warnings
run cargo test --workspace
run cargo run --quiet --release -p smalllm-engine --example parity -- "$artifact"
run bash scripts/smoke_server.sh "$artifact"

echo
echo "checks passed"
