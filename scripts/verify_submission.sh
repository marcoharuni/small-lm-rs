#!/usr/bin/env bash

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

artifact="${1:-artifacts/nilemini-8m-situ}"

pass() { printf '%-30s PASS\n' "$1"; }
fail() { printf '%-30s FAIL\n' "$1"; exit 1; }

echo "================================"
echo "NileMini submission verification"
echo "================================"

if [ ! -f "$artifact/model.safetensors" ]; then
  fail "Trained artifact"
fi
pass "Trained artifact"

(
  cd "$artifact" || exit 1
  sha256sum -c SHA256SUMS >/dev/null
) && pass "Artifact checksums" || fail "Artifact checksums"

if command -v uv >/dev/null 2>&1; then
  uv run --frozen pytest -q >/tmp/nilemini-python-tests.log 2>&1 \
    && pass "Python tests" \
    || { cat /tmp/nilemini-python-tests.log; fail "Python tests"; }
else
  fail "Python tests (uv missing)"
fi

cargo fmt --all -- --check >/tmp/nilemini-fmt.log 2>&1 \
  && pass "Rust format" \
  || { cat /tmp/nilemini-fmt.log; fail "Rust format"; }

cargo clippy --workspace --all-targets -- -D warnings >/tmp/nilemini-clippy.log 2>&1 \
  && pass "Rust clippy" \
  || { cat /tmp/nilemini-clippy.log; fail "Rust clippy"; }

cargo test --workspace >/tmp/nilemini-rust-tests.log 2>&1 \
  && pass "Rust tests" \
  || { cat /tmp/nilemini-rust-tests.log; fail "Rust tests"; }

cargo run --quiet --release -p nilemini-engine --example parity -- "$artifact" \
  >/tmp/nilemini-parity.log 2>&1 \
  && pass "JAX/Rust parity" \
  || { cat /tmp/nilemini-parity.log; fail "JAX/Rust parity"; }

bash scripts/smoke_server.sh "$artifact" >/tmp/nilemini-server-smoke-output.log 2>&1 \
  && pass "OpenAI chat server" \
  || { cat /tmp/nilemini-server-smoke-output.log; fail "OpenAI chat server"; }

echo
echo "SUBMISSION VERIFIED"
