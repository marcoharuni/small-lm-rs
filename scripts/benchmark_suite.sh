#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

artifact="${1:-artifacts/small-lm-8m}"
iterations="${SMALLLM_BENCH_ITERATIONS:-20}"
run_http="${SMALLLM_BENCH_HTTP:-0}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="benchmarks/runs/$stamp"
mkdir -p "$out_dir"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required" >&2
  exit 127
fi
if [ ! -f "$artifact/model.safetensors" ]; then
  echo "error: missing model artifact: $artifact/model.safetensors" >&2
  exit 1
fi

capture() {
  local name="$1"
  shift
  echo
  echo "=== $name ==="
  "$@" 2>&1 | tee "$out_dir/$name.txt"
}

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  if git diff --quiet --ignore-submodules HEAD -- 2>/dev/null && \
     git diff --cached --quiet --ignore-submodules HEAD -- 2>/dev/null; then
    echo "git_state=clean"
  else
    echo "git_state=dirty"
  fi
  echo "os=$(uname -srmo)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "logical_cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
  if command -v lscpu >/dev/null 2>&1; then
    echo "cpu=$(lscpu | awk -F: '/Model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}')"
    echo "avx2=$(lscpu | grep -qw avx2 && echo true || echo false)"
  fi
  if command -v free >/dev/null 2>&1; then
    echo "memory=$(free -h | awk '/Mem:/ {print $2}')"
  fi
  echo "rayon_threads=${RAYON_NUM_THREADS:-all available logical CPUs}"
  echo "artifact=$artifact"
  echo "iterations=$iterations"
  echo "http_benchmark=$run_http"
} | tee "$out_dir/metadata.txt"

capture fp32_cached_32x32 \
  cargo run --quiet --release -p smalllm-engine --example cached_benchmark -- \
  "$artifact" 32 32

capture fp32_cached_128x32 \
  cargo run --quiet --release -p smalllm-engine --example cached_benchmark -- \
  "$artifact" 128 32

if [ ! -f "$artifact/model.int8.safetensors" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required to generate the INT8 benchmark artifact" >&2
    exit 127
  fi
  capture int8_quantize \
    bash scripts/quantize_int8.sh \
    "$artifact/model.safetensors" "$artifact/model.int8.safetensors"
fi

capture int8_compare \
  cargo run --quiet --release -p smalllm-engine --example int8_compare -- \
  "$artifact"

cargo build --quiet --release -p smalllm-engine --example int8_benchmark

capture int8_fp32_timing \
  /usr/bin/time -v target/release/examples/int8_benchmark fp32 "$artifact" "$iterations"

capture int8_avx2_timing \
  /usr/bin/time -v target/release/examples/int8_benchmark int8 "$artifact" "$iterations"

capture paged_kv \
  cargo run --quiet --release -p smalllm-engine --example paged_kv_benchmark -- \
  "$artifact"

capture prefix_cache \
  cargo run --quiet --release -p smalllm-engine --example prefix_cache_benchmark -- \
  "$artifact" "$iterations"

if [ "$run_http" = "1" ]; then
  echo
  echo "=== http_concurrency ==="
  python3 scripts/benchmark_concurrency.py "$artifact" \
    2>&1 | tee "$out_dir/http_concurrency.json"
else
  printf '%s\n' \
    "HTTP concurrency benchmark skipped." \
    "Set SMALLLM_BENCH_HTTP=1 to include it." \
    | tee "$out_dir/http_concurrency_skipped.txt"
fi

cat <<EOF | tee "$out_dir/README.txt"
Benchmark suite complete.
Raw outputs: $out_dir

Re-run default suite:
  bash scripts/benchmark_suite.sh $artifact

Include HTTP concurrency:
  SMALLLM_BENCH_HTTP=1 bash scripts/benchmark_suite.sh $artifact
EOF
