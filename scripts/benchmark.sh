#!/usr/bin/env bash

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

artifact="${1:-artifacts/small-lm-8m}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required" >&2
  exit 127
fi

if [ ! -f "$artifact/model.safetensors" ]; then
  echo "error: missing model artifact: $artifact/model.safetensors" >&2
  exit 1
fi

git_head="unknown"
git_state="unknown"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_head="$(git rev-parse HEAD)"
  if git diff --quiet --ignore-submodules HEAD -- && git diff --cached --quiet --ignore-submodules HEAD --; then
    git_state="clean"
  else
    git_state="dirty"
  fi
fi

echo "SmallLM CPU benchmark"
echo "====================="
echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_head: $git_head"
echo "git_state: $git_state"
echo "os: $(uname -srmo)"
echo "rustc: $(rustc --version)"
echo "cargo: $(cargo --version)"
echo "logical_cpus: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
if command -v lscpu >/dev/null 2>&1; then
  echo "cpu: $(lscpu | awk -F: '/Model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}')"
fi
if command -v free >/dev/null 2>&1; then
  echo "memory: $(free -h | awk '/Mem:/ {print $2}')"
fi
echo "rayon_threads: ${RAYON_NUM_THREADS:-all available logical CPUs}"
echo "artifact: $artifact"
echo

echo "--- workload: prompt=32, decode=32 ---"
cargo run --quiet --release -p smalllm-engine --example cached_benchmark -- \
  "$artifact" 32 32 || exit 1
echo

echo "--- workload: prompt=128, decode=32 ---"
cargo run --quiet --release -p smalllm-engine --example cached_benchmark -- \
  "$artifact" 128 32 || exit 1
