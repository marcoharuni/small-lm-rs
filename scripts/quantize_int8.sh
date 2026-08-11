#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

input="${1:-artifacts/small-lm-8m/model.safetensors}"
output="${2:-artifacts/small-lm-8m/model.int8.safetensors}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to run the INT8 quantizer" >&2
  exit 127
fi

uv run --frozen python -m smalllm.quantization "$input" "$output"

input_bytes="$(stat -c%s "$input")"
output_bytes="$(stat -c%s "$output")"
python - "$input_bytes" "$output_bytes" <<'PY'
import sys

source = int(sys.argv[1])
quantized = int(sys.argv[2])
reduction = 100.0 * (1.0 - quantized / source)
print(f"FP32 artifact: {source / 1024 / 1024:.2f} MiB")
print(f"INT8 artifact: {quantized / 1024 / 1024:.2f} MiB")
print(f"size reduction: {reduction:.2f}%")
PY
