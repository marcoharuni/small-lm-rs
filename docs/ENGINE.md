# Rust Engine

`nilemini-engine` is the independent CPU inference implementation for the exported NileMini model. It does not call Python, JAX, PyTorch, or an external model server at inference time.

The engine implements configuration and SafeTensors loading, tokenizer/chat formatting, RMSNorm, RoPE, grouped-query attention, SiTU-GLU, transformer blocks, tied output projection, KV-cached prompt prefill, one-token cached decode, greedy/top-k/top-p sampling, and artifact-backed generation.

## CPU execution

Dense projections use the canonical exported `[out_features, in_features]` layout. Each output dot product preserves the reference accumulation order while independent output elements are distributed across a Rayon CPU worker pool. This keeps the numerical parity contract while using the available multicore CPU for the dominant dense projections.

The runtime also uses:

- FP32 weights, activations, and KV cache;
- BF16-equivalent input/weight rounding before dot products to match the JAX reference path;
- grouped-query KV storage rather than duplicating KV heads;
- prompt prefill followed by KV-cached one-token decoding;
- request-isolated KV caches;
- release-mode execution for performance measurements.

## Final trained-model parity

The checked-in trained artifact was compared against the exported JAX reference over a `[1, 10, 8192]` logit tensor (81,920 logits):

- max absolute error: `0.0528898239`
- mean absolute error: `0.0052069233`
- RMSE: `0.0074163827`
- cosine similarity: `0.9999967275`
- top-1 agreement: `10/10`
- final-position top-1 match: `true`
- accepted: `true`
- violations: none

Reproduce the parity check with:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/nilemini-8m-situ
```

## CPU benchmark

Run the release benchmark on the review machine with:

```bash
bash scripts/benchmark.sh
```

The benchmark records hardware/runtime information plus model-load time, prefill latency and tokens/s, cached-decode latency and tokens/s for fixed 32-token and 128-token prompts.

Measured submission-machine results are recorded in [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## Build and test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For the complete submission gate:

```bash
bash scripts/verify_submission.sh
```

## Server integration

`GenerationService` loads configuration, generation metadata, tokenizer, and model weights once. Each generation call receives a fresh KV cache. The asynchronous Rust server validates request controls, executes CPU inference through a serialized model service, and returns structured OpenAI-shaped responses rather than fabricated logits or text.
