# Rust Engine

`nilemini-engine` is the independent CPU inference implementation for the exported NileMini contract. It validates model configuration and SafeTensors, executes the full decoder, supports dense per-layer KV caching, greedy/top-k/top-p sampling, and exposes reusable artifact-backed generation through the server crate.

Implemented engine areas include configuration, tensor/weight loading, RMSNorm, RoPE, linear projection, stable softmax, GQA attention, SiTU-GLU, transformer blocks, full-model logits, KV-cache prefill/decode, sampling, generation, tokenizer/chat formatting, and parity tooling.

## Correctness gate

The smoke artifact was compared against JAX reference logits. The measured 10-token comparison recorded max absolute error `0.00518465042`, mean absolute error `0.000966181391`, RMSE `0.00121245466`, cosine similarity `0.999999107917`, and top-1 agreement `10/10`. Cached single-token decode also matched the engine's fresh-sequence path exactly for the recorded fixture.

These are engineering parity measurements on the smoke artifact, not model-quality results. Final trained weights must pass the same export/parity flow.

## Build and test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Artifact-backed service

`GenerationService` loads configuration, generation metadata, tokenizer, and model weights once. Each request receives a fresh KV cache; model/tokenizer state is reusable but decoding state is not shared. The server validates request limits before inference and returns structured errors rather than fabricated logits/text.
