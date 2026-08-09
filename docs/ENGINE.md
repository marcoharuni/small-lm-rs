# Rust inference engine

The Rust runtime executes the exported model directly. It does not call Python, JAX, PyTorch, or another model server during inference.

Implemented pieces include:

- SafeTensors and JSON configuration loading
- tokenizer and chat formatting
- RMSNorm and RoPE
- grouped-query causal attention
- gated feed-forward blocks
- tied output projection
- prompt prefill
- KV-cached token decoding
- greedy, temperature, top-k, and top-p sampling

## CPU path

Dense projection output elements are parallelized with Rayon. Each dot product keeps the reference accumulation order used by the JAX/Rust parity tests.

Weights, activations, and KV cache are currently FP32. Matrix operands are rounded to BF16-equivalent values before multiplication to match the training/reference numerics.

## Correctness

The checked-in artifact was compared against the JAX reference over 81,920 logits:

```text
max abs error   0.0528898239
mean abs error  0.0052069233
RMSE            0.0074163827
cosine          0.9999967275
top-1           10/10
```

Run the parity test with:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/small-lm-8m
```

## Benchmark

```bash
bash scripts/benchmark.sh
```

Measured results are kept in `benchmarks/RESULTS.md`.

## Tests

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
