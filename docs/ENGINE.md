# Rust inference engine

The Rust runtime executes the exported model directly. It does not call Python, JAX, PyTorch, or another model server during inference.

Implemented pieces include:

- SafeTensors and JSON configuration loading
- tokenizer and chat formatting
- RMSNorm and RoPE
- grouped-query causal attention
- SiTU-GLU feed-forward blocks
- tied output projection
- prompt prefill
- KV-cached token decoding
- greedy, temperature, top-k, and top-p sampling

## CPU path

Dense projection output elements are parallelized with Rayon. Each dot product keeps the reference accumulation order used by the JAX/Rust parity tests.

Weights, activations, and KV cache are currently stored in FP32. Transformer matrix projections use BF16-equivalent operands with FP32 accumulation to match the JAX training/reference path. The final tied vocabulary projection intentionally uses FP32 operands and FP32 accumulation because that is how the JAX reference computes the LM head.

The generic transformer projection path rounds each activation to its BF16-equivalent value once per projection and reuses it across output neurons, avoiding repeated conversion of the same activation inside every output dot product.

## Correctness

The checked-in artifact was compared against the JAX reference over 81,920 logits:

```text
max abs error   0.0465807915
mean abs error  0.0049628036
RMSE            0.0073501666
cosine          0.9999967821
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

The benchmark harness records the source revision and working-tree state together with hardware/runtime details. Measured results are kept in `benchmarks/RESULTS.md`.

## Tests

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Pull-request CI additionally verifies the bundled artifact checksums, runs the full JAX/Rust parity fixture, and starts the real server for an OpenAI-compatible API smoke test.
