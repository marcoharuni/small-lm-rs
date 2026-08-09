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

The transformer projection path rounds each activation to its BF16-equivalent value once per projection and reuses it across output neurons. For multi-token prefill, the projection weight matrix is also rounded once and reused across sequence rows instead of repeating the same conversion inside every row's dot products. Single-token cached decode keeps inline weight conversion to avoid allocating a temporary rounded matrix for every generated token.

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

## Performance diagnostics

For a coarse CPU breakdown of fresh-sequence prefill, run the opt-in stage profiler:

```bash
cargo run --release -p nilemini-engine --example stage_profile -- \
  artifacts/small-lm-8m 32 3
```

The final two arguments are prompt length and measurement iterations. The profiler warms the model path first, then reports embedding, transformer-stack, final-normalization, and tied-LM-head time separately. These timings are diagnostics for the current machine, not portable performance guarantees.

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
