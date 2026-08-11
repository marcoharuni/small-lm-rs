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
- request-local generation sessions and KV caches
- continuous request scheduling with capacity reuse
- batched cached attention, transformer, final-normalization, and tied-LM-head decode
- a dedicated HTTP batching worker for concurrent generation requests
- symmetric per-output-channel weight-only INT8 transformer projections
- a mixed FP32/INT8 SafeTensors loader and quantized decode backend

## CPU path

Dense projection output elements are parallelized with Rayon. Each dot product keeps the reference accumulation order used by the JAX/Rust parity tests.

The baseline FP32 runtime stores weights, activations, and KV cache in FP32. Transformer matrix projections use BF16-equivalent operands with FP32 accumulation to match the JAX training/reference path. The final tied vocabulary projection intentionally uses FP32 operands and FP32 accumulation because that is how the JAX reference computes the LM head.

The transformer projection path rounds each activation to its BF16-equivalent value once per projection and reuses it across output neurons. For multi-token prefill, the projection weight matrix is also rounded once and reused across sequence rows instead of repeating the same conversion inside every row's dot products. Single-token cached decode keeps inline weight conversion to avoid allocating a temporary rounded matrix for every generated token.

## Weight-only INT8

Milestone 2 adds a correctness-first INT8 path without changing the FP32 execution path. The offline quantizer converts the transformer Q/K/V/O and gate/up/down projection matrices to symmetric signed INT8 using one FP32 scale per output row. Token embeddings, RMSNorm weights, and the tied LM-head source embedding remain FP32.

The mixed SafeTensors package is validated by `QuantizedModelWeights`, and `QuantizedDecodeModel` implements the same decode-backend contract used by generation and scheduling. Request-local KV caches remain FP32.

The current `linear_int8` implementation is deliberately a reference kernel rather than a claimed optimized integer GEMM: activations follow the engine's BF16-equivalent operand contract, INT8 weights are dequantized from their per-output-row scale during the dot product, and accumulation is FP32. This establishes a stable correctness boundary for Milestone 3, where dedicated SIMD/AVX2 quantized kernels can replace the hot loop without changing the artifact format or model-level semantics.

On the measured HP EliteBook Folio 9480m, the mixed artifact shrank from **30.52 MiB to 13.73 MiB (55.03%)** and peak RSS fell from **65.46 MiB to 31.89 MiB (51.3%)**. Over the repeatable 20-run microbenchmark, however, the reference INT8 path was slower: mean ten-token prefill increased from **108.003 ms to 150.859 ms (+39.7%)**, and one cached decode step increased from **17.576 ms to 20.157 ms (+14.7%)**. The decode top-1 checksum remained identical. These are machine- and workload-specific measurements, and no INT8 speedup is claimed before Milestone 3.

Generate and inspect the mixed artifact with:

```bash
bash scripts/quantize_int8.sh
cargo run --release -p smalllm-engine --example int8_compare -- artifacts/small-lm-8m
```

Run the repeatable timing comparison with:

```bash
cargo build --release -p smalllm-engine --example int8_benchmark
/usr/bin/time -v target/release/examples/int8_benchmark fp32 artifacts/small-lm-8m 20
/usr/bin/time -v target/release/examples/int8_benchmark int8 artifacts/small-lm-8m 20
```

## Continuous batching

The post-submission performance path keeps one long-lived `BatchedDecodeModel` inside a dedicated CPU worker thread. HTTP tasks tokenize requests, enqueue generation work, and await independent one-shot responses instead of locking the model for an entire request.

Each admitted sequence owns its sampler state and KV cache. `GenerationScheduler` continuously admits queued requests up to the configured active-sequence capacity, emits the token produced during prompt prefill, removes completed requests, and reuses released capacity. For decode-ready requests, the scheduler makes one backend batch call rather than one model call per sequence.

`BatchedDecodeModel` shares the expensive dense work across active decode rows. Q/K/V projection and the attention output projection are batched, while RoPE positions and KV-cache attention remain request-local. Transformer feed-forward projections, final RMSNorm, and the tied vocabulary projection are also evaluated over the active rows as a batch.

Prompt prefill is currently performed per request during admission; it is not yet a batched or chunked prefill scheduler. The OpenAI-compatible SSE adapter also currently buffers completed generation before formatting response chunks, so it should not be presented as live token streaming.

The server exposes batching capacity explicitly:

```bash
./scripts/run_server.sh --max-active-sequences 8
```

## Correctness

The checked-in FP32 artifact was compared against the JAX reference over 81,920 logits:

```text
max abs error   0.0465807915
mean abs error  0.0049628036
RMSE            0.0073501666
cosine          0.9999967821
top-1           10/10
```

Run the parity test with:

```bash
cargo run --release -p smalllm-engine --example parity -- \
  artifacts/small-lm-8m
```

The measured INT8 comparison used 81,920 prefill logits and 8,192 cached-decode logits. Prefill cosine similarity was **0.999945633**, cached-decode cosine similarity was **0.999976331**, and the checked cached-decode top-1 token matched FP32. Quantization is intentionally validated separately from the stricter FP32 JAX/Rust parity thresholds because lossy INT8 storage changes the numerical contract.

The batched scheduler is additionally checked against independent cached generation on the bundled artifact. CI also starts the real HTTP server and completes four simultaneous chat-completion requests to detect request-state mixing, deadlocks, and capacity-management regressions.

## Performance diagnostics

For a coarse CPU breakdown of fresh-sequence prefill, run the opt-in stage profiler:

```bash
cargo run --release -p smalllm-engine --example stage_profile -- \
  artifacts/small-lm-8m 32 3
```

The final two arguments are prompt length and measurement iterations. The profiler warms the model path first, then reports embedding, transformer-stack, final-normalization, and tied-LM-head time separately. These timings are diagnostics for the current machine, not portable performance guarantees.

## Benchmarks

Single-request engine benchmark:

```bash
bash scripts/benchmark.sh
```

Continuous-batching HTTP benchmark:

```bash
python3 scripts/benchmark_concurrency.py artifacts/small-lm-8m
```

The concurrency harness exercises 1, 2, 4, and 8 simultaneous clients, records end-to-end request latency distributions, aggregate generated-token throughput, peak server RSS, source revision, and working-tree state. Because the current SSE adapter buffers completed generation, it intentionally does not label any measurement as HTTP TTFT or TPOT.

Measured results are kept in `benchmarks/RESULTS.md`.

## Tests

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Pull-request CI additionally verifies the bundled artifact checksums, runs the full JAX/Rust parity fixture, starts the real server for an OpenAI-compatible API smoke test including concurrent requests, and runs the continuous-batching benchmark on the performance-roadmap PR.
