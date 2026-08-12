# Final systems engineering report

This document summarizes the completed post-submission Rust inference-engine roadmap built on top of the original `small-lm-rs` submission.

The original submission remains preserved on `main`. The work described here was developed and validated on a chain of milestone branches and draft pull requests.

## Model and runtime boundary

The bundled model has 7,999,744 parameters, 8 transformer layers, hidden size 256, FFN size 704, 4 query heads, 2 KV heads, vocabulary size 8,192, and context length 512.

Training/export remains JAX -> SafeTensors. Inference remains native Rust with no Python/JAX/PyTorch dependency at runtime.

## Completed roadmap

| Milestone | Result |
| --- | --- |
| M1 | Continuous batching and shared multi-row cached decode |
| M2 | Symmetric per-output-channel weight-only INT8 projections |
| M3 | AVX2-assisted INT8 conversion/dequantization with scalar fallback |
| M4 | Request-local fixed-size paged KV storage with lazy allocation |
| M5 | Shared longest-prefix KV reuse across generation requests |
| M6 | Unified reproducible benchmark suite and consolidated results |
| M7 | Final documentation and project presentation |

FlashAttention-style tiled attention was intentionally removed from this shortened roadmap. The runtime therefore makes no claim of having a FlashAttention or PagedAttention implementation in the vLLM-specific sense.

## M1 — continuous batching

The server uses a long-lived generation worker and continuously admits queued requests up to the configured active-sequence capacity. Each request keeps independent sampler state and KV state.

Decode-ready rows are passed through a batched backend so expensive Q/K/V/O, FFN, final normalization, and tied-LM-head work can be evaluated across active rows instead of invoking the full model independently per request.

Prompt prefill remains request-local rather than chunked/batched prefill.

## M2 — weight-only INT8

Transformer Q/K/V/O and gate/up/down projection matrices can be exported as signed INT8 weights with one FP32 scale per output row.

Embeddings, tied LM-head source weights, normalization parameters, activations, and KV values remain FP32.

The checked artifact shrinks from 30.52 MiB to 13.73 MiB, a 55.03% serialized-size reduction.

This is weight-only quantization. It is not W8A8 integer-only matrix multiplication.

## M3 — AVX2-assisted INT8

On x86_64, runtime AVX2 detection selects an architecture-specific helper that converts/dequantizes INT8 weight chunks using AVX2 intrinsics. Non-AVX2 systems retain a scalar fallback.

The final ordered dot accumulation remains FP32. The implementation must therefore be described as **AVX2-assisted weight-only INT8**, not VNNI-style integer-only INT8 x INT8 compute.

On the measured Intel i5-4310U workload, AVX2 INT8 reduced peak RSS to about 31.72 MiB from a 65.46 MiB FP32 baseline while matching the checked decode top-1 checksum.

## M4 — paged KV cache

The production `KvCache` stores K/V rows in lazily allocated fixed-size 16-token pages rather than reserving a dense full-context buffer at cache creation.

Cached attention reads page-native key/value head slices. Whole-layer contiguous snapshots are materialized only when explicitly requested for diagnostics.

Truncation releases trailing pages and is used by generation rollback paths; `clear()` releases all pages.

For the bundled model, the dense 512-token FP32 KV payload is 4 MiB per request. With 16-token pages:

| Logical sequence length | Allocated KV payload | Saving vs dense reservation |
| ---: | ---: | ---: |
| 0 | 0 B | 100.000% |
| 1-16 | 128 KiB | 96.875% |
| 17-32 | 256 KiB | 93.750% |
| 128 | 1 MiB | 75.000% |
| 512 | 4 MiB | 0.000% |

Paging removes unused-capacity reservation. It does not compress active KV state, and at full context its payload converges to the dense payload size.

This is request-local lazy paging, not a global vLLM-style physical block allocator/free-list or shared block manager.

## M5 — prefix caching

The generation scheduler owns a bounded shared prefix cache. Entries contain token IDs, a cloneable paged-KV snapshot, and cached final logits.

New prompts use longest-token-prefix matching:

- an exact prompt hit restores cached KV/logits and skips prompt prefill;
- a partial hit restores the longest cached prefix and evaluates only the unmatched suffix;
- a miss performs normal prefill and stores the completed prompt for future reuse.

The current cache is correctness-first and bounded to 16 entries. It uses simple oldest-entry eviction rather than a sophisticated production admission/LRU policy.

On the M6 GitHub Actions runner, the checked 64-token exact hit took 0.049 ms versus 525.261 ms cold, and a 72-token prompt reusing 64 cached tokens took 94.952 ms versus 597.219 ms cold. These are machine/workload-specific ratios, not portable guarantees.

## M6 — consolidated benchmark suite

Run the default suite with:

```bash
bash scripts/benchmark_suite.sh artifacts/small-lm-8m
```

Include the real HTTP concurrency sweep with:

```bash
SMALLLM_BENCH_HTTP=1 bash scripts/benchmark_suite.sh artifacts/small-lm-8m
```

The suite records git revision/state, OS, CPU, logical CPU count, AVX2 support, Rust/Cargo versions, RAM, Rayon configuration, artifact path, and iteration count. Raw outputs are written below `benchmarks/runs/<UTC>/`.

The final M6 CI snapshot ran on an AMD EPYC 7763 GitHub runner with 4 logical CPUs and AVX2 support.

Selected values from that run:

| Metric | Result |
| --- | ---: |
| FP32 32-token prefill throughput | 127.41 tok/s |
| FP32 32-token decode throughput | 92.07 tok/s |
| FP32 peak RSS | 65.61 MiB |
| INT8 artifact-size reduction | 55.03% |
| AVX2 INT8 peak RSS | 31.93 MiB |
| Exact 64-token prefix hit | 0.049 ms |
| 64/72 partial-prefix hit | 94.952 ms |
| HTTP aggregate throughput, concurrency 1 | 34.55 tok/s |
| HTTP aggregate throughput, concurrency 8 | 59.57 tok/s |

The HTTP workload uses prompts with substantial shared token prefixes, so the final concurrency sweep reflects both continuous batching and prefix reuse and should not be used to isolate either optimization independently.

The server's SSE adapter still buffers completed generation before formatting chunks, so HTTP measurements are end-to-end request latency, not live-streaming TTFT/TPOT.

Full historical and final measurements are in [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## Correctness and compatibility

The FP32 JAX/Rust reference comparison covers 81,920 logits:

| Metric | Result |
| --- | ---: |
| Max absolute error | 0.0465807915 |
| Mean absolute error | 0.0049628036 |
| RMSE | 0.0073501666 |
| Cosine similarity | 0.9999967821 |
| Top-1 agreement | 10 / 10 |

The lossy INT8 path is validated separately. Its checked prefill cosine similarity is 0.999945633, cached-decode cosine similarity is 0.999976331, and the checked decode top-1 token agrees with FP32.

The milestone heads were also validated with Rust 1.85 MSRV checks, formatting, Clippy with warnings denied, Rust tests, Python checks, JAX/Rust parity, and live server smoke tests.

## Current engineering boundaries

The completed system is still intentionally a small CPU inference engine, not a claim of production-scale serving parity with vLLM/SGLang or frontier inference stacks.

Important remaining boundaries include:

- CPU-only execution;
- 8M-parameter bundled model and 512-token context;
- no FlashAttention-style tiled attention kernel;
- no global physical-page allocator or cross-request page sharing;
- simple 16-entry prefix-cache policy;
- no chunked/batched prefill scheduler;
- buffered rather than live token-progressive SSE;
- no speculative decoding;
- no INT4/FP8 path;
- no GPU kernels;
- no distributed tensor/pipeline/context/expert parallelism;
- no production authentication, TLS, quotas, or multi-tenant controls.

These are explicit scope boundaries, not hidden claims.

## Validation commands

Core checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv run --frozen pytest
```

Full repository verification:

```bash
bash scripts/check.sh
```

Full benchmark campaign:

```bash
SMALLLM_BENCH_HTTP=1 bash scripts/benchmark_suite.sh artifacts/small-lm-8m
```

## Branch preservation

The original/submission `main` branch is intentionally not modified by this milestone chain. M7 is based on the validated M6 branch, which in turn is based on M5 -> M4 -> M3 -> M2 -> M1.
