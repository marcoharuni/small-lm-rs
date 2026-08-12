# CPU benchmark

Measurements below were collected with the bundled FP32 model on:

```text
CPU            Intel Core i5-4310U @ 2.00 GHz
logical CPUs   4
RAM            7.6 GiB
Rust           1.97.1
Rayon threads  4
```

| Metric | 32 prompt + 32 decode | 128 prompt + 32 decode |
| --- | ---: | ---: |
| Model load | 53.16 ms | 45.65 ms |
| Prefill | 610.25 ms | 2462.19 ms |
| Prefill throughput | 52.44 tok/s | 51.99 tok/s |
| TTFT | 610.28 ms | 2462.22 ms |
| Decode throughput | 45.32 tok/s | 44.09 tok/s |
| Inter-token latency | 22.07 ms/token | 22.68 ms/token |
| Peak RSS | 65.61 MiB | 65.55 MiB |

These are machine-specific measurements, not performance guarantees.

Re-run on another machine with:

```bash
bash scripts/benchmark.sh
```

## Continuous batching HTTP benchmark

The post-submission `perf-roadmap` branch was also benchmarked through the real OpenAI-compatible HTTP server after integrating the continuous scheduler and `BatchedDecodeModel` into a dedicated batching worker.

The measurements below are from GitHub Actions CI run **#194** on 10 August 2026. The branch head was `9da16611c538789d29a063debc2b7ef4f33da9ec`; GitHub tested the PR merge checkout `de3f8be310cd6d6f28e45f5e2be4e82582f7b019` with a clean working tree.

```text
platform              Linux 6.17 Azure x86_64
logical CPUs          4
max active sequences  8
new tokens/request    8
rounds/concurrency    3
peak server RSS       54.34 MiB
```

| Concurrent requests | Samples | Aggregate generated tok/s | Mean latency | p50 | p95 | p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 33.23 | 239.98 ms | 228.86 ms | 261.02 ms | 263.88 ms |
| 2 | 6 | 33.91 | 471.15 ms | 470.89 ms | 473.13 ms | 473.18 ms |
| 4 | 12 | 36.21 | 882.63 ms | 878.84 ms | 895.10 ms | 895.18 ms |
| 8 | 24 | 36.51 | 1750.34 ms | 1728.24 ms | 1810.97 ms | 1810.99 ms |

On this runner and workload, aggregate generated-token throughput increased from **33.23 tok/s at concurrency 1 to 36.51 tok/s at concurrency 8**, an increase of about **9.9%**. Request latency increases with concurrency because the same four logical CPUs are shared across more active sequences.

These HTTP measurements are machine- and workload-specific and should not be compared directly with the Intel i5 cached-decode microbenchmark above. The older table measures the engine's cached decode path on different hardware; this table measures complete HTTP requests including prompt processing, scheduling, batched decode, tokenization, response construction, and network-loopback overhead.

The current SSE adapter formats chunks only after generation is complete. Therefore this benchmark intentionally reports **end-to-end request latency**, not HTTP TTFT or TPOT; reporting buffered SSE timing as token-streaming latency would be misleading.

Re-run the continuous-batching benchmark with:

```bash
python3 scripts/benchmark_concurrency.py artifacts/small-lm-8m
```

## Weight-only INT8 benchmark

Milestone 2 adds symmetric per-output-channel INT8 storage for the transformer projection matrices while keeping embeddings and normalization parameters in FP32. The reference `linear_int8` implementation dequantizes weights during the dot product and accumulates in FP32.

The measurements below were collected locally on an **HP EliteBook Folio 9480m** on 11 August 2026. They are machine-specific and should not be interpreted as portable speedups or slowdowns.

### Artifact size and numerical agreement

| Metric | Result |
| --- | ---: |
| FP32 artifact | 30.52 MiB |
| INT8 artifact | 13.73 MiB |
| Artifact-size reduction | 55.03% |
| Prefill values compared | 81,920 |
| Prefill max absolute error | 0.171354294 |
| Prefill mean absolute error | 0.024274951 |
| Prefill RMSE | 0.031206548 |
| Prefill cosine similarity | 0.999945633 |
| Decode values compared | 8,192 |
| Decode max absolute error | 0.109609604 |
| Decode mean absolute error | 0.018964311 |
| Decode RMSE | 0.023685229 |
| Decode cosine similarity | 0.999976331 |
| Decode top-1 agreement | true |

The INT8 path therefore preserves very high logit-direction agreement and selected the same checked decode top-1 token, while materially reducing the serialized model size.

### 20-run timing and peak RSS

Both modes use the same ten-token prompt and one cached decode step. The benchmark warms the model once before measuring 20 iterations. Peak RSS is collected by `/usr/bin/time -v` in separate FP32 and INT8 processes.

| Metric | FP32 | Milestone-2 INT8 | Change vs FP32 |
| --- | ---: | ---: | ---: |
| Mean prefill latency | 108.003 ms | 150.859 ms | +39.7% |
| Mean cached-decode latency | 17.576 ms | 20.157 ms | +14.7% |
| Peak RSS | 65.46 MiB | 31.89 MiB | -51.3% |
| Decode top-1 checksum | 5260 | 5260 | identical |

These results show the Milestone-2 tradeoff clearly: **weight-only INT8 cuts resident memory by about half, but the scalar/reference dequantization path is slower than FP32 on this machine**.

## AVX2-assisted INT8 benchmark

Milestone 3 adds runtime AVX2 detection on x86_64, an architecture-specific AVX2 helper isolated behind a narrow unsafe boundary, and a scalar fallback for non-AVX2 CPUs. AVX2 accelerates INT8-to-FP32 scale/dequantization in eight-weight chunks. Multi-row prefill dequantizes each projection matrix once and reuses it across input rows; single-token cached decode keeps an allocation-free ordered dot-product path.

The same HP EliteBook Folio 9480m reports AVX2 support with an Intel Core i5-4310U. The numerical-comparison output is unchanged from the Milestone-2 INT8 path: prefill cosine similarity is **0.999945633**, decode cosine similarity is **0.999976331**, and decode top-1 agreement remains true.

### 20-run AVX2 timing and peak RSS

| Metric | FP32 | Milestone-2 INT8 | Milestone-3 AVX2 INT8 |
| --- | ---: | ---: | ---: |
| Mean prefill latency | 108.003 ms | 150.859 ms | 104.654 ms |
| Mean cached-decode latency | 17.576 ms | 20.157 ms | 17.025 ms |
| Peak RSS | 65.46 MiB | 31.89 MiB | 31.72 MiB |
| Decode top-1 checksum | 5260 | 5260 | 5260 |

On this machine and workload, Milestone 3 reduces INT8 prefill latency by about **30.6%** and cached-decode latency by about **15.5%** relative to the Milestone-2 reference INT8 path. Relative to the earlier FP32 20-run baseline, AVX2 INT8 is about **3.1% lower-latency** for both prefill and cached decode while retaining roughly half the peak RSS.

This should be described precisely as **AVX2-assisted weight-only INT8 execution**. Weights remain INT8 in storage; AVX2 accelerates dequantization/conversion; the ordered dot-product accumulation remains FP32. It is not VNNI-style integer-only INT8×INT8 compute.

Generate the INT8 artifact and compare logits with:

```bash
bash scripts/quantize_int8.sh
cargo run --release -p smalllm-engine --example int8_compare -- artifacts/small-lm-8m
```

Run the repeatable timing benchmark with:

```bash
cargo build --release -p smalllm-engine --example int8_benchmark
/usr/bin/time -v target/release/examples/int8_benchmark fp32 artifacts/small-lm-8m 20
/usr/bin/time -v target/release/examples/int8_benchmark int8 artifacts/small-lm-8m 20
```
