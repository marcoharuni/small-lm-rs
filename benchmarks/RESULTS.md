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
