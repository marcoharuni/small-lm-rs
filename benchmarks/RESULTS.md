# CPU Benchmark Results

This file records **final trained-model CPU inference measurements** for the checked-in `nilemini-8m-situ` artifact.

Run:

```bash
bash scripts/benchmark.sh
```

## Submission-machine environment

Measured on `2026-08-08T05:21:10Z`:

```text
OS:           Linux 6.14.0-37-generic x86_64 GNU/Linux
CPU:          Intel(R) Core(TM) i5-4310U CPU @ 2.00GHz
Logical CPUs: 4
RAM:          7.6 GiB
Rust:         rustc 1.97.1 (8bab26f4f 2026-07-14)
Cargo:        cargo 1.97.1 (c980f4866 2026-06-30)
Rayon:        4 worker threads
Artifact:     artifacts/nilemini-8m-situ
```

The benchmark uses the final trained FP32 artifact and the multicore Rust inference path. TTFT is measured from the start of prompt prefill through selection of the first greedy token. Inter-token latency is the mean KV-cached decode time per token over the declared decode window. Peak RSS is the Linux process high-water resident set (`VmHWM`).

## Results

| Metric | 32-token prompt + 32 decode | 128-token prompt + 32 decode |
| --- | ---: | ---: |
| Model load | 53.16 ms | 45.65 ms |
| Prompt tokens | 32 | 128 |
| Prefill latency | 610.25 ms | 2462.19 ms |
| Prefill throughput | 52.44 tok/s | 51.99 tok/s |
| TTFT | 610.28 ms | 2462.22 ms |
| Decode tokens | 32 | 32 |
| Decode latency | 706.10 ms | 725.77 ms |
| Decode throughput | 45.32 tok/s | 44.09 tok/s |
| Inter-token latency | 22.07 ms/token | 22.68 ms/token |
| Peak RSS | 65.61 MiB | 65.55 MiB |
| Final KV-cache length | 64 | 160 |

Raw measured values:

```json
{
  "workload_32_32": {
    "model": "nilemini-8m-situ",
    "rayon_threads": 4,
    "prompt_tokens": 32,
    "decode_tokens": 32,
    "model_load_milliseconds": 53.160736,
    "prefill_milliseconds": 610.246163,
    "prefill_tokens_per_second": 52.43785531184077,
    "time_to_first_token_milliseconds": 610.280083,
    "first_token_id": 204,
    "decode_milliseconds": 706.104819,
    "decode_milliseconds_per_token": 22.06577559375,
    "decode_tokens_per_second": 45.31905057002592,
    "inter_token_latency_milliseconds": 22.06577559375,
    "peak_rss_mebibytes": 65.61328125,
    "final_cache_length": 64
  },
  "workload_128_32": {
    "model": "nilemini-8m-situ",
    "rayon_threads": 4,
    "prompt_tokens": 128,
    "decode_tokens": 32,
    "model_load_milliseconds": 45.647740999999996,
    "prefill_milliseconds": 2462.185791,
    "prefill_tokens_per_second": 51.9863287603547,
    "time_to_first_token_milliseconds": 2462.216143,
    "first_token_id": 45,
    "decode_milliseconds": 725.7684429999999,
    "decode_milliseconds_per_token": 22.680263843749998,
    "decode_tokens_per_second": 44.091197831262,
    "inter_token_latency_milliseconds": 22.680263843749998,
    "peak_rss_mebibytes": 65.5546875,
    "final_cache_length": 160
  }
}
```

These values are machine-specific measurements, not portable performance guarantees.

## Numerical correctness

Performance measurements are separate from correctness. The same final trained artifact passes the repository submission gate, including artifact checksums, Python tests, Rust formatting/clippy/tests, JAX↔Rust parity, server startup, `/health`, `/v1/models`, non-streaming `/v1/chat/completions`, and SSE completion framing.

See [`../docs/parity.md`](../docs/parity.md) and [`../docs/parity_report.json`](../docs/parity_report.json).
