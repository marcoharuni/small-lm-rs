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
