# CPU Benchmark Results

This file records **final trained-model CPU inference measurements** for the checked-in `nilemini-8m-situ` artifact.

Run:

```bash
bash scripts/benchmark.sh
```

The script records:

- UTC timestamp
- operating system
- CPU model
- logical CPU count
- installed memory
- Rust/Cargo versions
- Rayon thread configuration
- model load time
- prompt prefill latency
- prompt prefill tokens/s
- KV-cached decode latency
- decode milliseconds/token
- KV-cached decode tokens/s

Two fixed workloads are used:

```text
prompt=32 tokens,  decode=32 tokens
prompt=128 tokens, decode=32 tokens
```

## Submission-machine measurements

The final values must come from the actual submission machine and must not be estimated or copied from a development smoke model.

Paste the complete output of `bash scripts/benchmark.sh` here before the final release tag is created.

## Numerical correctness

Performance measurements are separate from correctness. The final trained artifact's JAX↔Rust parity report is in [`../docs/parity.md`](../docs/parity.md) and [`../docs/parity_report.json`](../docs/parity_report.json).
