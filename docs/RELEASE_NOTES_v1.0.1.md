# NileMini-8M-SiTU v1.0.1

`v1.0.1` is the finalized NileAGI inference-engine submission release.

## Included

- 7,999,744-parameter decoder-only NileMini model trained in JAX/Flax NNX
- final FP32 SafeTensors artifact with tokenizer, config, manifest, checksums, and JAX reference outputs
- independent Rust CPU Transformer inference engine
- grouped-query attention, RoPE, RMSNorm, SiTU-GLU, tied embeddings
- prompt prefill and KV-cached autoregressive decoding
- greedy, temperature, top-k, and top-p sampling
- multicore dense projections using Rayon
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions`
- `/health` and `/v1/models`
- SSE response framing ending in `data: [DONE]`
- simple browser chat interface served by the Rust server
- one-command submission verification gate
- final trained-artifact JAX ↔ Rust numerical parity report
- reproducible CPU benchmark tooling and measured submission-machine results

## Final correctness results

The final trained artifact was compared over 81,920 JAX/Rust logits:

- max absolute error: `0.0528898239`
- mean absolute error: `0.0052069233`
- RMSE: `0.0074163827`
- cosine similarity: `0.9999967275`
- top-1 agreement: `10/10`
- final-position top-1 match: `true`
- parity accepted: `true`

The final submission verification gate passed:

```text
Trained artifact               PASS
Artifact checksums             PASS
Python tests                   PASS
Rust format                    PASS
Rust clippy                    PASS
Rust tests                     PASS
JAX/Rust parity                PASS
OpenAI chat server             PASS

SUBMISSION VERIFIED
```

## CPU benchmark

Submission machine:

```text
CPU:          Intel(R) Core(TM) i5-4310U CPU @ 2.00GHz
Logical CPUs: 4
RAM:          7.6 GiB
Rust:         1.97.1
Rayon:        4 worker threads
```

Measured with the final trained FP32 artifact:

| Metric | 32 prompt + 32 decode | 128 prompt + 32 decode |
| --- | ---: | ---: |
| Model load | 53.16 ms | 45.65 ms |
| Prefill latency | 610.25 ms | 2462.19 ms |
| Prefill throughput | 52.44 tok/s | 51.99 tok/s |
| TTFT | 610.28 ms | 2462.22 ms |
| Decode throughput | 45.32 tok/s | 44.09 tok/s |
| Inter-token latency | 22.07 ms/token | 22.68 ms/token |
| Peak RSS | 65.61 MiB | 65.55 MiB |

See `benchmarks/RESULTS.md` for the raw measured values and environment details.

## Run

```bash
cargo run --release -p nilemini-server -- \
  --model-dir artifacts/nilemini-8m-situ \
  --host 127.0.0.1 \
  --port 8080
```

Then open `http://127.0.0.1:8080/` or call the OpenAI-compatible API directly.

## Verify

```bash
bash scripts/verify_submission.sh
bash scripts/benchmark.sh
```

This release intentionally emphasizes end-to-end systems correctness, reproducibility, independent Rust inference, and measured CPU serving behavior rather than frontier-scale model quality.
