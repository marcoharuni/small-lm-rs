# small-lm-rs

**Train in JAX. Export to SafeTensors. Run entirely in Rust.**

`small-lm-rs` is an end-to-end small-language-model systems project: a decoder-only model is trained in JAX/Flax NNX, exported as framework-independent artifacts, and executed by an independent Rust CPU inference engine with KV-cached generation and an OpenAI-compatible server. Python and JAX are not required at inference time.

## At a glance

| | |
| --- | ---: |
| Parameters | 7,999,744 |
| Pretraining tokens | 140,017,664 |
| Layers | 8 |
| Hidden / FFN size | 256 / 704 |
| Query / KV heads | 4 / 2 |
| Vocabulary | 8,192 |
| Context | 512 |
| Activation | SiTU-GLU |
| Runtime | Native Rust CPU |
| Decode baseline | ~44-45 tok/s on an Intel i5-4310U |
| Peak RSS baseline | ~66 MiB |

The repository contains the training pipeline, byte-level BPE tokenizer, deterministic data preparation, exported SafeTensors weights, Rust inference code, KV-cached decoding, sampling, browser chat, tests, parity fixtures, benchmarks, and an OpenAI-compatible API.

## Run

```bash
./scripts/run_server.sh
```

Then open `http://127.0.0.1:8080/`.

API endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

Example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "small-lm-8m",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

## Architecture

The checked-in model is a pre-normalized decoder-only transformer:

```text
embeddings
  -> [ RMSNorm -> causal GQA + RoPE -> residual
       RMSNorm -> SiTU-GLU FFN       -> residual ] x 8
  -> RMSNorm
  -> tied vocabulary projection
```

The model uses RMSNorm, grouped-query attention, RoPE, SiTU-GLU with bounded gate/up branches, tied embeddings, and no bias or dropout. The exact numeric contract is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`configs/model.json`](configs/model.json).

## Training

The bundled base model was trained on **140,017,664 FineWeb-Edu tokens** with **262,144 held-out validation tokens**. Final base validation loss was **3.8659** (perplexity **47.74**).

The bundled instruction-tuning pass used the reproducible `onehour_sft` profile: **448 SmolTalk training examples** and **64 validation examples**, reaching validation loss **2.6459**. Its purpose is to exercise the complete instruction-tuning and chat-serving path; an 8M-parameter model with this limited SFT set should not be treated as a strong general-purpose assistant.

Training uses JAX/Flax NNX with Muon for transformer projection matrices and AdamW for the remaining parameters. Parameters are stored in FP32; matrix products use BF16 operands with FP32 accumulation.

See [`docs/TRAINING.md`](docs/TRAINING.md), [`docs/DATA_CARD.md`](docs/DATA_CARD.md), and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Export boundary

Training and inference are intentionally separated:

```text
JAX training
    |
    | export
    v
SafeTensors + tokenizer.json + config.json + generation_config.json
    |
    v
Rust inference engine
    |
    v
OpenAI-compatible HTTP server
```

The artifact manifest and SHA-256 checksums make the checked-in model files verifiable. See [`docs/EXPORT_FORMAT.md`](docs/EXPORT_FORMAT.md).

## Rust inference engine

The runtime does **not** call JAX, PyTorch, Python, or another model server. It directly implements:

- SafeTensors loading and model-layout validation
- tokenizer and chat formatting
- RMSNorm and RoPE
- grouped-query causal attention
- SiTU-GLU feed-forward blocks
- tied output projection
- prompt prefill
- KV-cached token-by-token decoding
- greedy, temperature, top-k, and top-p sampling
- multicore dense projections with Rayon

See [`docs/ENGINE.md`](docs/ENGINE.md) and [`docs/cached_decoding.md`](docs/cached_decoding.md).

## JAX / Rust correctness

The exported JAX reference and Rust engine were compared over **81,920 logits**:

| Metric | Result |
| --- | ---: |
| Max absolute error | 0.0528898239 |
| Mean absolute error | 0.0052069233 |
| RMSE | 0.0074163827 |
| Cosine similarity | 0.9999967275 |
| Top-1 agreement | 10 / 10 |

Rust and JAX therefore select the same top-1 token at every checked reference position. Numerical differences remain because the Rust runtime reproduces BF16-equivalent matrix operands while executing on CPU.

Details and the machine-readable report are in [`docs/parity.md`](docs/parity.md) and [`docs/parity_report.json`](docs/parity_report.json).

## Measured CPU baseline

The bundled FP32 engine was measured on an **Intel Core i5-4310U @ 2.00 GHz**, 4 logical CPUs, and 7.6 GiB RAM:

| Metric | 32 prompt + 32 decode | 128 prompt + 32 decode |
| --- | ---: | ---: |
| Prefill throughput | 52.44 tok/s | 51.99 tok/s |
| Decode throughput | 45.32 tok/s | 44.09 tok/s |
| Inter-token latency | 22.07 ms/token | 22.68 ms/token |
| Peak RSS | 65.61 MiB | 65.55 MiB |

These are machine-specific measurements, not performance guarantees. Re-run them on another machine with:

```bash
bash scripts/benchmark.sh
```

Full results are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Reproducibility and checks

```bash
uv sync --locked --all-groups
uv run --frozen pytest
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The complete local verification path additionally checks artifact hashes, JAX/Rust parity, and the server smoke test:

```bash
bash scripts/check.sh
```

CI runs Python lint/type/tests and Rust formatting, Clippy, and workspace tests on pull requests and `main`.

## Layout

```text
src/nilemini/               JAX training, evaluation, SFT, and export
rust/engine/                independent Rust inference engine
rust/server/                OpenAI-compatible HTTP server and browser chat
artifacts/small-lm-8m/      bundled trained artifact and parity fixtures
configs/                    architecture and training profiles
docs/                       architecture, data, training, engine, API, limitations
benchmarks/                 reproducible CPU benchmark data
scripts/                    train/export/run/check helpers
```

## Scope and limitations

This project is deliberately small enough to understand end to end. It demonstrates the complete path from raw data and tokenizer training through model training, artifact conversion, native inference, cached autoregressive generation, API serving, numerical validation, and CPU benchmarking.

It is not presented as a production-scale general-purpose assistant. Current limitations include the small parameter count, 512-token context, FP32 model storage/runtime, CPU-only execution, and limited instruction tuning. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## License

Apache-2.0 for the repository code and documentation. Dataset use remains subject to the upstream dataset terms.
