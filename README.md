# small-lm-rs

**Train in JAX. Export to SafeTensors. Run entirely in Rust.**

`small-lm-rs` is an end-to-end small-language-model systems task: a decoder-only model is trained in JAX/Flax NNX, exported as framework-independent artifacts, and executed by an independent Rust CPU inference engine with KV-cached generation and an OpenAI-compatible server. Python and JAX are not required at inference time.

## Why JAX and Rust?

The choice of JAX and Rust was deliberate rather than a claim that they are universally better than PyTorch and C. I am already comfortable working with PyTorch, but at this stage I am learning JAX more deeply and wanted the training side of the task to give me more practical experience with its functional style and accelerator-oriented execution model.

For inference and serving, I chose Rust because I wanted a native runtime and HTTP server in one systems language, with strong type and memory safety and no Python dependency at inference time. I do not yet have practical experience with C, so choosing C would have introduced a second language-learning problem on top of implementing and validating the inference engine itself. Rust let me work close to the systems layer while remaining productive enough to build, test, benchmark, and debug the complete runtime end to end.

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
| Training | JAX / Flax NNX |
| Optimizers | Muon + AdamW |
| Runtime | Native Rust CPU |
| Decode baseline | ~44-45 tok/s on an Intel i5-4310U |
| Peak RSS baseline | ~66 MiB |

The repository contains the training pipeline, byte-level BPE tokenizer, deterministic data preparation, exported SafeTensors weights, Rust inference code, KV-cached decoding, sampling, browser chat, tests, parity fixtures, benchmarks, and an OpenAI-compatible API.

## Run

Requirements: **Git** and **Rust 1.85+** with Cargo. Python and JAX are not needed to run the exported model.

On a new machine:

```bash
git clone https://github.com/marcoharuni/small-lm-rs.git
cd small-lm-rs
cargo --version
./scripts/run_server.sh
```

If the repository is already cloned:

```bash
cd small-lm-rs
git pull
./scripts/run_server.sh
```

Keep the server terminal running, then open:

```text
http://127.0.0.1:8080/
```

On Linux you can open it directly with:

```bash
xdg-open http://127.0.0.1:8080/
```

Stop the server with `Ctrl+C`.

API endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

You can also chat from another terminal:

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

The bundled model was trained with **JAX/Flax NNX on an NVIDIA L4 GPU hosted by Modal**. The reproducible profiles are `configs/training/final_l4.json` for pretraining and `configs/training/final_sft_l4.json` for instruction tuning.

On a JAX-capable GPU machine, the complete flow is:

```bash
uv sync --locked --all-groups
uv run smalllm prepare --profile configs/training/final_l4.json
uv run smalllm pretrain --profile configs/training/final_l4.json
uv run smalllm sft \
  --profile configs/training/final_sft_l4.json \
  --base-profile configs/training/final_l4.json
uv run smalllm export --profile configs/training/final_sft_l4.json
```

The pipeline is **FineWeb-Edu -> pretraining -> SmolTalk SFT -> SafeTensors export**. Pretraining processed **140,017,664 tokens** and finished at validation loss **3.8659** (perplexity **47.74**). SFT used **448 training / 64 validation examples** and reached validation loss **2.6459**.

Training uses Muon for transformer projection matrices and AdamW for the remaining parameters. The exporter writes the Rust-consumable package to `exports/small-lm-8m/` by default.

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
| Max absolute error | 0.0465807915 |
| Mean absolute error | 0.0049628036 |
| RMSE | 0.0073501666 |
| Cosine similarity | 0.9999967821 |
| Top-1 agreement | 10 / 10 |

Rust and JAX therefore select the same top-1 token at every checked reference position. Transformer projections reproduce the JAX BF16-operand/FP32-accumulation path, while the tied LM head uses FP32 operands and FP32 accumulation to match the JAX reference exactly at that boundary.

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

The benchmark output records the exact Git revision and whether the working tree is clean so before/after measurements remain traceable. Full checked-in results are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

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

Pull-request CI verifies the Python pipeline, artifact SHA-256 checksums, Rust formatting and Clippy, the Rust test suite, the full 81,920-logit JAX/Rust parity fixture, and a live OpenAI-compatible server smoke test.

## Layout

```text
src/smalllm/               JAX training, evaluation, SFT, and export
rust/engine/                independent Rust inference engine
rust/server/                OpenAI-compatible HTTP server and browser chat
artifacts/small-lm-8m/      bundled trained artifact and parity fixtures
configs/                    architecture and training profiles
docs/                       architecture, data, training, engine, API, limitations
benchmarks/                 reproducible CPU benchmark data
scripts/                    train/export/run/check helpers
```

## Scope and limitations

This task is deliberately small enough to understand end to end. It demonstrates the complete path from raw data and tokenizer training through model training, artifact conversion, native inference, cached autoregressive generation, API serving, numerical validation, and CPU benchmarking.

It is not presented as a production-scale general-purpose assistant. Current limitations include the small parameter count, 512-token context, FP32 model storage/runtime, CPU-only execution, and limited instruction tuning. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## License

Apache-2.0 for the repository code and documentation. Dataset use remains subject to the upstream dataset terms.

## AI use disclosure

I used ChatGPT and Codex as engineering assistants during parts of the development process, mainly for debugging, reviewing implementation details, refactoring, strengthening tests and CI, repository cleanup, and improving documentation.

I made the model and systems design decisions, configured and ran the data preparation, pretraining, instruction-tuning and export workflows, ran the CPU benchmarks, tested the Rust runtime and API, reviewed and integrated changes, investigated failures, and validated the final implementation with the test suite and JAX/Rust numerical parity checks.

Thank you.
