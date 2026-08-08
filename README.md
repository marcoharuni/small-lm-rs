# small-lm-rs

A small decoder-only language model trained in JAX and served by an independent Rust CPU inference engine.

The repository contains the training pipeline, tokenizer, exported SafeTensors weights, Rust inference code, KV-cached generation, sampling, a browser chat, and an OpenAI-compatible API.

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
    "model": "nilemini-8m-situ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

The bundled artifact keeps its original model identifier, `nilemini-8m-situ`, so exported weights, manifests, tests, and parity fixtures remain reproducible. That identifier is not the project name.

## Model

The checked-in model has:

| | |
| --- | ---: |
| Parameters | 7,999,744 |
| Layers | 8 |
| Hidden size | 256 |
| FFN size | 704 |
| Vocabulary | 8,192 |
| Context | 512 |
| Query / KV heads | 4 / 2 |
| Head dimension | 64 |

The decoder uses RMSNorm, grouped-query attention, RoPE, a bounded gated FFN, tied embeddings, and no bias or dropout. The exact architecture is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Training

The base model was trained on 140,017,664 FineWeb-Edu tokens with 262,144 held-out validation tokens. The final base validation loss was 3.8659 (perplexity 47.74).

A small supervised fine-tuning pass used 448 SmolTalk training examples and 64 validation examples. Its purpose was to exercise the complete instruction-tuning and chat-serving path; the model is too small to be treated as a strong general-purpose assistant.

Training uses JAX/Flax NNX with Muon for the transformer projection matrices and AdamW for the remaining parameters. See [`docs/TRAINING.md`](docs/TRAINING.md).

## Rust engine

The inference implementation does not call JAX or Python at runtime. It loads the exported artifacts directly and implements:

- SafeTensors weight loading
- tokenizer and chat formatting
- RMSNorm and RoPE
- grouped-query causal attention
- feed-forward blocks
- prompt prefill and KV-cached decode
- greedy, temperature, top-k, and top-p sampling
- multicore dense projections with Rayon

See [`docs/ENGINE.md`](docs/ENGINE.md).

## JAX / Rust parity

The exported JAX reference and Rust engine were compared over 81,920 logits:

```text
max absolute error   0.0528898239
mean absolute error  0.0052069233
RMSE                 0.0074163827
cosine similarity    0.9999967275
top-1 agreement      10/10
```

Details and the machine-readable report are in [`docs/parity.md`](docs/parity.md) and [`docs/parity_report.json`](docs/parity_report.json).

## Benchmark

CPU measurements are kept in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). They are deliberately kept out of the chat UI.

Run them on another machine with:

```bash
bash scripts/benchmark.sh
```

## Development

```bash
uv sync --locked --all-groups
uv run --frozen pytest
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The quick repository check is:

```bash
bash scripts/check.sh
```

## Layout

```text
src/nilemini/               JAX training and export code
rust/engine/                Rust inference engine
rust/server/                HTTP server and browser chat
artifacts/nilemini-8m-situ/ bundled trained artifact
configs/                    model and training configuration
docs/                       implementation notes
benchmarks/                 CPU benchmark data
```

## License

Apache-2.0 for the repository code and documentation. Dataset use remains subject to the upstream dataset terms.
