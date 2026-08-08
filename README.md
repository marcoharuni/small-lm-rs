# NileMini-8M-SiTU

[![CI](https://github.com/marcoharuni/nilemini-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/marcoharuni/nilemini-rs/actions/workflows/ci.yml)

**NileMini-8M-SiTU** is a **7,999,744-parameter decoder-only language model** trained in JAX/Flax NNX and executed by an independent Rust CPU inference engine.

The repository contains the complete trained release: deterministic data/training code, frozen tokenizer, FP32 SafeTensors weights, JAX reference outputs, Rust Transformer inference, KV-cached generation, sampling, a browser chat UI, and an OpenAI-compatible HTTP API.

## Chat with NileMini

Clone the repository and start the Rust server:

```bash
git clone https://github.com/marcoharuni/nilemini-rs.git
cd nilemini-rs

cargo run --release -p nilemini-server -- \
  --model-dir artifacts/nilemini-8m-situ \
  --host 127.0.0.1 \
  --port 8080
```

Then open:

```text
http://127.0.0.1:8080/
```

The browser interface is served directly by the Rust binary and talks to the same `/v1/chat/completions` endpoint as API clients. No Node.js frontend or second web server is required.

> NileMini is intentionally tiny. The 8M-parameter scale and small SFT pass make this primarily a language-model systems/reproducibility project, not a frontier chat model.

## Release results

| Item | Result |
| --- | ---: |
| Parameters | **7,999,744** |
| Vocabulary | **8,192** |
| Context length | **512** |
| Base dataset | FineWeb-Edu |
| Pretraining tokens | **140,017,664** |
| Pretraining updates | **4,273** |
| Final base validation loss | **3.8659** |
| Final base validation perplexity | **47.74** |
| SFT dataset | SmolTalk |
| SFT split | **448 train / 64 validation** |
| SFT updates | **56** |
| Final SFT validation loss | **2.6459** |
| Exported FP32 weights | **30.52 MiB** |
| JAX ↔ Rust logits compared | **81,920** |
| JAX ↔ Rust top-1 agreement | **10/10 (100%)** |
| JAX ↔ Rust cosine similarity | **0.9999967** |
| Parity accepted | **true** |

The SFT pass is intentionally small. It validates the instruction-tuning and chat-serving path; it is not presented as a large-scale instruction-tuning run.

## Architecture

The cross-language contract in [`configs/model.json`](configs/model.json) is:

- decoder-only Transformer
- 8,192-token byte-level BPE vocabulary
- 512-token context
- 8 transformer layers
- hidden size 256
- feed-forward size 704
- grouped-query attention: 4 query heads / 2 KV heads
- head dimension 64
- RMSNorm epsilon `1e-5`
- interleaved RoPE theta `10000`
- SiTU-GLU with beta gate `4` and beta up `25`
- tied token embedding / output projection
- no bias
- no dropout
- exact parameter count: **7,999,744**

Reserved token IDs are fixed:

```text
0 <|pad|>
1 <|bos|>
2 <|eos|>
3 <|system|>
4 <|user|>
5 <|assistant|>
```

## Training

### Base pretraining

The released base checkpoint was trained on revision-pinned **FineWeb-Edu** with [`configs/training/onehour_final.json`](configs/training/onehour_final.json):

- train tokens: **140,017,664**
- validation tokens: **262,144**
- global batch: 64 sequences × 512 tokens = **32,768 tokens/update**
- updates: **4,273**
- Muon peak LR: `0.02`
- AdamW peak LR: `3e-4`
- weight decay: `0.1`
- gradient clipping: `1.0`
- 2% warmup with cosine decay

Transformer Q/K/V/O and feed-forward gate/up/down matrices use Muon. Embeddings, normalization parameters, and the remaining parameters use AdamW. Parameters are FP32; matmul operands are BF16 with FP32 accumulation.

### Supervised fine-tuning

The release was instruction-tuned on revision-pinned **SmolTalk** with [`configs/training/onehour_sft.json`](configs/training/onehour_sft.json):

- selected examples: **512**
- train: **448**
- validation: **64**
- batch size: 8
- updates: **56**
- Muon LR: `0.003`
- AdamW LR: `5e-5`

Loss is applied only to assistant content and assistant EOS tokens.

See [`docs/TRAINING.md`](docs/TRAINING.md) and [`docs/MODAL_TRAINING.md`](docs/MODAL_TRAINING.md).

## Trained artifact

The trained package is checked in at:

```text
artifacts/nilemini-8m-situ/
├── model.safetensors
├── tokenizer.json
├── config.json
├── generation_config.json
├── reference_inputs.json
├── reference_outputs.safetensors
├── manifest.json
└── SHA256SUMS
```

The exported model contains **74 tensors**, **7,999,744 parameters**, and **30.52 MiB** of FP32 weights.

Verify it:

```bash
cd artifacts/nilemini-8m-situ
sha256sum -c SHA256SUMS
cd ../..
```

## JAX ↔ Rust parity

The independent Rust CPU engine was compared against the exported JAX reference over the complete `[1, 10, 8192]` logit tensor:

- logits compared: **81,920**
- max absolute error: **0.0528898239**
- mean absolute error: **0.0052069233**
- RMSE: **0.0074163827**
- cosine similarity: **0.9999967275**
- top-1 agreement: **10/10**
- final-position top-1 match: **true**
- accepted: **true**
- violations: **none**

Reproduce:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/nilemini-8m-situ
```

See [`docs/parity.md`](docs/parity.md).

## OpenAI-compatible API

The same Rust server exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nilemini-8m-situ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

`stream: true` returns SSE frames ending with `data: [DONE]`. The current server completes generation synchronously before emitting the buffered SSE body; this is wire-compatible SSE framing rather than token-time streaming.

See [`docs/API.md`](docs/API.md).

## Repository layout

```text
configs/model.json             Cross-language architecture contract
configs/training/              Training and SFT profiles
src/nilemini/config.py         Validated model/dataset/profile settings
src/nilemini/tokenizer.py      BPE preparation and tokenizer validation
src/nilemini/data.py           FineWeb-Edu split, tokenization, uint16 packing
src/nilemini/model.py          JAX/Flax NNX model
src/nilemini/optimizer.py      Muon + AdamW partition and schedules
src/nilemini/trainer.py        JIT training/evaluation and accumulation
src/nilemini/checkpoint.py     Orbax checkpoint/resume
src/nilemini/pretrain.py       Pretraining orchestration
src/nilemini/sft.py            SmolTalk preparation and SFT
src/nilemini/generation.py     Chat template and JAX reference generation
src/nilemini/export.py         FP32 SafeTensors export
src/nilemini/reference.py      Independent NumPy reference math
infra/modal_train.py           Modal L4 workflow
rust/engine/                   Independent Rust CPU inference engine
rust/server/                   OpenAI-compatible API + browser chat
artifacts/nilemini-8m-situ/    Final trained artifact
docs/DEMO.md                   Short demo/video walkthrough
```

## Verification

Python:

```bash
uv sync --locked --all-groups
uv run nilemini doctor
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

Rust:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Real trained-artifact parity:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/nilemini-8m-situ
```

Some expensive real-artifact integration tests are intentionally ignored in the default debug test suite and document their release-mode requirements. The explicit parity command above exercises the final trained artifact in release mode.

## Demo

A 3–5 minute reproducible presentation flow is in [`docs/DEMO.md`](docs/DEMO.md): artifact checksum → JAX/Rust parity → Rust server → browser chat → API request.

## Scope

NileMini is an end-to-end language-model systems implementation focused on:

1. training from a frozen architecture contract,
2. deterministic data preparation and checkpointing,
3. framework-independent SafeTensors export,
4. independent Rust inference,
5. numerical JAX ↔ Rust validation,
6. KV-cached autoregressive generation and sampling,
7. browser-based local chat, and
8. OpenAI-compatible serving.

Model quality is constrained by the **8M parameter scale** and the small SFT set. The repository emphasizes reproducibility, measured results, and systems correctness rather than frontier-model capability.

## License

Apache-2.0 for repository code and documentation. Dataset and model use must also comply with the applicable upstream dataset licenses and terms.
