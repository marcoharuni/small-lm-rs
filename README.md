# NileMini-8M-SiTU

[![CI](https://github.com/marcoharuni/nilemini-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/marcoharuni/nilemini-rs/actions/workflows/ci.yml)

NileMini is an **7,999,744-parameter** decoder-only language-model project: JAX/Flax NNX for training and an independent Rust CPU engine/server for inference.

## Current status

The engineering path is implemented: tokenizer contract, JAX model, Muon/AdamW training loop, Orbax checkpoint/resume, FP32 SafeTensors export, Rust weight loading, JAX↔Rust parity, KV-cached generation, sampling, and OpenAI-shaped HTTP endpoints.

**The final 1.6B-token pretraining run and 70k-example SFT run have not been completed yet.** The checked-in model package is a smoke/parity artifact, not the final trained release.

## Frozen architecture

- vocabulary: 8,192 byte-level BPE tokens
- context: 512 tokens
- layers: 18
- hidden size: 640
- feed-forward size: 1,664
- attention: 4 query heads / 2 KV heads / head dimension 64
- RMSNorm epsilon: `1e-5`
- interleaved RoPE theta: `10000`
- SiTU-GLU: beta gate `4`, beta up `25`
- tied embeddings, no bias, no dropout
- exact parameters: **7,999,744**

The six reserved token IDs are frozen as pad=0, bos=1, eos=2, system=3, user=4, assistant=5.

## Source of truth

```text
configs/model.json       Cross-language architecture contract
configs/training/        Smoke, 20M pilot, 1.6B full, and 70k SFT profiles
src/nilemini/config.py   Validated architecture/dataset/profile settings
src/nilemini/tokenizer.py Frozen BPE preparation and validation
src/nilemini/data.py     FineWeb-Edu deterministic split and uint16 packing
src/nilemini/model.py    JAX/Flax NNX decoder
src/nilemini/optimizer.py Muon + AdamW partition and schedules
src/nilemini/trainer.py  JIT loss, microbatch accumulation, evaluation
src/nilemini/checkpoint.py Orbax persistence and resume
src/nilemini/pretrain.py Pretraining orchestration
src/nilemini/sft.py      SmolTalk preparation and instruction tuning
src/nilemini/generation.py Frozen chat prompt and JAX reference generation
src/nilemini/export.py   FP32 SafeTensors/Rust export
src/nilemini/reference.py Independent NumPy reference math
infra/modal_train.py     Persistent Modal L4 workflow
rust/engine/             Independent Rust inference engine
rust/server/             OpenAI-compatible HTTP/SSE server
```

The notebook is now a thin, readable entry point. It is **not** a second copy of the training implementation.

## Frozen training plan

| Profile | Budget |
| --- | ---: |
| `smoke.json` | 16,384 train / 4,096 validation tokens |
| `pilot_l4.json` | 20,000,000 train / 1,000,000 validation tokens |
| `full_l4.json` | 1,600,000,000 train / 10,000,000 validation tokens |
| `sft_l4.json` | 70,000 selected = 68,000 train + 2,000 validation examples |

Pretraining uses revision-pinned FineWeb-Edu. SFT uses revision-pinned SmolTalk. Transformer Q/K/V/O and gate/up/down matrices use Muon; embeddings and normalization parameters use AdamW. Parameters are FP32, matmul operands are BF16, and accumulation is FP32.

## Verify the repository

```bash
uv sync --locked --all-groups
uv run nilemini doctor
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Modal L4: smoke → pilot → full → SFT → export

Authenticate Modal once:

```bash
uvx --from 'modal==1.5.3' modal setup
```

Then run the tiny paid-GPU smoke first:

```bash
./scripts/modal_train.sh --stage smoke
```

After its loss/checkpoint behavior is healthy:

```bash
./scripts/modal_train.sh --stage pilot
```

Only after reviewing the 20M pilot:

```bash
./scripts/modal_train.sh --stage full
./scripts/modal_train.sh --stage sft
./scripts/modal_train.sh --stage export
```

Each stage prepares data on CPU before allocating the L4. Training checkpoints are committed to the persistent Modal Volume and automatically resumed after interruption. Public Hugging Face datasets work anonymously; setting `HF_TOKEN` is optional outside the checked-in code.

See [Training](docs/TRAINING.md) and [Modal Training](docs/MODAL_TRAINING.md).

## Rust server

With a complete artifact directory containing `model.safetensors`:

```bash
cargo run --release --package nilemini-server -- \
  --model-dir artifacts/nilemini-8m-situ \
  --host 127.0.0.1 \
  --port 8080
```

The server exposes `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`. See [API](docs/API.md).

## Artifact policy

Raw datasets, packed training data, Orbax checkpoints, run logs, secrets, and final trained weights stay out of Git. Final model files should be published separately with immutable checksums after the real training/SFT/export/parity sequence is complete.

## License

Apache-2.0 for repository code/documentation. Dataset and model-release licensing must follow the upstream sources and the final release review.
