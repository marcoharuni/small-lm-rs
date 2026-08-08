# Training Guide

The authoritative implementation lives in `src/nilemini/`.

## Released model

NileMini-8M-SiTU v1.0.0 uses the architecture in `configs/model.json`:

- 7,999,744 parameters
- 8 layers
- hidden size 256
- intermediate size 704
- 4 query heads / 2 KV heads
- head dimension 64
- context length 512
- vocabulary 8,192

## Data contracts

Pretraining is pinned to:

- dataset: `HuggingFaceFW/fineweb-edu`
- config: `sample-10BT`
- revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`

Instruction tuning is pinned to:

- dataset: `HuggingFaceTB/smoltalk`
- config: `smol-magpie-ultra`
- revision: `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`

FineWeb documents use a SHA-256 document-ID split: 99% train / 1% validation. Documents are BPE encoded, terminated with EOS, packed contiguously, and stored as little-endian `uint16` token IDs.

## Frozen tokenizer

The repository contains the reviewed 8,192-token tokenizer at `artifacts/nilemini-8m-situ/tokenizer.json`.

Reserved IDs:

```text
0 <|pad|>
1 <|bos|>
2 <|eos|>
3 <|system|>
4 <|user|>
5 <|assistant|>
```

`uv run nilemini doctor` validates the architecture, profiles, tokenizer digest, and special IDs without downloading data or requiring a GPU.

## Released pretraining profile

The final released base model uses `configs/training/onehour_final.json`:

- train tokens: **140,017,664**
- validation tokens: **262,144**
- global sequences/update: 64
- microbatch sequences: 8
- context: 512
- tokens/update: **32,768**
- updates: **4,273**
- validation every 500 updates
- checkpoint every 500 updates

Measured final base validation:

- loss: **3.8659**
- perplexity: **47.74**

The repository also retains smoke, probe, pilot, and larger planning profiles as engineering/reproducibility utilities. They are not claims that those larger planned budgets were used for the v1.0.0 weights.

## Model numerics

- FP32 parameters
- BF16 matmul operands
- FP32 preferred accumulation
- causal grouped-query attention
- interleaved RoPE with zero-based absolute positions
- pre-norm residual blocks
- SiTU gate: `4*tanh(gate/4)*sigmoid(gate)`
- SiTU up: `25*tanh(up/25)`
- tied embedding/output projection

## Optimizer

Muon owns every 2D transformer Q/K/V/O and gate/up/down matrix. AdamW owns the remaining parameters; RMSNorm weights use zero weight decay.

Released pretraining settings:

- Muon LR `0.02`
- AdamW LR `3e-4`
- weight decay `0.1`
- warmup fraction `0.02`
- cosine decay
- global gradient clipping `1.0`

Released SFT settings:

- Muon LR `0.003`
- AdamW LR `5e-5`
- weight decay `0.01`
- warmup fraction `0.02`
- gradient clipping `1.0`

## Checkpoint/resume

Periodic Orbax checkpoints contain parameters, optimizer state, completed update, and processed-token count. The newest `step-XXXXXXXX` directory is restored automatically. Final parameter-only checkpoints form the handoff to SFT/export.

The training loop records JSONL metrics including loss, token accuracy, gradient norm, SiTU product maximum, processed tokens, validation loss/perplexity, and wall time.

## Supervised fine-tuning

SmolTalk rows are restricted to plain message conversations without tool/image payloads, canonical-JSON deduplicated, and validated for one optional leading system message followed by alternating user/assistant turns ending in assistant. Sequences longer than the 512-token contract are rejected. Loss is applied only to assistant content and assistant EOS.

The released SFT profile is `configs/training/onehour_sft.json`:

- selected examples: **512**
- training examples: **448**
- validation examples: **64**
- batch size: 8
- updates: **56**
- final validation loss: **2.6459**

This is a small instruction-tuning pass intended to complete and validate the end-to-end chat pipeline. It should not be presented as a large-scale instruction-tuning corpus.

## Release execution path

```text
repository checks
    ↓
8M probe / throughput validation
    ↓
140,017,664-token FineWeb-Edu pretraining
    ↓
448-example SmolTalk SFT (+64 validation)
    ↓
FP32 SafeTensors export
    ↓
JAX ↔ Rust parity
    ↓
Rust chat/completions server smoke test
    ↓
v1.0.0
```

The final trained package lives at `artifacts/nilemini-8m-situ/`.
