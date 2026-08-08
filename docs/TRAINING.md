# Training Guide

The authoritative training implementation lives directly in `src/nilemini/`. There is no duplicate notebook-only or `src/nilemini/training/` implementation.

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

The repository contains the reviewed 8,192-token tokenizer at `artifacts/nilemini-8m-situ/tokenizer.json`. Full/pilot training **reuses this exact file** rather than silently training another tokenizer.

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

## Pretraining profiles

`smoke.json` is the first real execution check: 16,384 train tokens and 4,096 validation tokens.

`pilot_l4.json` is the required 20M-token pilot before committing to the expensive run.

`full_l4.json` is the final base-model plan: 1.6B train tokens and 10M held-out validation tokens.

The L4 profiles use 32 sequences per global update, microbatches of 2, and context 512: 16,384 target tokens per full update. The final partial update is masked so the processed-token count equals the configured budget exactly.

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

Muon owns every 2D transformer Q/K/V/O and gate/up/down matrix. AdamW owns the remaining parameters; RMSNorm weights use zero weight decay. The code validates the exact optimizer partition before training.

Pretraining peaks:

- Muon LR `0.02`
- AdamW LR `3e-4`
- weight decay `0.1`
- 2% warmup + cosine to 10% of peak
- global gradient clipping `1.0`

SFT peaks:

- Muon LR `0.003`
- AdamW LR `5e-5`
- weight decay `0.01`

## Checkpoint/resume

Periodic Orbax checkpoints contain parameters, optimizer state, completed update, and processed-token count. The newest `step-XXXXXXXX` directory is restored automatically. Final parameter-only checkpoints form the handoff to SFT/export.

The training loop records JSONL metrics including loss, token accuracy, gradient norm, SiTU product maximum, processed tokens, validation loss/perplexity, and wall-time throughput logs.

## SFT

SmolTalk rows are restricted to plain message conversations without tool/image payloads, canonical-JSON deduplicated, and validated for one optional leading system message followed by alternating user/assistant turns ending in assistant. Sequences longer than the 512-token contract are rejected. Loss is applied only to assistant content and assistant EOS.

The selected 70,000 examples are split deterministically: 68,000 train / 2,000 validation. SFT also uses periodic Orbax checkpoint/resume.

## Execution order

```text
repository/CI checks
    ↓
Modal L4 smoke
    ↓
20M-token L4 pilot
    ↓
review loss, perplexity, gradient norm, SiTU max, tok/s, checkpoints
    ↓
1.6B-token full pretraining
    ↓
70k SFT
    ↓
FP32 SafeTensors export
    ↓
JAX ↔ Rust final parity + server smoke
```

Do not describe the model as fully trained until the 1.6B + SFT stages and final parity/export checks have actually completed.
