# Model Card: NileMini-8M-SiTU

## Status

NileMini-8M-SiTU is an implemented **training + inference project whose final training run is still pending**. The repository's checked-in artifact is a smoke/parity artifact; it is not the final 1.6B-token + SFT model.

## Architecture

18-layer, 640-wide causal decoder; 1,664-wide SiTU-GLU FFN; 10 query / 2 KV heads; RoPE; RMSNorm; tied 8,192-token embeddings; 512-token context; exactly 7,999,744 parameters.

## Planned training

- FineWeb-Edu: 1.6B training tokens + 10M deterministic held-out validation tokens
- SmolTalk: 70k selected conversations = 68k train + 2k validation
- JAX/Flax NNX, Muon + AdamW, FP32 parameters / BF16 matmuls

The source revisions and processing rules are in [DATA_CARD.md](DATA_CARD.md).

## Intended use

This project demonstrates end-to-end small-language-model engineering: reproducible training, export, independent Rust inference, KV caching, sampling, cross-language parity, and an OpenAI-shaped local API. Any future public trained artifact needs a revision-specific evaluation and safety review.

## Current evidence

Engineering parity and server behavior are measured on the smoke artifact. No claim is made yet about final language quality, factuality, safety, bias, memorization, or benchmark performance because the full pretraining/SFT run has not happened.

## Release gate

A final release should include the full-run configuration/lineage, held-out metrics, SafeTensors/tokenizer checksums, final JAX↔Rust parity, example generations, performance measurements with hardware provenance, and safety/licensing review.
