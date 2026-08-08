# Architecture

`configs/model.json` is the cross-language numeric contract for NileMini-8M-SiTU.

| Property | Value |
| --- | ---: |
| Vocabulary | 8,192 |
| Context | 512 |
| Layers | 18 |
| Hidden width | 640 |
| FFN width | 1,664 |
| Query heads | 10 |
| KV heads | 2 |
| Head dimension | 64 |
| RoPE theta | 10,000 |
| RMSNorm epsilon | 1e-5 |
| SiTU beta gate/up | 4 / 25 |
| Parameter count | 7,999,744 |

The model is a causal pre-normalized decoder with GQA, interleaved RoPE, RMSNorm, SiTU-GLU, residual connections, and tied embeddings. It has no bias or dropout.

## Exact block

```text
x
├─ RMSNorm → Q/K/V → RoPE → causal GQA → O → + residual
└─ RMSNorm → gate/up → SiTU-GLU product → down → + residual
```

JAX keeps parameters FP32, casts matmul operands to BF16, requests FP32 accumulation, computes attention scores/softmax and normalization in FP32, and exports linear kernels transposed to `[out_features, in_features]` for Rust.

## System boundary

```text
frozen tokenizer + revision-pinned datasets
                ↓
         JAX train / SFT
                ↓
          Orbax checkpoints
                ↓
      FP32 SafeTensors export
                ↓
    independent Rust CPU engine
                ↓
       KV-cache + sampling
                ↓
      OpenAI-shaped HTTP API
```

The Rust engine does not call Python/JAX. SafeTensors plus JSON/tokenizer artifacts form the interchange boundary.

## Parity

The smoke artifact has already been used for end-to-end JAX↔Rust logit parity, cached-prefill/decode checks, and token-generation checks. Final trained weights must repeat the same parity gate after export; passing smoke parity does not prove the future trained checkpoint until that final check is run.
