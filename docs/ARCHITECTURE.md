# Architecture

The bundled model is a small causal decoder. `configs/model.json` is the numeric contract shared by the JAX training code and the Rust runtime.

| Property | Value |
| --- | ---: |
| Parameters | 7,999,744 |
| Vocabulary | 8,192 |
| Context | 512 |
| Layers | 8 |
| Hidden size | 256 |
| FFN size | 704 |
| Query heads | 4 |
| KV heads | 2 |
| Head dimension | 64 |
| RoPE theta | 10,000 |
| RMSNorm epsilon | 1e-5 |

Each block is pre-normalized:

```text
x -> RMSNorm -> Q/K/V -> RoPE -> causal GQA -> output projection -> residual
  -> RMSNorm -> gated FFN -> down projection -> residual
```

The FFN uses SiTU-GLU with gate/up bounds of 4 and 25. Embeddings are tied to the output projection. The model has no bias or dropout.

JAX stores parameters in FP32 and uses BF16 operands for matrix products with FP32 accumulation. Exported linear weights use `[out_features, in_features]`, which is the layout consumed by the Rust engine.

## Runtime boundary

```text
JAX training
    |
SafeTensors + tokenizer + JSON config
    |
Rust inference engine
    |
KV cache + sampling
    |
HTTP API
```

The Rust runtime has no Python/JAX dependency during inference.
