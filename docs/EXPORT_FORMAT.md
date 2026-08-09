# Export format

The JAX exporter writes the files consumed by the Rust engine:

```text
artifacts/small-lm-8m/
├── model.safetensors
├── tokenizer.json
├── config.json
├── generation_config.json
├── reference_inputs.json
├── reference_outputs.safetensors
├── manifest.json
└── SHA256SUMS
```

The bundled model identifier is `small-lm-8m`.

All model tensors are exported as FP32. Linear kernels are transposed to `[out_features, in_features]` for the Rust runtime.

Layer tensor names follow:

```text
layers.N.attention_norm.weight
layers.N.q_proj.weight
layers.N.k_proj.weight
layers.N.v_proj.weight
layers.N.o_proj.weight
layers.N.ffn_norm.weight
layers.N.gate_proj.weight
layers.N.up_proj.weight
layers.N.down_proj.weight
```

Global tensors are:

```text
token_embedding.weight
final_norm.weight
```

The output projection reuses `token_embedding.weight`.

Artifact facts:

```text
parameters       7,999,744
tensors          74
FP32 weights     30.52 MiB
```

Verify checksums:

```bash
cd artifacts/small-lm-8m
sha256sum -c SHA256SUMS
```
