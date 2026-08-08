# Export Format

The JAX exporter writes the artifact contract already consumed by `nilemini-engine`:

```text
nilemini-8m-situ/
├── model.safetensors
├── tokenizer.json
├── config.json
├── generation_config.json
├── reference_inputs.json
├── reference_outputs.safetensors
├── manifest.json
└── SHA256SUMS
```

All exported model tensors are FP32. JAX linear kernels are transposed to `[out_features, in_features]`.

For each layer `N`, tensor names are:

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

The tied output projection reuses `token_embedding.weight`; there is no duplicate LM-head tensor. The exporter asserts the logical tensor element count is exactly 7,999,744 and writes SHA-256 checksums for the package. `reference_inputs.json` plus `reference_outputs.safetensors` provide a JAX logit fixture for Rust parity validation.

The repository does not commit the large final `model.safetensors`; the trained artifact should be distributed separately after final training/export/parity.
