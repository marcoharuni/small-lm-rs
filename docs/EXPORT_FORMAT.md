# Export Format

The JAX exporter writes the artifact contract consumed by `nilemini-engine`:

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

The final v1.0.0 package is checked in at `artifacts/nilemini-8m-situ/`.

All exported model tensors are FP32. JAX linear kernels are transposed to `[out_features, in_features]` for the Rust engine.

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

The tied output projection reuses `token_embedding.weight`; there is no duplicate LM-head tensor.

## Release artifact facts

- logical tensor element count: **7,999,744**
- exported tensors: **74**
- FP32 weight size: **30.52 MiB**
- model name: `nilemini-8m-situ`

`SHA256SUMS` and `manifest.json` provide integrity metadata. `reference_inputs.json` and `reference_outputs.safetensors` provide the frozen JAX fixture used for independent Rust parity validation.

Verify locally:

```bash
cd artifacts/nilemini-8m-situ
sha256sum -c SHA256SUMS
```

Run the cross-language parity check:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/nilemini-8m-situ
```
