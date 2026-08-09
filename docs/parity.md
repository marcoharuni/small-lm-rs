# JAX / Rust parity

The Rust engine is checked against the exported JAX reference for the bundled model.

The reference fixture contains 81,920 FP32 logits with shape `[1, 10, 8192]`.

Measured result after matching the tied vocabulary projection to the JAX FP32 path:

```text
max absolute error    0.04658079147338867
mean absolute error   0.004962803585675224
RMSE                  0.007350166617063019
cosine similarity     0.99999678209871
top-1 agreement       10 / 10
final token match     true
```

Rust and JAX produced the same top-1 token IDs at all reference positions:

```text
267 267 18 474 84 17 18 453 17 2451
```

The transformer projection matrices use BF16-equivalent operands with FP32 accumulation. The final tied vocabulary projection uses FP32 operands and FP32 accumulation, matching the JAX reference implementation.

Run the check with:

```bash
cargo run --release -p smalllm-engine --example parity -- \
  artifacts/small-lm-8m
```

The machine-readable result is in `parity_report.json`.
