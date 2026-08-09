# JAX / Rust parity

The Rust engine is checked against the exported JAX reference for the bundled model.

The reference fixture contains 81,920 FP32 logits with shape `[1, 10, 8192]`.

Measured result:

```text
max absolute error    0.05288982391357422
mean absolute error   0.005206923344155711
RMSE                  0.007416382676593778
cosine similarity     0.9999967275464626
top-1 agreement       10 / 10
final token match     true
```

Rust and JAX produced the same top-1 token IDs at all reference positions:

```text
267 267 18 474 84 17 18 453 17 2451
```

Run the check with:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/small-lm-8m
```

The machine-readable result is in `parity_report.json`.
