# JAX–Rust Logit Parity

This report compares the independent Rust CPU engine against the exported JAX reference for the final trained `nilemini-8m-situ` v1.0.0 artifact.

The comparison covers every logit in shape `[1, 10, 8192]`: **81,920 FP32 values**.

## Measured result

- Accepted: **true**
- Maximum absolute error: `0.05288982391357422`
- Mean absolute error: `0.005206923344155711`
- Root-mean-square error: `0.007416382676593778`
- Cosine similarity: `0.9999967275464626`
- Top-1 agreement: **10/10 (1.0)**
- Final-position next-token match: **true**
- Violations: **none**

Rust and JAX top-1 token IDs were identical at every reference position:

```text
[267, 267, 18, 474, 84, 17, 18, 453, 17, 2451]
```

## Acceptance contract

- Maximum absolute error ≤ `0.5`
- Mean absolute error ≤ `0.05`
- RMSE ≤ `0.1`
- Cosine similarity ≥ `0.99`
- Top-1 agreement ≥ `0.8`
- Final-position top-1 token must match

Reproduce with:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/nilemini-8m-situ
```

The machine-readable result is stored in `docs/parity_report.json`.
