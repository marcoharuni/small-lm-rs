# JAX–Rust Logit Parity

This report compares the Rust CPU engine against the exported JAX reference
for the smoke artifact. The comparison covers every logit in shape
`[1, 10, 8192]`
(81,920 FP32 values).

## Measured result

- Accepted: **true**
- Maximum absolute error: `0.00518465042`
- Mean absolute error: `0.000966181391`
- Root-mean-square error: `0.00121245466`
- Cosine similarity: `0.999999107917`
- Top-1 agreement: `10/10`
- Final-position next-token match: `true`
- Rust final token: `17`
- JAX final token: `17`

## Baseline acceptance contract

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

The JSON result is stored in `docs/parity_report.json`.
