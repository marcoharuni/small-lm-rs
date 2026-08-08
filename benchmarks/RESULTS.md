# Benchmark / Parity Results

The repository has engineering measurements for the **smoke/parity artifact**, not final model-quality benchmarks.

Recorded JAX↔Rust logits over the frozen 10-token reference prompt:

- max absolute error: `0.00518465042`
- mean absolute error: `0.000966181391`
- RMSE: `0.00121245466`
- cosine similarity: `0.999999107917`
- top-1 agreement: `10/10`

A recorded local release CPU smoke measured 9-token prefill at `3880.774283 ms` and one cached decode token at `375.152134 ms/token`. This measurement is retained only as a development baseline because the hardware/provenance record is incomplete; it should not be presented as a portable performance claim.

No language-quality benchmark is reported until the 1.6B pretraining + SFT run is complete.
