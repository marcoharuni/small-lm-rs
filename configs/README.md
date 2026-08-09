# Configuration

`model.json` is the architecture contract shared by JAX and Rust.

Training profiles are in `configs/training/`:

- `smoke.json` — quick pipeline test
- `probe_l4.json` — throughput probe
- `final_l4.json` — released base training run
- `final_sft_l4.json` — released instruction-tuning run
- `pilot_l4.json`, `full_l4.json`, `sft_l4.json` — earlier development/planning profiles

The released weights were produced from `final_l4.json` followed by `final_sft_l4.json`.
