# Configuration

`model.json` is the architecture contract shared by JAX and Rust.

Training profiles are in `configs/training/`:

- `smoke.json` — quick pipeline test
- `onehour_probe.json` — throughput probe
- `onehour_final.json` — released base training run
- `onehour_sft.json` — released instruction-tuning run
- `pilot_l4.json`, `full_l4.json`, `sft_l4.json` — earlier development/planning profiles

The released weights were produced from `onehour_final.json` followed by `onehour_sft.json`.
