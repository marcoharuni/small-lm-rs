# Configuration

`model.json` is the frozen architecture contract shared by JAX and Rust.

Resolved executable training profiles are in `configs/training/`:

- `smoke.json` — 16,384 train tokens
- `pilot_l4.json` — 20M train tokens
- `full_l4.json` — 1.6B train + 10M validation tokens
- `sft_l4.json` — 70k selected SmolTalk examples

The old unresolved scaffold `tokenizer.json`, `pretrain.json`, and `sft.json` files have been removed so there is only one set of training decisions.
