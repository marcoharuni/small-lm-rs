# Modal L4 Training

`infra/modal_train.py` is the cloud entry point. It builds the pinned Python/JAX environment, mounts the canonical source and tokenizer, and uses a persistent Modal Volume at `/vol/nilemini` for prepared data, checkpoints, logs, and exports.

## Authentication

```bash
uvx --from 'modal==1.5.3' modal setup
```

FineWeb-Edu and SmolTalk are public datasets. `HF_TOKEN` is optional, though authentication can improve Hugging Face rate limits.

## v1.0.0 execution path

The released 8M model was produced with the one-hour profiles, not the legacy larger planning profiles.

### 1. Probe

```bash
./scripts/modal_train.sh --stage prepare-onehour-probe
./scripts/modal_train.sh --stage onehour-probe
```

The probe validated the exact 7,999,744-parameter architecture and measured healthy L4 throughput before the final run.

### 2. Final base preparation and pretraining

```bash
./scripts/modal_train.sh --stage prepare-onehour-final
./scripts/modal_train.sh --stage onehour-final
```

`onehour_final.json` contains the released training budget:

- 140,017,664 train tokens
- 262,144 validation tokens
- 4,273 updates
- 32,768 tokens/update

The final base checkpoint was written under:

```text
/vol/nilemini/checkpoints/pretrain/onehour-8m-final/final-params
```

### 3. SFT preparation and training

```bash
./scripts/modal_train.sh --stage prepare-onehour-sft
./scripts/modal_train.sh --stage onehour-sft
```

The released SFT profile contains:

- 512 selected examples
- 448 train examples
- 64 validation examples
- 56 updates

The final SFT parameters were written under:

```text
/vol/nilemini/checkpoints/sft/onehour-8m-sft/final-params
```

### 4. Export

```bash
./scripts/modal_train.sh --stage onehour-export
```

The exporter restores the final SFT parameters and writes the release package to:

```text
/vol/nilemini/exports/nilemini-8m-situ
```

The checked-in copy is `artifacts/nilemini-8m-situ/`.

## Durability

Training jobs use periodic Orbax checkpoints followed by Modal Volume commits. Restarted or retried jobs reload the Volume, discover the newest numeric checkpoint, restore optimizer and parameter state, and continue from the recorded update.

The detached launch path uses Modal `.spawn()` for long-running GPU stages so a local client disconnect does not cancel the remote training function.

## Legacy profiles

The repository may retain smoke, pilot, `full_l4.json`, and `sft_l4.json` profiles as development and reproducibility references. The v1.0.0 weights were **not** trained with the 1.6B-token / 70k-example planning profiles. The authoritative release profiles are:

```text
configs/training/onehour_final.json
configs/training/onehour_sft.json
```

## Release verification

After export, the final artifact is validated independently by the Rust engine:

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/nilemini-8m-situ
```

The v1.0.0 parity run was accepted with 10/10 top-1 agreement and no threshold violations.
