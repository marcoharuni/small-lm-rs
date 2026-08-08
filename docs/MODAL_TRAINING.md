# Modal training

`infra/modal_train.py` is the cloud entry point for the JAX training workflow. It mounts a persistent Modal Volume for prepared data, checkpoints, logs, and exports.

## Setup

```bash
uvx --from 'modal==1.5.3' modal setup
```

FineWeb-Edu and SmolTalk are public datasets. `HF_TOKEN` is optional.

## Base model

Prepare and train the released base profile:

```bash
./scripts/modal_train.sh --stage prepare-onehour-final
./scripts/modal_train.sh --stage onehour-final
```

The profile contains:

```text
training tokens     140,017,664
validation tokens   262,144
updates             4,273
tokens/update       32,768
```

## Instruction tuning

```bash
./scripts/modal_train.sh --stage prepare-onehour-sft
./scripts/modal_train.sh --stage onehour-sft
```

The SFT profile uses 448 training examples and 64 validation examples for 56 updates.

## Export

```bash
./scripts/modal_train.sh --stage onehour-export
```

The exported artifact is copied into:

```text
artifacts/nilemini-8m-situ/
```

That directory retains the original model identifier for compatibility with its manifest and parity fixtures.

## Checkpointing

Periodic Orbax checkpoints are committed to the Modal Volume. Restarted jobs discover the newest numeric checkpoint and restore model parameters, optimizer state, update count, and processed-token count before continuing.
