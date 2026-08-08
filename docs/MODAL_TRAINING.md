# Modal L4 Training

`infra/modal_train.py` is the cloud entry point. It builds a pinned Python 3.12/JAX environment, copies the canonical source and frozen tokenizer into the image, and mounts a persistent Modal Volume at `/vol/nilemini`.

## One-time authentication

```bash
uvx --from 'modal==1.5.3' modal setup
```

FineWeb-Edu and SmolTalk are public. `HF_TOKEN` is not required by this launcher; you may use your own Hugging Face authentication outside the repository if rate limits require it.

## Stages

Run the smoke first:

```bash
./scripts/modal_train.sh --stage smoke
```

That command prepares the 16K smoke data on CPU, commits it to the Volume, then allocates an L4 for the training run.

Next:

```bash
./scripts/modal_train.sh --stage pilot
```

`pilot` CPU-prepares the frozen 20M-token dataset, then runs/resumes the L4 job.

Only after reviewing the pilot metrics:

```bash
./scripts/modal_train.sh --stage full
```

The full stage CPU-prepares the 1.6B/10M packed token files before allocating the L4. The preparation and training data persist in the Volume.

After full pretraining:

```bash
./scripts/modal_train.sh --stage sft
./scripts/modal_train.sh --stage export
```

SFT preparation happens on CPU before its L4 job. Export restores the final SFT parameters and writes the SafeTensors package under `/vol/nilemini/exports/nilemini-8m-situ`.

Individual preparation stages are also available:

```bash
./scripts/modal_train.sh --stage prepare-smoke
./scripts/modal_train.sh --stage prepare-pilot
./scripts/modal_train.sh --stage prepare-full
./scripts/modal_train.sh --stage prepare-sft
```

## Durability

Training functions can run for up to 24 hours per attempt. Every configured periodic checkpoint is followed by a Volume commit. A restarted/retried job reloads the Volume, discovers the newest numeric Orbax checkpoint, restores optimizer + parameter state, and continues from the recorded update/token count.

The full profile checkpoints every 250 updates, so interruptions do not require restarting 1.6B tokens from zero.

## What to inspect after the pilot

Before authorizing `--stage full`, inspect:

- training loss trend
- validation loss/perplexity
- finite gradient norm
- bounded/finite SiTU maximum
- tokens/second and projected runtime
- checkpoint creation and a deliberate resume test
- L4 memory behavior / OOM status

If the pilot is unhealthy, change the profile/code, rerun smoke/pilot, and do not spend on the full run.
