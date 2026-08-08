# Training

The JAX implementation lives in `src/nilemini/`. The package name is kept for compatibility with the existing training/export scripts; the repository itself is branded `small-lm-rs`.

## Base model

The bundled model has 7,999,744 parameters:

```text
layers              8
hidden size         256
FFN size            704
query / KV heads    4 / 2
head dimension      64
context             512
vocabulary          8192
```

The final pretraining profile is `configs/training/onehour_final.json`:

```text
training tokens     140,017,664
validation tokens   262,144
tokens/update       32,768
updates             4,273
validation loss     3.8659
perplexity          47.74
```

Parameters are stored in FP32. Matrix products use BF16 operands with FP32 accumulation.

## Optimizer split

Muon is used for 2-D transformer projection matrices. AdamW handles embeddings, normalization parameters, and the remaining parameters. RMSNorm weights use zero weight decay.

Base settings:

```text
Muon LR             0.02
AdamW LR            3e-4
weight decay        0.1
warmup              2%
gradient clip       1.0
schedule            cosine decay
```

## Instruction tuning

The SFT profile selects 512 examples after preprocessing: 448 training examples and 64 validation examples. It runs for 56 updates and reaches validation loss 2.6459.

Loss is applied only to assistant content and assistant EOS tokens.

SFT optimizer settings:

```text
Muon LR             0.003
AdamW LR            5e-5
weight decay        0.01
warmup              2%
gradient clip       1.0
```

## Checkpoints and export

Orbax checkpoints store parameters, optimizer state, completed update, and processed-token count. The exporter writes FP32 SafeTensors plus the tokenizer, model config, generation config, manifest, and JAX reference fixtures.

The checked-in artifact is at:

```text
artifacts/nilemini-8m-situ/
```

That directory keeps its original identifier because it is part of the frozen artifact contract.
