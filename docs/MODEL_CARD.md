# Bundled 8M model

The repository includes a trained 7,999,744-parameter decoder model used to develop and validate the Rust inference path.

## Architecture

- 8 transformer layers
- hidden size 256
- FFN size 704
- 4 query heads / 2 KV heads
- 8,192-token vocabulary
- 512-token context
- RMSNorm, RoPE, grouped-query attention, bounded gated FFN
- tied token embeddings

## Training

Base pretraining used 140,017,664 FineWeb-Edu training tokens and 262,144 validation tokens. The final base validation loss was 3.8659 (perplexity 47.74).

The instruction-tuning pass used 448 SmolTalk training examples and 64 validation examples, with final validation loss 2.6459.

See `TRAINING.md` and `DATA_CARD.md` for data and optimizer details.

## Intended use

This model is mainly a test bed for training/export/runtime work. Its small parameter count and limited instruction-tuning set make it unsuitable as a strong general-purpose assistant.

## Runtime validation

The exported JAX reference and Rust implementation agree on all 10 reference top-1 predictions. The complete parity report is in `parity.md` and `parity_report.json`.

The bundled model identifier is `small-lm-8m`.
