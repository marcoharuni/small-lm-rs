# Data

## Pretraining

The released base model uses `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, pinned to revision:

```text
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
```

The training stream contains 140,017,664 tokens and the validation set contains 262,144 tokens.

Documents are split deterministically from a stable document identifier using SHA-256. Text is encoded with the frozen BPE tokenizer, EOS is appended per document, and token IDs are packed as little-endian `uint16` values.

## Instruction tuning

Instruction tuning uses `HuggingFaceTB/smoltalk`, config `smol-magpie-ultra`, pinned to revision:

```text
5feaf2fd3ffca7c237fc38d1861bc30365d48ffa
```

The configured preprocessing pipeline produced 2,163 valid examples. From that pool, 512 were selected deterministically: 448 for training and 64 for validation.

Rows containing tools or image payloads are excluded. Conversations are deduplicated and checked for valid role ordering. Loss is applied to assistant content and assistant EOS tokens only.

## Notes

Web and instruction datasets can contain inaccurate, biased, offensive, duplicated, copyrighted, or private material. The small model in this repository has not been evaluated as a production assistant and should not be treated as one.
