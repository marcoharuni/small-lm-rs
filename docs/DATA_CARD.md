# Data Card

## Frozen sources

| Stage | Source | Config | Revision | Planned budget |
| --- | --- | --- | --- | ---: |
| Pretraining | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 1.6B train + 10M validation tokens |
| SFT | `HuggingFaceTB/smoltalk` | `smol-magpie-ultra` | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | 70k selected examples |

The real full dataset materialization/training run is still pending. Small smoke data has been used to validate the engineering pipeline.

## Pretraining processing

FineWeb-Edu is streamed from the pinned revision. Empty text is rejected. A stable document identifier is taken from `id`, then URL, then SHA-256 of text. SHA-256 assigns documents deterministically to a 99% train / 1% validation split. Text is encoded with the frozen BPE, EOS is appended per document, and tokens are packed contiguously into little-endian uint16 files until each exact budget is reached.

Prepared-data manifests record dataset/config/revision, tokenizer SHA-256, exact token counts, and dtype. Existing files are reused only when their manifest and byte sizes match the selected profile.

## SFT processing

SmolTalk rows with tools or images are rejected. Message JSON is canonicalized and SHA-256 deduplicated. Conversations must contain at most one leading system message followed by alternating user/assistant messages and end in assistant. Over-context examples are rejected. Loss masking includes assistant content and assistant EOS only.

The first 70,000 valid unique examples from the pinned stream are split 68,000 train / 2,000 validation.

## Risks and release review

Web and instruction corpora can contain copyrighted, private, offensive, inaccurate, duplicated, or biased material. Source filtering does not eliminate these risks. Before publishing final trained weights, the exact run should receive licensing, privacy/memorization, safety/bias, and held-out quality review. Raw/prepared data remain outside Git.
