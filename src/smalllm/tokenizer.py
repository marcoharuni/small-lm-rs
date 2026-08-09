# mypy: ignore-errors
"""Frozen byte-level BPE tokenizer preparation and loading."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from smalllm.config import (
    FINEWEB_CONFIG,
    FINEWEB_DATASET,
    FINEWEB_REVISION,
    MODEL,
    SPECIAL_TOKENS,
    artifact_dir,
    data_dir,
    load_profile,
)


def optional_hf_token() -> str | None:
    """Return a Hugging Face token from the environment when available."""

    token = os.environ.get("HF_TOKEN", "").strip()
    return token or None


def fineweb_documents() -> Iterator[dict[str, str]]:
    """Stream non-empty FineWeb-Edu documents from the frozen revision."""

    dataset = load_dataset(
        FINEWEB_DATASET,
        name=FINEWEB_CONFIG,
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
        token=optional_hf_token(),
    )
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        document_id = str(
            row.get("id") or row.get("url") or hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
        yield {"id": document_id, "text": text}


def build_tokenizer_corpus(path: Path, byte_limit: int) -> None:
    """Write a deterministic-size tokenizer corpus from the pinned stream."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for document in fineweb_documents():
            piece = document["text"] + "\n"
            handle.write(piece)
            written += len(piece.encode("utf-8"))
            if written >= byte_limit:
                break
    if not path.exists() or path.stat().st_size < byte_limit:
        raise RuntimeError("Tokenizer corpus is incomplete")


def train_tokenizer(corpus_path: Path, output_path: Path, min_frequency: int = 2) -> Tokenizer:
    """Train one 8,192-token byte-level BPE tokenizer."""

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=MODEL.vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    """Validate vocabulary size, reserved IDs, and byte-level round trips."""

    if tokenizer.get_vocab_size() != MODEL.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary is {tokenizer.get_vocab_size()}, expected {MODEL.vocab_size}"
        )
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(token) != expected_id:
            raise RuntimeError(f"Special token {token} does not have ID {expected_id}")
    for sample in (
        "Hello, world!",
        "Kiswahili: Habari, dunia!",
        "https://example.com",
        'fn main() { println!("hello"); }',
        "Emoji 😀🚀",
        "Unicode: café 東京 العربية",
        "",
    ):
        encoded = tokenizer.encode(sample, add_special_tokens=False).ids
        if tokenizer.decode(encoded, skip_special_tokens=False) != sample:
            raise RuntimeError(f"Tokenizer failed round-trip for {sample!r}")


def tokenizer_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a tokenizer artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load and validate a serialized SmallLM tokenizer."""

    tokenizer_path = Path(path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer artifact is missing: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    validate_tokenizer(tokenizer)
    return tokenizer


def ensure_tokenizer(
    byte_limit: int,
    *,
    force: bool = False,
    root: Path | None = None,
    frozen_source: Path | None = None,
) -> Path:
    """Install the frozen tokenizer when available, otherwise train it exactly once."""

    output = artifact_dir(root) / "tokenizer.json"
    if frozen_source is not None and frozen_source.exists():
        source = frozen_source.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if force or not output.exists():
            if source != output.resolve():
                shutil.copy2(source, output)
        load_tokenizer(output)
        if tokenizer_sha256(output) != tokenizer_sha256(source):
            raise RuntimeError("workspace tokenizer does not match the frozen checked-in tokenizer")
        return output

    if output.exists() and not force:
        load_tokenizer(output)
        return output

    corpus = data_dir(root) / "tokenizer" / "fineweb_tokenizer.txt"
    build_tokenizer_corpus(corpus, byte_limit)
    tokenizer = train_tokenizer(corpus, output, min_frequency=2)
    if tokenizer.get_vocab_size() != MODEL.vocab_size:
        tokenizer = train_tokenizer(corpus, output, min_frequency=1)
    validate_tokenizer(tokenizer)
    return output


def load_frozen_tokenizer(root: Path | None = None) -> Tokenizer:
    """Load and validate the workspace's frozen tokenizer."""

    return load_tokenizer(artifact_dir(root) / "tokenizer.json")


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare or verify the frozen tokenizer from a resolved training profile."""

    parser = argparse.ArgumentParser(description="Prepare the SmallLM tokenizer")
    parser.add_argument("--profile", default="configs/training/smoke.json", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    frozen = Path("artifacts") / MODEL.model_name / "tokenizer.json"
    path = ensure_tokenizer(
        profile.tokenizer_bytes,
        force=args.force,
        frozen_source=frozen if frozen.exists() else None,
    )
    print(f"Tokenizer ready: {path} | sha256={tokenizer_sha256(path)}")


if __name__ == "__main__":
    main()
