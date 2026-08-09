# mypy: ignore-errors
"""Deterministic FineWeb-Edu token packing for pretraining."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from smalllm.config import (
    FINEWEB_CONFIG,
    FINEWEB_DATASET,
    FINEWEB_REVISION,
    MODEL,
    TrainingProfile,
    data_dir,
)
from smalllm.tokenizer import fineweb_documents, tokenizer_sha256


def document_split(document_id: str) -> str:
    """Assign documents to a deterministic 99/1 train/validation split."""

    value = int.from_bytes(
        hashlib.sha256(document_id.encode("utf-8")).digest()[:8],
        "big",
    )
    return "validation" if value % 10_000 < 100 else "train"


def paths_for(profile: TrainingProfile, root: Path | None = None) -> tuple[Path, Path, Path]:
    """Return train, validation, and manifest paths for a profile."""

    directory = data_dir(root) / profile.name
    directory.mkdir(parents=True, exist_ok=True)
    return (
        directory / "pretrain_train.bin",
        directory / "pretrain_validation.bin",
        directory / "pretrain_manifest.json",
    )


def prepare_pretraining_data(
    profile: TrainingProfile,
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    *,
    root: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Materialize exactly the configured train/validation token budgets."""

    train_path, validation_path, manifest_path = paths_for(profile, root)
    expected_manifest = {
        "dataset": FINEWEB_DATASET,
        "config": FINEWEB_CONFIG,
        "revision": FINEWEB_REVISION,
        "tokenizer_sha256": tokenizer_sha256(tokenizer_path),
        "train_tokens": profile.train_tokens,
        "validation_tokens": profile.validation_tokens,
        "dtype": "uint16",
    }
    if not force and train_path.exists() and validation_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing == expected_manifest:
            expected_train_bytes = 2 * (profile.train_tokens + 1)
            expected_validation_bytes = 2 * (profile.validation_tokens + 1)
            if (
                train_path.stat().st_size == expected_train_bytes
                and validation_path.stat().st_size == expected_validation_bytes
            ):
                return train_path, validation_path

    train_path.unlink(missing_ok=True)
    validation_path.unlink(missing_ok=True)
    targets = {
        "train": profile.train_tokens + 1,
        "validation": profile.validation_tokens + 1,
    }
    counts = {"train": 0, "validation": 0}
    eos_id = int(tokenizer.token_to_id("<|eos|>"))

    with train_path.open("wb") as train_file, validation_path.open("wb") as validation_file:
        files = {"train": train_file, "validation": validation_file}
        for document in fineweb_documents():
            split = document_split(document["id"])
            remaining = targets[split] - counts[split]
            if remaining <= 0:
                if all(counts[name] >= targets[name] for name in targets):
                    break
                continue
            token_ids = tokenizer.encode(document["text"], add_special_tokens=False).ids
            token_ids.append(eos_id)
            array = np.asarray(token_ids[:remaining], dtype="<u2")
            array.tofile(files[split])
            counts[split] += int(array.size)
            if all(counts[name] >= targets[name] for name in targets):
                break

    if counts != targets:
        raise RuntimeError(f"Incomplete pretraining data: {counts} != {targets}")
    manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
    return train_path, validation_path


class TokenBatcher:
    """Deterministic contiguous next-token batches over uint16 packed tokens."""

    def __init__(self, path: Path, target_tokens: int):
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.target_tokens = target_tokens

    def batch(self, update: int, sequences: int):
        """Return inputs, targets, and loss mask for one global update."""

        length = MODEL.context_length
        start_target = update * sequences * length
        valid_targets = min(
            max(0, self.target_tokens - start_target),
            sequences * length,
        )
        inputs = np.zeros((sequences, length), dtype=np.int32)
        targets = np.zeros_like(inputs)
        mask = np.zeros((sequences, length), dtype=np.float32)
        valid_sequences = math.ceil(valid_targets / length) if valid_targets else 0
        for row in range(valid_sequences):
            start = start_target + row * length
            take = min(length, self.target_tokens - start)
            segment = np.asarray(self.tokens[start : start + take + 1], dtype=np.int32)
            inputs[row, :take] = segment[:-1]
            targets[row, :take] = segment[1:]
            mask[row, :take] = 1.0
        return inputs, targets, mask

    def batches(self, sequences: int):
        """Iterate over the complete target-token budget."""

        updates = math.ceil(self.target_tokens / (sequences * MODEL.context_length))
        for update in range(updates):
            yield self.batch(update, sequences)


def main(argv=None) -> None:
    """Prepare packed data for one resolved profile using the frozen tokenizer."""

    import argparse

    from smalllm.config import load_profile, workspace_root
    from smalllm.tokenizer import ensure_tokenizer, load_frozen_tokenizer

    parser = argparse.ArgumentParser(description="Prepare SmallLM FineWeb-Edu token files")
    parser.add_argument("--profile", type=Path, default=Path("configs/training/pilot_l4.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = workspace_root()
    profile = load_profile(args.profile)
    frozen = Path("artifacts") / MODEL.model_name / "tokenizer.json"
    tokenizer_path = ensure_tokenizer(
        profile.tokenizer_bytes,
        root=root,
        frozen_source=frozen if frozen.exists() else None,
    )
    tokenizer = load_frozen_tokenizer(root)
    train_path, validation_path = prepare_pretraining_data(
        profile,
        tokenizer,
        tokenizer_path,
        root=root,
        force=args.force,
    )
    print(train_path)
    print(validation_path)


if __name__ == "__main__":
    main()
