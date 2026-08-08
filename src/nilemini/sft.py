# mypy: ignore-errors
"""Revision-pinned SmolTalk preparation and resumable supervised fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

from nilemini.checkpoint import (
    latest_checkpoint,
    restore_parameters,
    restore_training_checkpoint,
    save_parameters,
    save_training_checkpoint,
)
from nilemini.config import (
    MODEL,
    SMOLTALK_CONFIG,
    SMOLTALK_DATASET,
    SMOLTALK_REVISION,
    SPECIAL_TOKENS,
    SFTProfile,
    checkpoint_dir,
    data_dir,
    load_profile,
    load_sft_profile,
    run_dir,
    workspace_root,
)
from nilemini.model import initialize_model
from nilemini.optimizer import build_optimizer
from nilemini.tokenizer import load_frozen_tokenizer, optional_hf_token, tokenizer_sha256
from nilemini.trainer import (
    compile_apply,
    compile_evaluation_step,
    compile_loss_and_grad,
    optimizer_update,
)


def append_tokens(
    token_ids: list[int],
    loss_mask: list[int],
    values: Sequence[int],
    include_loss: bool,
) -> None:
    token_ids.extend(int(value) for value in values)
    loss_mask.extend([int(include_loss)] * len(values))


def format_conversation(tokenizer, messages: Sequence[Mapping[str, Any]]):
    """Format one SFT conversation and mask loss to assistant text/EOS only."""

    role_tokens = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
    }
    roles = [str(message.get("role", "")) for message in messages]
    dialogue = roles[1:] if roles and roles[0] == "system" else roles
    valid_order = (
        len(dialogue) >= 2
        and dialogue[-1] == "assistant"
        and all(
            role == ("user" if index % 2 == 0 else "assistant")
            for index, role in enumerate(dialogue)
        )
        and roles.count("system") <= 1
    )
    if not valid_order:
        return None

    bos_id = int(tokenizer.token_to_id("<|bos|>"))
    eos_id = int(tokenizer.token_to_id("<|eos|>"))
    pad_id = int(tokenizer.token_to_id("<|pad|>"))
    newline = tokenizer.encode("\n", add_special_tokens=False).ids
    token_ids = [bos_id]
    loss_mask = [0]

    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in role_tokens or not isinstance(content, str) or not content.strip():
            return None
        if any(marker in content for marker in SPECIAL_TOKENS):
            return None
        append_tokens(
            token_ids,
            loss_mask,
            [int(tokenizer.token_to_id(role_tokens[role]))],
            False,
        )
        append_tokens(token_ids, loss_mask, newline, False)
        append_tokens(
            token_ids,
            loss_mask,
            tokenizer.encode(content.strip(), add_special_tokens=False).ids,
            role == "assistant",
        )
        append_tokens(token_ids, loss_mask, [eos_id], role == "assistant")
        append_tokens(token_ids, loss_mask, newline, False)

    if len(token_ids) > MODEL.context_length + 1 or sum(loss_mask) == 0:
        return None
    padded_ids = np.full(MODEL.context_length + 1, pad_id, dtype=np.uint16)
    padded_mask = np.zeros(MODEL.context_length + 1, dtype=np.uint8)
    padded_ids[: len(token_ids)] = token_ids
    padded_mask[: len(loss_mask)] = loss_mask
    return padded_ids[:-1], padded_ids[1:], padded_mask[1:]


def _sft_manifest(profile: SFTProfile, *, root: Path) -> dict[str, object]:
    tokenizer_path = root / "artifacts" / MODEL.model_name / "tokenizer.json"
    return {
        "dataset": SMOLTALK_DATASET,
        "config": SMOLTALK_CONFIG,
        "revision": SMOLTALK_REVISION,
        "tokenizer_sha256": tokenizer_sha256(tokenizer_path),
        "selected_examples": profile.selected_examples,
        "train_examples": profile.train_examples,
        "validation_examples": profile.validation_examples,
    }


def _sft_required_files(directory: Path) -> list[Path]:
    return [
        directory / "train_inputs.npy",
        directory / "train_targets.npy",
        directory / "train_mask.npy",
        directory / "validation_inputs.npy",
        directory / "validation_targets.npy",
        directory / "validation_mask.npy",
    ]


def prepare_sft_data(profile: SFTProfile, *, root: Path, force: bool = False) -> None:
    """Select/deduplicate the pinned 70k SmolTalk conversation subset."""

    directory = data_dir(root) / profile.name
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "manifest.json"
    tokenizer = load_frozen_tokenizer(root)
    expected = _sft_manifest(profile, root=root)
    required = _sft_required_files(directory)
    if (
        marker.exists()
        and not force
        and all(path.exists() for path in required)
        and json.loads(marker.read_text(encoding="utf-8")) == expected
    ):
        return

    dataset = load_dataset(
        SMOLTALK_DATASET,
        name=SMOLTALK_CONFIG,
        split="train",
        streaming=True,
        revision=SMOLTALK_REVISION,
        token=optional_hf_token(),
    )
    examples = []
    seen: set[bytes] = set()
    for row in dataset:
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or row.get("tools")
            or row.get("images")
            or row.get("image")
        ):
            continue
        canonical = json.dumps(
            messages,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        if digest in seen:
            continue
        formatted = format_conversation(tokenizer, messages)
        if formatted is None:
            continue
        seen.add(digest)
        examples.append(formatted)
        if len(examples) == profile.selected_examples:
            break
    if len(examples) != profile.selected_examples:
        raise RuntimeError(f"Only {len(examples)} valid SFT examples found")

    def save_split(name: str, rows) -> None:
        np.save(directory / f"{name}_inputs.npy", np.stack([row[0] for row in rows]))
        np.save(directory / f"{name}_targets.npy", np.stack([row[1] for row in rows]))
        np.save(directory / f"{name}_mask.npy", np.stack([row[2] for row in rows]))

    save_split("train", examples[: profile.train_examples])
    save_split("validation", examples[profile.train_examples :])
    marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


def verify_sft_data(profile: SFTProfile, *, root: Path) -> Path:
    """Verify that the CPU-prepared SFT arrays match the frozen dataset/tokenizer contract."""

    directory = data_dir(root) / profile.name
    marker = directory / "manifest.json"
    missing = [path for path in [marker, *_sft_required_files(directory)] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SFT data is not prepared; missing: " + ", ".join(str(path) for path in missing)
        )
    if json.loads(marker.read_text(encoding="utf-8")) != _sft_manifest(profile, root=root):
        raise RuntimeError("prepared SFT manifest does not match the selected profile")
    train_inputs = np.load(directory / "train_inputs.npy", mmap_mode="r")
    validation_inputs = np.load(directory / "validation_inputs.npy", mmap_mode="r")
    if train_inputs.shape != (profile.train_examples, MODEL.context_length):
        raise RuntimeError("prepared SFT train inputs have the wrong shape")
    if validation_inputs.shape != (profile.validation_examples, MODEL.context_length):
        raise RuntimeError("prepared SFT validation inputs have the wrong shape")
    return directory


class SFTBatcher:
    """Deterministic fixed-shape SFT batches."""

    def __init__(self, directory: Path, split: str, seed: int):
        self.inputs = np.load(directory / f"{split}_inputs.npy")
        self.targets = np.load(directory / f"{split}_targets.npy")
        self.mask = np.load(directory / f"{split}_mask.npy")
        self.seed = seed

    def epoch(self, batch_size: int, *, start_update: int = 0):
        order = np.random.default_rng(self.seed).permutation(len(self.inputs))
        for update, start in enumerate(range(0, len(order), batch_size)):
            if update < start_update:
                continue
            indices = order[start : start + batch_size]
            size = len(indices)
            inputs = np.zeros((batch_size, MODEL.context_length), dtype=np.int32)
            targets = np.zeros_like(inputs)
            mask = np.zeros_like(inputs, dtype=np.float32)
            inputs[:size] = self.inputs[indices]
            targets[:size] = self.targets[indices]
            mask[:size] = self.mask[indices]
            yield inputs, targets, mask

    def batches(self, batch_size: int):
        for start in range(0, len(self.inputs), batch_size):
            end = min(start + batch_size, len(self.inputs))
            yield (
                self.inputs[start:end].astype(np.int32),
                self.targets[start:end].astype(np.int32),
                self.mask[start:end].astype(np.float32),
            )


def evaluate_sft(params, batcher: SFTBatcher, evaluation_step, batch_size: int):
    """Evaluate assistant-only SFT loss, perplexity, and token accuracy."""

    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_tokens = jnp.asarray(0.0, dtype=jnp.float32)
    total_correct = jnp.asarray(0.0, dtype=jnp.float32)
    for inputs, targets, mask in batcher.batches(batch_size):
        loss_sum, valid_tokens, correct = evaluation_step(
            params,
            jnp.asarray(inputs),
            jnp.asarray(targets),
            jnp.asarray(mask),
        )
        total_loss = total_loss + loss_sum
        total_tokens = total_tokens + valid_tokens
        total_correct = total_correct + correct
    token_count = float(total_tokens)
    if token_count <= 0.0:
        raise RuntimeError("SFT validation contains no assistant targets")
    mean_loss = float(total_loss) / token_count
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "accuracy": float(total_correct) / token_count,
    }


def run_sft(
    profile: SFTProfile,
    *,
    root: Path,
    base_params_path: Path,
    checkpoint_hook=None,
    prepare_if_missing: bool = True,
) -> Path:
    """Run/resume one deterministic SFT epoch from completed pretraining parameters."""

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("NileMini SFT requires a JAX GPU runtime")
    if prepare_if_missing:
        prepare_sft_data(profile, root=root)
        if checkpoint_hook is not None:
            checkpoint_hook()
    else:
        verify_sft_data(profile, root=root)
    graphdef, params = initialize_model(profile.seed)
    params = restore_parameters(base_params_path, params)
    optimizer = build_optimizer(params, profile.updates, profile.optimizer)
    optimizer_state = optimizer.init(params)
    loss_and_grad = compile_loss_and_grad(graphdef)
    apply_gradients = compile_apply(optimizer)
    evaluation_step = compile_evaluation_step(graphdef)
    directory = data_dir(root) / profile.name
    train = SFTBatcher(directory, "train", profile.seed)
    validation = SFTBatcher(directory, "validation", profile.seed)

    stage_dir = checkpoint_dir(root) / "sft" / profile.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    start_update = 0
    latest = latest_checkpoint(stage_dir)
    if latest is not None:
        target = {"params": params, "optimizer": optimizer_state, "step": 0}
        restored = restore_training_checkpoint(latest, target)
        params = restored["params"]
        optimizer_state = restored["optimizer"]
        start_update = int(restored["step"])
        print(f"Resuming {profile.name} from update {start_update:,}")

    history_path = run_dir(root) / f"sft_{profile.name}.jsonl"
    started = time.perf_counter()
    for completed, batch in enumerate(
        train.epoch(profile.batch_size, start_update=start_update),
        start=start_update + 1,
    ):
        params, optimizer_state, metrics = optimizer_update(
            params,
            optimizer_state,
            batch,
            loss_and_grad,
            apply_gradients,
            profile.microbatch_sequences,
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"kind": "train", "update": completed, **metrics}, sort_keys=True) + "\n"
            )
        if completed % profile.log_every_updates == 0 or completed == 1:
            print(
                f"{profile.name} {completed:,}/{profile.updates:,} "
                f"loss={metrics['loss']:.4f} grad={metrics['gradient_norm']:.3f}"
            )
        if completed % profile.validate_every_updates == 0 or completed == profile.updates:
            validation_metrics = evaluate_sft(
                params,
                validation,
                evaluation_step,
                profile.batch_size,
            )
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"kind": "validation", "update": completed, **validation_metrics},
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(
                f"SFT validation loss={validation_metrics['loss']:.4f} "
                f"ppl={validation_metrics['perplexity']:.2f}"
            )
        if completed % profile.checkpoint_every_updates == 0 or completed == profile.updates:
            save_training_checkpoint(
                stage_dir,
                {"params": params, "optimizer": optimizer_state, "step": int(completed)},
                completed,
            )
            if checkpoint_hook is not None:
                checkpoint_hook()

    final_metrics = evaluate_sft(params, validation, evaluation_step, profile.batch_size)
    output = stage_dir / "final-params"
    save_parameters(output, params)
    (run_dir(root) / f"sft_{profile.name}_summary.json").write_text(
        json.dumps(
            {
                "profile": profile.name,
                "updates": profile.updates,
                "validation": final_metrics,
                "wall_seconds_this_attempt": time.perf_counter() - started,
                "final_params": str(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if checkpoint_hook is not None:
        checkpoint_hook()
    print(f"SFT validation loss={final_metrics['loss']:.4f}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run/resume NileMini SFT")
    parser.add_argument("--profile", default="configs/training/sft_l4.json", type=Path)
    parser.add_argument("--base-profile", default="configs/training/full_l4.json", type=Path)
    args = parser.parse_args(argv)
    root = workspace_root()
    profile = load_sft_profile(args.profile)
    base = load_profile(args.base_profile)
    base_params = checkpoint_dir(root) / "pretrain" / base.name / "final-params"
    print(run_sft(profile, root=root, base_params_path=base_params))


if __name__ == "__main__":
    main()
