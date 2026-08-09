# mypy: ignore-errors
"""Resumable smoke, pilot, and full pretraining orchestration."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import jax

from smalllm.checkpoint import (
    latest_checkpoint,
    restore_training_checkpoint,
    save_parameters,
    save_training_checkpoint,
)
from smalllm.config import (
    FINEWEB_CONFIG,
    FINEWEB_DATASET,
    FINEWEB_REVISION,
    MODEL,
    TrainingProfile,
    artifact_dir,
    checkpoint_dir,
    load_profile,
    run_dir,
    workspace_root,
)
from smalllm.data import TokenBatcher, paths_for, prepare_pretraining_data
from smalllm.model import initialize_model
from smalllm.optimizer import build_optimizer, optimizer_group_counts
from smalllm.tokenizer import (
    ensure_tokenizer,
    load_frozen_tokenizer,
    tokenizer_sha256,
)
from smalllm.trainer import (
    compile_apply,
    compile_evaluation_step,
    compile_loss_and_grad,
    evaluate,
    optimizer_update,
)

CheckpointHook = Callable[[], None]


def require_gpu() -> None:
    """Fail before paid training if JAX cannot see a GPU."""

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("SmallLM training requires a JAX GPU runtime")


def prepare_profile_data(
    profile: TrainingProfile,
    *,
    root: Path,
    force_tokenizer: bool = False,
    force_data: bool = False,
    frozen_tokenizer_source: Path | None = None,
) -> tuple[Path, Path]:
    """Ensure the frozen tokenizer and packed FineWeb-Edu data exist."""

    tokenizer_path = ensure_tokenizer(
        profile.tokenizer_bytes,
        force=force_tokenizer,
        root=root,
        frozen_source=frozen_tokenizer_source,
    )
    tokenizer = load_frozen_tokenizer(root)
    return prepare_pretraining_data(
        profile,
        tokenizer,
        tokenizer_path,
        root=root,
        force=force_data,
    )


def verify_profile_data(profile: TrainingProfile, *, root: Path) -> tuple[Path, Path]:
    """Verify prepared files before allocating a paid GPU."""

    train_path, validation_path, manifest_path = paths_for(profile, root)
    tokenizer_path = artifact_dir(root) / "tokenizer.json"
    missing = [
        path
        for path in (train_path, validation_path, manifest_path, tokenizer_path)
        if not path.exists()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"training data is not prepared; missing: {joined}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": FINEWEB_DATASET,
        "config": FINEWEB_CONFIG,
        "revision": FINEWEB_REVISION,
        "tokenizer_sha256": tokenizer_sha256(tokenizer_path),
        "train_tokens": profile.train_tokens,
        "validation_tokens": profile.validation_tokens,
        "dtype": "uint16",
    }
    if manifest != expected:
        raise RuntimeError("prepared-data manifest does not match the selected profile")
    if train_path.stat().st_size != 2 * (profile.train_tokens + 1):
        raise RuntimeError("prepared train token file has the wrong size")
    if validation_path.stat().st_size != 2 * (profile.validation_tokens + 1):
        raise RuntimeError("prepared validation token file has the wrong size")
    return train_path, validation_path


def _existing_best_validation(history_path: Path) -> float:
    best = float("inf")
    if not history_path.exists():
        return best
    for line in history_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "validation":
            best = min(best, float(record["loss"]))
    return best


def _write_run_manifest(profile: TrainingProfile, root: Path, params) -> None:
    manifest = {
        "model_name": MODEL.model_name,
        "parameter_count": MODEL.expected_parameter_count,
        "profile": profile.name,
        "train_tokens": profile.train_tokens,
        "validation_tokens": profile.validation_tokens,
        "dataset": {
            "name": FINEWEB_DATASET,
            "config": FINEWEB_CONFIG,
            "revision": FINEWEB_REVISION,
        },
        "tokenizer_sha256": tokenizer_sha256(artifact_dir(root) / "tokenizer.json"),
        "jax_version": jax.__version__,
        "flax_version": metadata.version("flax"),
        "optax_version": metadata.version("optax"),
        "orbax_checkpoint_version": metadata.version("orbax-checkpoint"),
        "devices": [str(device) for device in jax.devices()],
        "optimizer_parameter_counts": optimizer_group_counts(params),
    }
    (run_dir(root) / f"pretrain_{profile.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_pretraining(
    profile: TrainingProfile,
    *,
    root: Path,
    checkpoint_hook: CheckpointHook | None = None,
    require_gpu_runtime: bool = True,
    prepare_if_missing: bool = True,
    frozen_tokenizer_source: Path | None = None,
) -> Path:
    """Run or resume a complete pretraining profile and return final parameters."""

    if require_gpu_runtime:
        require_gpu()
    root.mkdir(parents=True, exist_ok=True)
    if prepare_if_missing:
        train_path, validation_path = prepare_profile_data(
            profile,
            root=root,
            frozen_tokenizer_source=frozen_tokenizer_source,
        )
    else:
        train_path, validation_path = verify_profile_data(profile, root=root)

    graphdef, params = initialize_model(profile.seed)
    optimizer = build_optimizer(params, profile.updates, profile.optimizer)
    optimizer_state = optimizer.init(params)
    loss_and_grad = compile_loss_and_grad(graphdef)
    apply_gradients = compile_apply(optimizer)
    evaluation_step = compile_evaluation_step(graphdef)
    train_batcher = TokenBatcher(train_path, profile.train_tokens)
    validation_batcher = TokenBatcher(validation_path, profile.validation_tokens)
    _write_run_manifest(profile, root, params)

    stage_dir = checkpoint_dir(root) / "pretrain" / profile.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    start_update = 0
    tokens_processed = 0
    latest = latest_checkpoint(stage_dir)
    if latest is not None:
        target = {
            "params": params,
            "optimizer": optimizer_state,
            "step": 0,
            "tokens_processed": 0,
        }
        restored = restore_training_checkpoint(latest, target)
        params = restored["params"]
        optimizer_state = restored["optimizer"]
        start_update = int(restored["step"])
        tokens_processed = int(restored["tokens_processed"])
        print(f"Resuming {profile.name} from update {start_update:,}")

    history_path = run_dir(root) / f"pretrain_{profile.name}.jsonl"
    best_validation_loss = _existing_best_validation(history_path)
    started = time.perf_counter()
    log_started = started
    log_tokens = tokens_processed

    for update in range(start_update, profile.updates):
        params, optimizer_state, metrics = optimizer_update(
            params,
            optimizer_state,
            train_batcher.batch(update, profile.global_sequences),
            loss_and_grad,
            apply_gradients,
            profile.microbatch_sequences,
        )
        tokens_processed += metrics["tokens"]
        completed = update + 1
        record = {
            "kind": "train",
            "update": completed,
            "tokens_processed": tokens_processed,
            **metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        if completed % profile.log_every_updates == 0 or completed == 1:
            now = time.perf_counter()
            tok_s = (tokens_processed - log_tokens) / max(now - log_started, 1e-9)
            elapsed = (now - started) / 60.0
            print(
                f"{profile.name} {completed:,}/{profile.updates:,} "
                f"loss={metrics['loss']:.4f} grad={metrics['gradient_norm']:.3f} "
                f"situ={metrics['situ_max']:.3f} tok/s={tok_s:,.0f} "
                f"tokens={tokens_processed:,} elapsed={elapsed:.1f}m"
            )
            log_started = now
            log_tokens = tokens_processed

        if completed % profile.validate_every_updates == 0 or completed == profile.updates:
            validation = evaluate(
                params,
                validation_batcher,
                evaluation_step,
                sequences=profile.validation_sequences,
            )
            best_validation_loss = min(best_validation_loss, validation["loss"])
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "kind": "validation",
                            "update": completed,
                            "tokens_processed": tokens_processed,
                            **validation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(f"validation loss={validation['loss']:.4f} ppl={validation['perplexity']:.2f}")

        if completed % profile.checkpoint_every_updates == 0 or completed == profile.updates:
            save_training_checkpoint(
                stage_dir,
                {
                    "params": params,
                    "optimizer": optimizer_state,
                    "step": int(completed),
                    "tokens_processed": int(tokens_processed),
                },
                completed,
            )
            if checkpoint_hook is not None:
                checkpoint_hook()

    if tokens_processed != profile.train_tokens:
        raise RuntimeError(
            f"Processed {tokens_processed:,} tokens, expected {profile.train_tokens:,}"
        )
    final_params = stage_dir / "final-params"
    save_parameters(final_params, params)
    if checkpoint_hook is not None:
        checkpoint_hook()
    summary = {
        "profile": profile.name,
        "updates": profile.updates,
        "tokens_processed": tokens_processed,
        "best_validation_loss": best_validation_loss,
        "wall_seconds_this_attempt": time.perf_counter() - started,
        "final_params": str(final_params),
    }
    (run_dir(root) / f"pretrain_{profile.name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_params


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run/resume SmallLM pretraining")
    parser.add_argument("--profile", default="configs/training/pilot_l4.json", type=Path)
    args = parser.parse_args(argv)
    output = run_pretraining(load_profile(args.profile), root=workspace_root())
    print(output)


if __name__ == "__main__":
    main()
