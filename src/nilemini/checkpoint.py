# mypy: ignore-errors
"""Orbax checkpointing for interruption-safe Modal training."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import orbax.checkpoint as ocp

_STEP_RE = re.compile(r"^step-(\d{8})$")


def step_path(directory: Path, step: int) -> Path:
    """Return the deterministic checkpoint directory for one optimizer step."""

    return directory / f"step-{step:08d}"


def latest_checkpoint(directory: Path) -> Path | None:
    """Return the numerically newest completed step checkpoint."""

    if not directory.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _STEP_RE.match(path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def save_training_checkpoint(
    directory: Path,
    payload,
    step: int,
    *,
    keep: int = 2,
) -> Path:
    """Save a complete training payload and retain only the newest step checkpoints."""

    if keep <= 0:
        raise ValueError("keep must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    path = step_path(directory, step)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(path, payload, force=True)
        checkpointer.wait_until_finished()
    candidates: list[tuple[int, Path]] = []
    for candidate in directory.iterdir():
        match = _STEP_RE.match(candidate.name)
        if match and candidate.is_dir():
            candidates.append((int(match.group(1)), candidate))
    for _, stale in sorted(candidates, reverse=True)[keep:]:
        shutil.rmtree(stale)
    return path


def restore_training_checkpoint(path: Path, target):
    """Restore a complete training payload into an exact target tree."""

    with ocp.StandardCheckpointer() as checkpointer:
        return checkpointer.restore(path, target)


def save_parameters(path: Path, params) -> Path:
    """Save parameter-only state for SFT or export handoff."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(path, params, force=True)
        checkpointer.wait_until_finished()
    return path


def restore_parameters(path: Path, target_params):
    """Restore parameter-only state into the current model structure."""

    if not path.exists():
        raise FileNotFoundError(f"Parameter checkpoint is missing: {path}")
    with ocp.StandardCheckpointer() as checkpointer:
        return checkpointer.restore(path, target_params)
