# mypy: ignore-errors
"""Deterministic FineWeb-Edu validation for a trained NileMini checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nilemini.checkpoint import restore_parameters
from nilemini.config import checkpoint_dir, load_profile, run_dir, workspace_root
from nilemini.data import TokenBatcher
from nilemini.model import initialize_model
from nilemini.pretrain import require_gpu, verify_profile_data
from nilemini.trainer import compile_evaluation_step, evaluate


def evaluate_checkpoint(
    profile,
    params_path: Path,
    *,
    root: Path,
    require_gpu_runtime: bool = True,
):
    """Evaluate one parameter checkpoint on the profile's frozen validation tokens."""

    if require_gpu_runtime:
        require_gpu()
    _, validation_path = verify_profile_data(profile, root=root)
    graphdef, params = initialize_model(profile.seed)
    params = restore_parameters(params_path, params)
    validation = TokenBatcher(validation_path, profile.validation_tokens)
    metrics = evaluate(
        params,
        validation,
        compile_evaluation_step(graphdef),
        sequences=profile.validation_sequences,
    )
    output = run_dir(root) / f"evaluation_{profile.name}.json"
    output.write_text(
        json.dumps(
            {"profile": profile.name, "params": str(params_path), "metrics": metrics},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained NileMini checkpoint")
    parser.add_argument("--profile", type=Path, default=Path("configs/training/pilot_l4.json"))
    parser.add_argument("--params", type=Path)
    args = parser.parse_args(argv)
    root = workspace_root()
    profile = load_profile(args.profile)
    params = args.params or checkpoint_dir(root) / "pretrain" / profile.name / "final-params"
    metrics = evaluate_checkpoint(profile, params, root=root)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
