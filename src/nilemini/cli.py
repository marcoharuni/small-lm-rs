"""Unified command line for preparing, training, evaluating, and exporting NileMini."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nilemini.config import (
    MODEL,
    checkpoint_dir,
    load_profile,
    load_sft_profile,
    workspace_root,
)


def _doctor() -> None:
    from nilemini.tokenizer import load_tokenizer, validate_tokenizer

    profiles = [
        load_profile("configs/training/smoke.json"),
        load_profile("configs/training/pilot_l4.json"),
        load_profile("configs/training/full_l4.json"),
    ]
    sft = load_sft_profile("configs/training/sft_l4.json")
    tokenizer_path = Path("artifacts") / MODEL.model_name / "tokenizer.json"
    validate_tokenizer(load_tokenizer(tokenizer_path))
    print(
        json.dumps(
            {
                "model": MODEL.model_name,
                "parameters": MODEL.expected_parameter_count,
                "pretraining_profiles": {
                    profile.name: {"tokens": profile.train_tokens, "updates": profile.updates}
                    for profile in profiles
                },
                "sft": {
                    "profile": sft.name,
                    "selected": sft.selected_examples,
                    "train": sft.train_examples,
                    "validation": sft.validation_examples,
                },
                "tokenizer": str(tokenizer_path),
                "status": "ready",
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nilemini", description="NileMini training toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="validate frozen architecture, profiles, and tokenizer")

    prepare = sub.add_parser("prepare", help="prepare tokenizer and packed pretraining data")
    prepare.add_argument("--profile", type=Path, default=Path("configs/training/pilot_l4.json"))
    prepare.add_argument("--force-data", action="store_true")

    pretrain = sub.add_parser("pretrain", help="run or resume pretraining")
    pretrain.add_argument("--profile", type=Path, default=Path("configs/training/pilot_l4.json"))

    sft = sub.add_parser("sft", help="run or resume instruction tuning")
    sft.add_argument("--profile", type=Path, default=Path("configs/training/sft_l4.json"))
    sft.add_argument("--base-profile", type=Path, default=Path("configs/training/full_l4.json"))

    evaluate = sub.add_parser("evaluate", help="evaluate a pretraining checkpoint")
    evaluate.add_argument("--profile", type=Path, default=Path("configs/training/pilot_l4.json"))
    evaluate.add_argument("--params", type=Path)

    export = sub.add_parser("export", help="export final parameters for Rust")
    export.add_argument("--profile", type=Path, default=Path("configs/training/sft_l4.json"))
    export.add_argument("--params", type=Path)
    export.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        _doctor()
        return

    root = workspace_root()
    frozen = Path("artifacts") / MODEL.model_name / "tokenizer.json"
    if args.command == "prepare":
        from nilemini.pretrain import prepare_profile_data

        profile = load_profile(args.profile)
        train_path, validation_path = prepare_profile_data(
            profile,
            root=root,
            force_data=args.force_data,
            frozen_tokenizer_source=frozen,
        )
        print(train_path)
        print(validation_path)
        return
    if args.command == "pretrain":
        from nilemini.pretrain import run_pretraining

        print(
            run_pretraining(
                load_profile(args.profile),
                root=root,
                frozen_tokenizer_source=frozen,
            )
        )
        return
    if args.command == "sft":
        from nilemini.sft import run_sft

        sft_profile = load_sft_profile(args.profile)
        base_profile = load_profile(args.base_profile)
        base_params = checkpoint_dir(root) / "pretrain" / base_profile.name / "final-params"
        print(run_sft(sft_profile, root=root, base_params_path=base_params))
        return
    if args.command == "evaluate":
        from nilemini.evaluate import evaluate_checkpoint

        eval_profile = load_profile(args.profile)
        params = (
            args.params or checkpoint_dir(root) / "pretrain" / eval_profile.name / "final-params"
        )
        print(json.dumps(evaluate_checkpoint(eval_profile, params, root=root), indent=2))
        return
    if args.command == "export":
        from nilemini.export import export_model

        export_profile = load_sft_profile(args.profile)
        params = args.params or checkpoint_dir(root) / "sft" / export_profile.name / "final-params"
        print(export_model(params, root=root, output_dir=args.output))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
