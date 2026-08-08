"""Static checks that the real implementation lives in canonical package modules."""

from pathlib import Path

from nilemini.config import MODEL, load_profile, load_sft_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT_ROOT / "src" / "nilemini"
PROFILES = PROJECT_ROOT / "configs" / "training"


def test_canonical_training_modules_exist_without_duplicate_subpackage() -> None:
    required = {
        "config.py",
        "tokenizer.py",
        "data.py",
        "model.py",
        "optimizer.py",
        "checkpoint.py",
        "trainer.py",
        "pretrain.py",
        "sft.py",
        "generation.py",
        "evaluate.py",
        "export.py",
        "cli.py",
    }
    assert required <= {path.name for path in PACKAGE.iterdir() if path.is_file()}
    assert not (PACKAGE / "training").exists()


def test_frozen_model_and_run_budgets_are_exact() -> None:
    assert MODEL.expected_parameter_count == 7_999_744
    assert load_profile(PROFILES / "pilot_l4.json").train_tokens == 20_000_000
    full = load_profile(PROFILES / "full_l4.json")
    assert (full.train_tokens, full.validation_tokens) == (1_600_000_000, 10_000_000)
    sft = load_sft_profile(PROFILES / "sft_l4.json")
    assert (sft.selected_examples, sft.train_examples, sft.validation_examples) == (
        70_000,
        68_000,
        2_000,
    )


def test_legacy_todo_stage_configs_are_removed() -> None:
    assert not (PROJECT_ROOT / "configs" / "tokenizer.json").exists()
    assert not (PROJECT_ROOT / "configs" / "pretrain.json").exists()
    assert not (PROJECT_ROOT / "configs" / "sft.json").exists()
