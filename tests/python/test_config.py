"""Tests for the frozen architecture and resolved training profiles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nilemini.config import ConfigurationError, load_model_config, load_profile, load_sft_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
PROFILES = CONFIG_DIR / "training"


def test_model_config_matches_frozen_architecture() -> None:
    config = load_model_config(CONFIG_DIR / "model.json")
    assert config.model_name == "nilemini-8m-situ"
    assert config.vocab_size == 8192
    assert config.context_length == 512
    assert config.num_layers == 8
    assert config.hidden_size == 256
    assert config.intermediate_size == 704
    assert config.num_query_heads == 4
    assert config.num_key_value_heads == 2
    assert config.head_dimension == 64
    assert config.expected_parameter_count == 7_999_744
    assert config.calculated_parameter_count == config.expected_parameter_count


def test_parameter_count_mismatch_fails_explicitly() -> None:
    config = load_model_config(CONFIG_DIR / "model.json")
    with pytest.raises(ConfigurationError, match="expected_parameter_count does not match"):
        replace(config, expected_parameter_count=1)


def test_resolved_training_budgets_are_exact() -> None:
    smoke = load_profile(PROFILES / "smoke.json")
    pilot = load_profile(PROFILES / "pilot_l4.json")
    full = load_profile(PROFILES / "full_l4.json")
    sft = load_sft_profile(PROFILES / "sft_l4.json")
    assert smoke.train_tokens == 16_384
    assert pilot.train_tokens == 20_000_000
    assert full.train_tokens == 1_600_000_000
    assert full.validation_tokens == 10_000_000
    assert sft.selected_examples == 70_000
    assert (sft.train_examples, sft.validation_examples) == (68_000, 2_000)


def test_invalid_json_reports_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid JSON"):
        load_model_config(path)
