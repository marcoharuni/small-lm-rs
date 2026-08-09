"""Frozen prompt parity fixture shared by JAX export and the Rust tokenizer tests."""

from pathlib import Path

import pytest

from smalllm.generation import chat_prompt
from smalllm.tokenizer import load_tokenizer

pytestmark = pytest.mark.parity
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_hello_prompt_token_ids() -> None:
    tokenizer = load_tokenizer(PROJECT_ROOT / "artifacts/small-lm-8m/tokenizer.json")
    assert chat_prompt(tokenizer, [{"role": "user", "content": "Hello"}]) == [
        1,
        4,
        204,
        45,
        474,
        84,
        2,
        204,
        5,
        204,
    ]
