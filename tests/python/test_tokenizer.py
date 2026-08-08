"""Tests for the checked-in frozen tokenizer and chat prompt."""

from pathlib import Path

from nilemini.config import MODEL, SPECIAL_TOKENS
from nilemini.generation import chat_prompt
from nilemini.tokenizer import load_tokenizer, tokenizer_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = PROJECT_ROOT / "artifacts" / MODEL.model_name / "tokenizer.json"


def test_frozen_tokenizer_contract_and_digest() -> None:
    tokenizer = load_tokenizer(TOKENIZER)
    assert tokenizer.get_vocab_size() == 8192
    assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == list(range(6))
    assert tokenizer_sha256(TOKENIZER) == (
        "b053a986af42c0f98ff8b73b75b449b0c843a527acb2f75b19d23d0bcd09465a"
    )


def test_hello_prompt_matches_cross_language_reference() -> None:
    tokenizer = load_tokenizer(TOKENIZER)
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
