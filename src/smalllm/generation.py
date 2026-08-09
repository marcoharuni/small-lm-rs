# mypy: ignore-errors
"""Frozen chat prompt formatting and simple JAX reference generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from flax import nnx

from smalllm.config import MODEL, SPECIAL_TOKENS

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN = SPECIAL_TOKENS
ROLE_TOKENS = {
    "system": SYSTEM_TOKEN,
    "user": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
}


def _validate_messages(messages: Sequence[Mapping[str, str]]) -> None:
    """Validate the inference conversation accepted by the Rust server contract."""

    if not messages:
        raise ValueError("chat messages must not be empty")
    roles = [str(message.get("role", "")) for message in messages]
    dialogue = roles[1:] if roles and roles[0] == "system" else roles
    if roles.count("system") > 1 or "system" in dialogue:
        raise ValueError("system role is allowed at most once and only as the first message")
    if not dialogue or dialogue[-1] != "user":
        raise ValueError("inference chat history must end with a user message")
    for index, role in enumerate(dialogue):
        expected = "user" if index % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError(f"chat roles must alternate user/assistant; expected {expected}")
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in ROLE_TOKENS or not isinstance(content, str) or not content.strip():
            raise ValueError("every chat message needs a supported role and non-empty content")
        if any(marker in content for marker in SPECIAL_TOKENS):
            raise ValueError("chat content must not contain reserved tokenizer markers")


def chat_prompt(tokenizer, messages: Sequence[Mapping[str, str]]) -> list[int]:
    """Encode the exact frozen prompt consumed by JAX and Rust inference."""

    _validate_messages(messages)
    token_ids = [int(tokenizer.token_to_id(BOS_TOKEN))]
    newline = tokenizer.encode("\n", add_special_tokens=False).ids
    eos_id = int(tokenizer.token_to_id(EOS_TOKEN))
    for message in messages:
        role = str(message["role"])
        token_ids.append(int(tokenizer.token_to_id(ROLE_TOKENS[role])))
        token_ids.extend(newline)
        token_ids.extend(
            tokenizer.encode(str(message["content"]).strip(), add_special_tokens=False).ids
        )
        token_ids.append(eos_id)
        token_ids.extend(newline)
    token_ids.append(int(tokenizer.token_to_id(ASSISTANT_TOKEN)))
    token_ids.extend(newline)
    if len(token_ids) >= MODEL.context_length:
        raise ValueError("formatted prompt leaves no room for generation")
    return token_ids


def generate(
    graphdef,
    parameter_tree,
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    max_new_tokens: int = 32,
) -> tuple[str, list[int]]:
    """Run deterministic uncached greedy generation as a JAX reference path."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    token_ids = chat_prompt(tokenizer, messages)
    eos_id = int(tokenizer.token_to_id(EOS_TOKEN))
    model = nnx.merge(graphdef, parameter_tree)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        if len(token_ids) >= MODEL.context_length:
            break
        logits = model(jnp.asarray(token_ids, dtype=jnp.int32)[None, :])[0, -1]
        next_token = int(jnp.argmax(logits))
        if next_token == eos_id:
            break
        token_ids.append(next_token)
        generated.append(next_token)
    return tokenizer.decode(generated, skip_special_tokens=True), generated
