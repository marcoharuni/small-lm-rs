"""Opt-in integration test for the real OpenAI-shaped chat endpoint."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

import pytest


@pytest.mark.api
def test_chat_completions_returns_a_valid_response() -> None:
    server_url = os.environ.get("NILEMINI_SERVER_URL")
    if server_url is None:
        pytest.skip("set NILEMINI_SERVER_URL to run API integration tests")

    request = Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "nilemini-8m-situ",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "nilemini-8m-situ"
    assert len(payload["choices"]) == 1
    assert payload["choices"][0]["message"]["role"] == "assistant"
