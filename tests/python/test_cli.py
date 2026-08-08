"""CLI tests that exercise configuration/tokenizer validation without network or GPU."""

import json

import pytest

from nilemini.cli import main


def test_doctor_reports_ready(capsys: pytest.CaptureFixture[str]) -> None:
    main(["doctor"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["parameters"] == 7_999_744
    assert payload["pretraining_profiles"]["pilot-l4-20m"]["tokens"] == 20_000_000
    assert payload["sft"]["train"] == 68_000
