"""Validate the checked-in JAX logit reference consumed by Rust parity tests."""

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

pytestmark = pytest.mark.parity
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_exported_jax_reference_logits_are_complete_and_finite() -> None:
    tensors = load_file(
        str(PROJECT_ROOT / "artifacts/small-lm-8m/reference_outputs.safetensors")
    )
    logits = tensors["logits"]
    assert logits.shape == (1, 10, 8192)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))
