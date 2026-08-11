from __future__ import annotations

import numpy as np
import pytest

from smalllm.quantization import (
    SCALE_SUFFIX,
    is_quantized_projection,
    quantize_per_output_channel,
    quantize_tensors,
)


def test_projection_selection_is_explicit() -> None:
    assert is_quantized_projection("layers.0.q_proj.weight")
    assert is_quantized_projection("layers.7.down_proj.weight")
    assert not is_quantized_projection("token_embedding.weight")
    assert not is_quantized_projection("layers.0.attention_norm.weight")
    assert not is_quantized_projection("final_norm.weight")


def test_per_channel_quantization_uses_symmetric_row_scales() -> None:
    weight = np.array(
        [
            [1.0, -0.5, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )

    quantized, scales = quantize_per_output_channel(weight)

    assert quantized.dtype == np.int8
    assert scales.dtype == np.float32
    np.testing.assert_array_equal(quantized[0], np.array([127, -64, 0], dtype=np.int8))
    np.testing.assert_array_equal(quantized[1], np.zeros(3, dtype=np.int8))
    np.testing.assert_array_equal(quantized[2], np.array([127, -127, 64], dtype=np.int8))
    np.testing.assert_allclose(scales, np.array([1.0 / 127.0, 1.0, 2.0 / 127.0], np.float32))


def test_quantize_tensors_keeps_embeddings_and_norms_fp32() -> None:
    tensors = {
        "token_embedding.weight": np.arange(8, dtype=np.float32).reshape(4, 2),
        "layers.0.attention_norm.weight": np.ones(2, dtype=np.float32),
        "layers.0.q_proj.weight": np.array([[1.0, -1.0], [0.5, -0.25]], dtype=np.float32),
        "final_norm.weight": np.ones(2, dtype=np.float32),
    }

    result = quantize_tensors(tensors)

    assert result["token_embedding.weight"].dtype == np.float32
    assert result["layers.0.attention_norm.weight"].dtype == np.float32
    assert result["final_norm.weight"].dtype == np.float32
    assert result["layers.0.q_proj.weight"].dtype == np.int8
    assert result[f"layers.0.q_proj.weight{SCALE_SUFFIX}"].shape == (2,)


def test_quantization_rejects_invalid_projection_inputs() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        quantize_per_output_channel(np.ones(4, dtype=np.float32))

    with pytest.raises(ValueError, match="float32"):
        quantize_per_output_channel(np.ones((2, 2), dtype=np.float64))

    invalid = np.ones((2, 2), dtype=np.float32)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        quantize_per_output_channel(invalid)


def test_quantize_tensors_rejects_non_fp32_source_tensor() -> None:
    with pytest.raises(ValueError, match="must be float32"):
        quantize_tensors({"layers.0.q_proj.weight": np.ones((2, 2), dtype=np.float16)})
