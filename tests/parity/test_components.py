"""Independent NumPy checks for frozen RMSNorm, RoPE, and SiTU semantics."""

import numpy as np
import pytest

from nilemini.reference import apply_rope, rms_norm, situ_glu

pytestmark = pytest.mark.parity


def test_rmsnorm_rope_and_situ_glu_reference_semantics() -> None:
    x = np.asarray([[3.0, 4.0]], dtype=np.float32)
    weight = np.asarray([1.0, 2.0], dtype=np.float32)
    normalized = rms_norm(x, weight, epsilon=1e-5)
    assert normalized.shape == x.shape
    assert np.all(np.isfinite(normalized))

    query = np.arange(16, dtype=np.float32).reshape(1, 2, 1, 8)
    key = query.copy()
    rotated_query, rotated_key = apply_rope(
        query,
        key,
        positions=np.asarray([0, 1], dtype=np.int32),
        theta=10_000.0,
    )
    np.testing.assert_allclose(rotated_query[:, 0], query[:, 0])
    np.testing.assert_allclose(rotated_key[:, 0], key[:, 0])
    assert not np.allclose(rotated_query[:, 1], query[:, 1])

    product = situ_glu(query, key, beta_gate=4.0, beta_up=25.0)
    assert product.shape == query.shape
    assert np.all(np.isfinite(product))
