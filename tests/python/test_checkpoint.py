"""Tests for real Orbax step discovery and parameter checkpoint round trips."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nilemini.checkpoint import latest_checkpoint, restore_parameters, save_parameters, step_path


def test_step_paths_sort_numerically(tmp_path: Path) -> None:
    step_path(tmp_path, 2).mkdir()
    step_path(tmp_path, 10).mkdir()
    assert latest_checkpoint(tmp_path) == step_path(tmp_path, 10)


def test_parameter_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "params"
    source = {"weight": jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)}
    target = {"weight": jnp.zeros((3,), dtype=jnp.float32)}
    save_parameters(path, source)
    restored = restore_parameters(path, target)
    np.testing.assert_allclose(np.asarray(restored["weight"]), np.asarray(source["weight"]))
