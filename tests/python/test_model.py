"""Numerical smoke tests for the real JAX/Flax decoder."""

import jax
import jax.numpy as jnp
from flax import nnx

from smalllm.config import ModelConfig
from smalllm.model import SmallLM


def tiny_config() -> ModelConfig:
    vocab = 128
    layers = 2
    hidden = 32
    intermediate = 64
    q_heads = 4
    kv_heads = 2
    head_dim = 8
    kv_width = kv_heads * head_dim
    count = (
        vocab * hidden
        + layers
        * (2 * hidden * hidden + 2 * hidden * kv_width + 3 * hidden * intermediate + 2 * hidden)
        + hidden
    )
    return ModelConfig(
        model_name="tiny-test",
        vocab_size=vocab,
        context_length=16,
        num_layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_query_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dimension=head_dim,
        expected_parameter_count=count,
    )


def test_tiny_model_forward_is_finite_and_has_expected_shape() -> None:
    config = tiny_config()
    model = SmallLM(config, rngs=nnx.Rngs(params=7))
    logits, situ = model(jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32), collect_stats=True)
    assert logits.shape == (1, 4, config.vocab_size)
    assert situ.shape == (config.num_layers,)
    assert bool(jnp.all(jnp.isfinite(logits)))
    graphdef, params = nnx.split(model, nnx.Param)
    assert graphdef is not None
    assert (
        sum(int(leaf.size) for leaf in jax.tree.leaves(params)) == config.expected_parameter_count
    )
