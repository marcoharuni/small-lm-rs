# mypy: ignore-errors
"""Exact Flax NNX NileMini-8M-SiTU training model."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from nilemini.config import MODEL, ModelConfig

PARAM_DTYPE = jnp.float32
COMPUTE_DTYPE = jnp.bfloat16


def normal_param(rngs: nnx.Rngs, shape: tuple[int, ...], std: float) -> nnx.Param:
    """Create one FP32 normally initialized NNX parameter."""

    value = jax.random.normal(rngs.params(), shape, dtype=jnp.float32) * std
    return nnx.Param(value.astype(PARAM_DTYPE))


def matmul(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Run BF16 multiplicands with FP32 accumulation, matching the frozen notebook."""

    x_compute = x.astype(COMPUTE_DTYPE)
    weight_compute = weight.astype(COMPUTE_DTYPE)
    return jax.lax.dot_general(
        x_compute,
        weight_compute,
        (((x_compute.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


class Linear(nnx.Module):
    """Bias-free dense projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        std: float = 0.02,
    ):
        self.kernel = normal_param(rngs, (in_features, out_features), std)

    def __call__(self, x: jax.Array) -> jax.Array:
        return matmul(x, self.kernel.get_value())


class RMSNorm(nnx.Module):
    """FP32 RMSNorm."""

    def __init__(self, width: int, epsilon: float):
        self.weight = nnx.Param(jnp.ones((width,), dtype=PARAM_DTYPE))
        self.epsilon = epsilon

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32)
        scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.epsilon)
        return x * scale * self.weight.get_value()


def apply_rope(x: jax.Array, positions: jax.Array, theta: float) -> jax.Array:
    """Apply interleaved rotary position embeddings."""

    head_dimension = x.shape[-1]
    inverse_frequency = theta ** (
        -jnp.arange(0, head_dimension, 2, dtype=jnp.float32) / head_dimension
    )
    angles = positions.astype(jnp.float32)[:, None] * inverse_frequency[None, :]
    cosine = jnp.cos(angles)[None, :, None, :]
    sine = jnp.sin(angles)[None, :, None, :]
    even = x.astype(jnp.float32)[..., 0::2]
    odd = x.astype(jnp.float32)[..., 1::2]
    rotated = jnp.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        axis=-1,
    )
    return rotated.reshape(x.shape)


class GroupedQueryAttention(nnx.Module):
    """10-query-head / 2-KV-head causal grouped-query attention."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        self.config = config
        self.q_proj = Linear(config.hidden_size, config.hidden_size, rngs=rngs)
        kv_width = config.num_key_value_heads * config.head_dimension
        self.k_proj = Linear(config.hidden_size, kv_width, rngs=rngs)
        self.v_proj = Linear(config.hidden_size, kv_width, rngs=rngs)
        self.o_proj = Linear(
            config.hidden_size,
            config.hidden_size,
            rngs=rngs,
            std=residual_std,
        )

    def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        config = self.config
        batch, length, _ = x.shape
        groups = config.num_query_heads // config.num_key_value_heads

        query = self.q_proj(x).reshape(
            batch,
            length,
            config.num_key_value_heads,
            groups,
            config.head_dimension,
        )
        key = self.k_proj(x).reshape(
            batch,
            length,
            config.num_key_value_heads,
            config.head_dimension,
        )
        value = self.v_proj(x).reshape(
            batch,
            length,
            config.num_key_value_heads,
            config.head_dimension,
        )

        query = apply_rope(
            query.reshape(
                batch,
                length,
                config.num_query_heads,
                config.head_dimension,
            ),
            positions,
            config.rope_theta,
        ).reshape(
            batch,
            length,
            config.num_key_value_heads,
            groups,
            config.head_dimension,
        )
        key = apply_rope(key, positions, config.rope_theta)

        scores = jnp.einsum(
            "btkgh,bskh->bkgts",
            query.astype(jnp.float32),
            key.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ) / math.sqrt(config.head_dimension)
        causal_mask = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
        scores = jnp.where(
            causal_mask[None, None, None, :, :],
            scores,
            jnp.finfo(jnp.float32).min,
        )
        probabilities = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum(
            "bkgts,bskh->btkgh",
            probabilities,
            value.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        return self.o_proj(attended.reshape(batch, length, config.hidden_size))


class SiTUGLU(nnx.Module):
    """SiTU-gated feed-forward block."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        self.config = config
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.down_proj = Linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
            std=residual_std,
        )

    def __call__(self, x: jax.Array, *, return_product_max: bool = False):
        config = self.config
        gate = self.gate_proj(x).astype(jnp.float32)
        up = self.up_proj(x).astype(jnp.float32)
        gate_branch = (
            config.situ_beta_gate * jnp.tanh(gate / config.situ_beta_gate) * jax.nn.sigmoid(gate)
        )
        up_branch = config.situ_beta_up * jnp.tanh(up / config.situ_beta_up)
        product = gate_branch * up_branch
        output = self.down_proj(product)
        if return_product_max:
            return output, jnp.max(jnp.abs(product))
        return output


class TransformerBlock(nnx.Module):
    """Pre-norm residual transformer block."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.attention = GroupedQueryAttention(config, rngs=rngs)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.ffn = SiTUGLU(config, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        *,
        collect_stats: bool = False,
    ):
        x = x + self.attention(self.attention_norm(x), positions)
        if collect_stats:
            output, product_max = self.ffn(
                self.ffn_norm(x),
                return_product_max=True,
            )
            return x + output, product_max
        return x + self.ffn(self.ffn_norm(x))


class NileMini(nnx.Module):
    """7,999,744-parameter decoder-only language model."""

    def __init__(self, config: ModelConfig = MODEL, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = normal_param(
            rngs,
            (config.vocab_size, config.hidden_size),
            0.02,
        )
        for layer in range(config.num_layers):
            setattr(self, f"block_{layer}", TransformerBlock(config, rngs=rngs))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)

    def __call__(self, token_ids: jax.Array, *, collect_stats: bool = False):
        if token_ids.ndim != 2:
            raise ValueError("Token IDs must have shape [batch, sequence].")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("Sequence exceeds the context length.")

        positions = jnp.arange(token_ids.shape[1], dtype=jnp.int32)
        hidden = self.token_embedding.get_value()[token_ids]
        product_maxima = []
        for layer in range(self.config.num_layers):
            block = getattr(self, f"block_{layer}")
            if collect_stats:
                hidden, product_max = block(hidden, positions, collect_stats=True)
                product_maxima.append(product_max)
            else:
                hidden = block(hidden, positions)

        hidden = self.final_norm(hidden)
        logits = jnp.einsum(
            "btd,vd->btv",
            hidden.astype(jnp.float32),
            self.token_embedding.get_value().astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        if collect_stats:
            return logits, jnp.stack(product_maxima)
        return logits


def initialize_model(seed: int, config: ModelConfig = MODEL):
    """Construct NileMini and return its NNX graph definition and parameter state."""

    model = NileMini(config, rngs=nnx.Rngs(params=seed))
    graphdef, params = nnx.split(model, nnx.Param)
    count = sum(int(leaf.size) for leaf in jax.tree.leaves(params))
    if count != config.expected_parameter_count:
        raise RuntimeError(f"parameter count {count} != {config.expected_parameter_count}")
    return graphdef, params
