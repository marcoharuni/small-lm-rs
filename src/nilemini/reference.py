"""Independent NumPy reference implementation of the frozen decoder semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from nilemini.config import ModelConfig

FloatArray = npt.NDArray[np.float32]


def rms_norm(inputs: FloatArray, weight: FloatArray, *, epsilon: float) -> FloatArray:
    """Compute FP32 RMSNorm over the final dimension."""

    values = np.asarray(inputs, dtype=np.float32)
    scale = 1.0 / np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + epsilon)
    return np.asarray(values * scale * np.asarray(weight, dtype=np.float32), dtype=np.float32)


def _rope_one(x: FloatArray, positions: npt.NDArray[np.int32], theta: float) -> FloatArray:
    head_dimension = x.shape[-1]
    if head_dimension % 2:
        raise ValueError("RoPE head dimension must be even")
    inverse_frequency = theta ** (
        -np.arange(0, head_dimension, 2, dtype=np.float32) / np.float32(head_dimension)
    )
    angles = np.asarray(positions, dtype=np.float32)[:, None] * inverse_frequency[None, :]
    cosine = np.cos(angles)[None, :, None, :].astype(np.float32)
    sine = np.sin(angles)[None, :, None, :].astype(np.float32)
    values = np.asarray(x, dtype=np.float32)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = np.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1)
    return np.asarray(rotated.reshape(values.shape), dtype=np.float32)


def apply_rope(
    query: FloatArray,
    key: FloatArray,
    *,
    positions: npt.NDArray[np.int32],
    theta: float,
) -> tuple[FloatArray, FloatArray]:
    """Apply the same interleaved zero-based RoPE to query and key tensors."""

    return _rope_one(query, positions, theta), _rope_one(key, positions, theta)


def situ_glu(
    gate: FloatArray,
    up: FloatArray,
    *,
    beta_gate: float,
    beta_up: float,
) -> FloatArray:
    """Apply the exact SiTU-GLU product before the down projection."""

    gate_values = np.asarray(gate, dtype=np.float32)
    up_values = np.asarray(up, dtype=np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-gate_values))
    gate_branch = beta_gate * np.tanh(gate_values / beta_gate) * sigmoid
    up_branch = beta_up * np.tanh(up_values / beta_up)
    return np.asarray(gate_branch * up_branch, dtype=np.float32)


def _linear(x: FloatArray, exported_weight: FloatArray) -> FloatArray:
    """Apply an exported [out_features, in_features] matrix."""

    return np.asarray(
        np.matmul(x, np.asarray(exported_weight, dtype=np.float32).T),
        dtype=np.float32,
    )


def _softmax(x: FloatArray) -> FloatArray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return np.asarray(exponent / np.sum(exponent, axis=-1, keepdims=True), dtype=np.float32)


def forward_logits(
    config: ModelConfig,
    parameters: Mapping[str, FloatArray],
    input_ids: npt.NDArray[np.int32],
) -> FloatArray:
    """Run an independent FP32 decoder over tensors in the exported Rust layout."""

    ids = np.asarray(input_ids, dtype=np.int32)
    if ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if ids.shape[1] > config.context_length:
        raise ValueError("sequence exceeds context length")
    if np.any(ids < 0) or np.any(ids >= config.vocab_size):
        raise ValueError("input_ids contain an out-of-vocabulary token")

    embedding = np.asarray(parameters["token_embedding.weight"], dtype=np.float32)
    hidden = embedding[ids]
    batch, length, _ = hidden.shape
    positions = np.arange(length, dtype=np.int32)
    groups = config.num_query_heads // config.num_key_value_heads
    causal = np.arange(length)[:, None] >= np.arange(length)[None, :]

    for layer in range(config.num_layers):
        prefix = f"layers.{layer}"
        normalized = rms_norm(
            hidden,
            np.asarray(parameters[f"{prefix}.attention_norm.weight"], dtype=np.float32),
            epsilon=config.rms_norm_epsilon,
        )
        query = _linear(normalized, parameters[f"{prefix}.q_proj.weight"]).reshape(
            batch, length, config.num_query_heads, config.head_dimension
        )
        key = _linear(normalized, parameters[f"{prefix}.k_proj.weight"]).reshape(
            batch, length, config.num_key_value_heads, config.head_dimension
        )
        value = _linear(normalized, parameters[f"{prefix}.v_proj.weight"]).reshape(
            batch, length, config.num_key_value_heads, config.head_dimension
        )
        query, key = apply_rope(query, key, positions=positions, theta=config.rope_theta)
        query = query.reshape(
            batch,
            length,
            config.num_key_value_heads,
            groups,
            config.head_dimension,
        )
        scores = np.einsum("btkgh,bskh->bkgts", query, key, optimize=True).astype(np.float32)
        scores /= np.float32(math.sqrt(config.head_dimension))
        scores = np.where(causal[None, None, None, :, :], scores, np.float32(-np.inf))
        probabilities = _softmax(scores)
        attended = np.einsum("bkgts,bskh->btkgh", probabilities, value, optimize=True)
        attention_output = _linear(
            np.asarray(attended.reshape(batch, length, config.hidden_size), dtype=np.float32),
            parameters[f"{prefix}.o_proj.weight"],
        )
        hidden = np.asarray(hidden + attention_output, dtype=np.float32)

        normalized = rms_norm(
            hidden,
            np.asarray(parameters[f"{prefix}.ffn_norm.weight"], dtype=np.float32),
            epsilon=config.rms_norm_epsilon,
        )
        gate = _linear(normalized, parameters[f"{prefix}.gate_proj.weight"])
        up = _linear(normalized, parameters[f"{prefix}.up_proj.weight"])
        product = situ_glu(
            gate,
            up,
            beta_gate=config.situ_beta_gate,
            beta_up=config.situ_beta_up,
        )
        hidden = np.asarray(
            hidden + _linear(product, parameters[f"{prefix}.down_proj.weight"]),
            dtype=np.float32,
        )

    hidden = rms_norm(
        hidden,
        np.asarray(parameters["final_norm.weight"], dtype=np.float32),
        epsilon=config.rms_norm_epsilon,
    )
    return np.asarray(np.einsum("btd,vd->btv", hidden, embedding, optimize=True), dtype=np.float32)
