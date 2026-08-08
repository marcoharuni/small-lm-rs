# mypy: ignore-errors
"""JIT-compiled loss, gradient accumulation, and evaluation loops."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


def token_loss_sum(logits, targets, mask):
    """Return summed CE loss, valid-token count, and token accuracy count."""

    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32),
        targets,
    )
    loss_sum = jnp.sum(losses * mask)
    valid_tokens = jnp.sum(mask)
    correct = jnp.sum((jnp.argmax(logits, axis=-1) == targets) * mask)
    return loss_sum, valid_tokens, correct


def compile_loss_and_grad(graphdef):
    """Compile the microbatch objective around an immutable NNX graph definition."""

    @jax.jit
    def loss_and_grad(parameter_tree, inputs, targets, mask):
        def objective(candidate_params):
            current_model = nnx.merge(graphdef, candidate_params)
            logits, product_maxima = current_model(inputs, collect_stats=True)
            loss_sum, valid_tokens, correct = token_loss_sum(logits, targets, mask)
            return loss_sum, (valid_tokens, correct, jnp.max(product_maxima))

        return jax.value_and_grad(objective, has_aux=True)(parameter_tree)

    return loss_and_grad


def compile_apply(optimizer):
    """Compile one Optax update application."""

    @jax.jit
    def apply(parameter_tree, optimizer_state, gradients):
        updates, next_state = optimizer.update(gradients, optimizer_state, parameter_tree)
        return optax.apply_updates(parameter_tree, updates), next_state

    return apply


def optimizer_update(
    parameter_tree,
    optimizer_state,
    batch,
    loss_and_grad,
    apply_gradients,
    microbatch_size: int,
):
    """Accumulate token-summed gradients and apply one normalized global update.

    Host synchronization is intentionally delayed until the entire optimizer update has
    completed. This avoids one GPU synchronization per microbatch during long L4 runs.
    """

    inputs, targets, mask = batch
    gradient_sum = jax.tree.map(jnp.zeros_like, parameter_tree)
    loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
    valid_tokens = jnp.asarray(0.0, dtype=jnp.float32)
    correct_tokens = jnp.asarray(0.0, dtype=jnp.float32)
    situ_max = jnp.asarray(0.0, dtype=jnp.float32)

    for start in range(0, inputs.shape[0], microbatch_size):
        micro_mask = mask[start : start + microbatch_size]
        if not np.any(micro_mask):
            continue
        (micro_loss, auxiliary), gradients = loss_and_grad(
            parameter_tree,
            jnp.asarray(inputs[start : start + microbatch_size]),
            jnp.asarray(targets[start : start + microbatch_size]),
            jnp.asarray(micro_mask),
        )
        micro_valid, micro_correct, micro_situ_max = auxiliary
        gradient_sum = jax.tree.map(
            lambda total, gradient: total + gradient,
            gradient_sum,
            gradients,
        )
        loss_sum = loss_sum + micro_loss
        valid_tokens = valid_tokens + micro_valid
        correct_tokens = correct_tokens + micro_correct
        situ_max = jnp.maximum(situ_max, micro_situ_max)

    valid_host = float(valid_tokens)
    if valid_host == 0.0:
        raise RuntimeError("No valid targets in optimizer update")
    gradients = jax.tree.map(lambda gradient: gradient / valid_tokens, gradient_sum)
    gradient_norm = optax.global_norm(gradients)
    parameter_tree, optimizer_state = apply_gradients(
        parameter_tree,
        optimizer_state,
        gradients,
    )

    loss_host = float(loss_sum)
    correct_host = float(correct_tokens)
    gradient_norm_host = float(gradient_norm)
    situ_max_host = float(situ_max)
    if not math.isfinite(gradient_norm_host):
        raise FloatingPointError("Non-finite gradient norm")
    if not math.isfinite(loss_host):
        raise FloatingPointError("Non-finite training loss")

    return (
        parameter_tree,
        optimizer_state,
        {
            "loss": loss_host / valid_host,
            "accuracy": correct_host / valid_host,
            "tokens": int(valid_host),
            "gradient_norm": gradient_norm_host,
            "situ_max": situ_max_host,
        },
    )


def compile_evaluation_step(graphdef):
    """Compile deterministic evaluation over one batch."""

    @jax.jit
    def evaluation_step(parameter_tree, inputs, targets, mask):
        current_model = nnx.merge(graphdef, parameter_tree)
        logits = current_model(inputs)
        return token_loss_sum(logits, targets, mask)

    return evaluation_step


def evaluate(parameter_tree, batcher, evaluation_step, sequences: int = 1):
    """Evaluate a complete token batcher and return loss/perplexity/accuracy."""

    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_tokens = jnp.asarray(0.0, dtype=jnp.float32)
    total_correct = jnp.asarray(0.0, dtype=jnp.float32)
    for inputs, targets, mask in batcher.batches(sequences):
        loss_sum, valid_tokens, correct = evaluation_step(
            parameter_tree,
            jnp.asarray(inputs),
            jnp.asarray(targets),
            jnp.asarray(mask),
        )
        total_loss = total_loss + loss_sum
        total_tokens = total_tokens + valid_tokens
        total_correct = total_correct + correct

    token_count = float(total_tokens)
    if token_count <= 0.0:
        raise RuntimeError("Evaluation dataset contains no valid targets")
    mean_loss = float(total_loss) / token_count
    if not math.isfinite(mean_loss):
        raise FloatingPointError("Non-finite evaluation loss")
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "accuracy": float(total_correct) / token_count,
    }
