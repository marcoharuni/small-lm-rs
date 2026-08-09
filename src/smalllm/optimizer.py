# mypy: ignore-errors
"""Muon for transformer matrices and AdamW for embeddings/norms."""

from __future__ import annotations

from typing import Any

import jax
import optax

from smalllm.config import MODEL, ModelConfig, OptimizerSpec

MUON_PARAMETER_PATHS = frozenset(
    {
        ("attention", "q_proj"),
        ("attention", "k_proj"),
        ("attention", "v_proj"),
        ("attention", "o_proj"),
        ("ffn", "gate_proj"),
        ("ffn", "up_proj"),
        ("ffn", "down_proj"),
    }
)
OPTIMIZER_GROUPS = ("muon", "adam_decay", "adam_no_decay")


def path_names(path: tuple[Any, ...]) -> tuple[str, ...]:
    """Normalize JAX key-path entries to strings."""

    return tuple(str(getattr(entry, "key", getattr(entry, "idx", entry))) for entry in path)


def parameter_group(path: tuple[Any, ...], leaf) -> str:
    """Assign one parameter to its exact optimizer group."""

    names = path_names(path)
    if names and names[-1] == ".value":
        names = names[:-1]
    is_transformer_matrix = (
        leaf.ndim == 2
        and len(names) >= 4
        and names[0].startswith("block_")
        and names[-1] == "kernel"
        and tuple(names[-3:-1]) in MUON_PARAMETER_PATHS
    )
    if is_transformer_matrix:
        return "muon"
    if leaf.ndim == 1 and len(names) >= 2 and names[-1] == "weight" and "norm" in names[-2]:
        return "adam_no_decay"
    return "adam_decay"


def labels_for(parameter_tree):
    """Build the optimizer label tree."""

    return jax.tree_util.tree_map_with_path(parameter_group, parameter_tree)


def muon_dimension_numbers(update_tree):
    """Declare matrix reduction/output axes for Optax Muon."""

    return jax.tree.map(
        lambda _: optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1),
        update_tree,
    )


def cosine_schedule(peak: float, updates: int, warmup_fraction: float):
    """Create the warmup + cosine schedule used in the notebook."""

    warmup = max(1, round(warmup_fraction * updates))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak,
        warmup_steps=warmup,
        decay_steps=max(updates, warmup + 1),
        end_value=peak * 0.1,
    )


def build_optimizer(
    parameter_tree,
    updates: int,
    config: OptimizerSpec,
    *,
    model_config: ModelConfig = MODEL,
):
    """Build clipped Muon + AdamW multi-transform optimizer."""

    validate_optimizer_partition(parameter_tree, model_config=model_config)
    muon_schedule = cosine_schedule(
        config.muon_learning_rate,
        updates,
        config.warmup_fraction,
    )
    adam_schedule = cosine_schedule(
        config.adam_learning_rate,
        updates,
        config.warmup_fraction,
    )
    muon = optax.chain(
        optax.contrib.scale_by_muon(
            beta=0.95,
            ns_steps=5,
            nesterov=True,
            weight_dimension_numbers=muon_dimension_numbers,
        ),
        optax.add_decayed_weights(config.weight_decay),
        optax.scale_by_learning_rate(muon_schedule),
    )
    adam_decay = optax.adamw(
        learning_rate=adam_schedule,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    adam_no_decay = optax.adamw(
        learning_rate=adam_schedule,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=0.0,
    )
    partitioned = optax.multi_transform(
        {
            "muon": muon,
            "adam_decay": adam_decay,
            "adam_no_decay": adam_no_decay,
        },
        labels_for(parameter_tree),
    )
    return optax.chain(optax.clip_by_global_norm(config.gradient_clip_norm), partitioned)


def optimizer_group_counts(parameter_tree) -> dict[str, int]:
    """Return parameter counts assigned to each optimizer group."""

    labels = labels_for(parameter_tree)
    counts = {group: 0 for group in OPTIMIZER_GROUPS}
    for label, leaf in zip(jax.tree.leaves(labels), jax.tree.leaves(parameter_tree), strict=True):
        counts[str(label)] += int(leaf.size)
    return counts


def expected_optimizer_group_counts(config: ModelConfig = MODEL) -> dict[str, int]:
    """Return exact parameter counts expected in the three optimizer groups."""

    kv_width = config.num_key_value_heads * config.head_dimension
    attention = 2 * config.hidden_size * config.hidden_size + 2 * config.hidden_size * kv_width
    ffn = 3 * config.hidden_size * config.intermediate_size
    muon = config.num_layers * (attention + ffn)
    no_decay = (2 * config.num_layers + 1) * config.hidden_size
    decay = config.expected_parameter_count - muon - no_decay
    return {"muon": muon, "adam_decay": decay, "adam_no_decay": no_decay}


def validate_optimizer_partition(
    parameter_tree,
    *,
    model_config: ModelConfig = MODEL,
) -> dict[str, int]:
    """Verify every parameter is assigned once and Muon owns exact block matrices."""

    counts = optimizer_group_counts(parameter_tree)
    expected = expected_optimizer_group_counts(model_config)
    if sum(counts.values()) != model_config.expected_parameter_count:
        raise RuntimeError("optimizer partition does not cover every model parameter exactly once")
    if counts != expected:
        raise RuntimeError(f"optimizer partition {counts} does not match expected {expected}")
    return counts
