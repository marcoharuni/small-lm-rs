"""Compatibility re-exports for the canonical optimizer module."""

from nilemini.optimizer import (
    OPTIMIZER_GROUPS,
    build_optimizer,
    labels_for,
    optimizer_group_counts,
    validate_optimizer_partition,
)

__all__ = [
    "OPTIMIZER_GROUPS",
    "build_optimizer",
    "labels_for",
    "optimizer_group_counts",
    "validate_optimizer_partition",
]
