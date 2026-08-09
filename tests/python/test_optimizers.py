"""Tests for the frozen Muon/AdamW partition contract."""

from smalllm.config import MODEL
from smalllm.optimizer import expected_optimizer_group_counts


def test_optimizer_group_counts_cover_the_exact_model() -> None:
    counts = expected_optimizer_group_counts(MODEL)
    assert counts["muon"] > 0
    assert counts["adam_decay"] == MODEL.vocab_size * MODEL.hidden_size
    assert counts["adam_no_decay"] == (2 * MODEL.num_layers + 1) * MODEL.hidden_size
    assert sum(counts.values()) == 7_999_744
