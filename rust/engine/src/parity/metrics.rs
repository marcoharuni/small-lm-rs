//! Numerical metrics for comparing Rust and JAX logits.

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// Aggregate numerical and top-1 agreement statistics for two logit tensors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParityMetrics {
    /// Number of scalar logits compared.
    pub element_count: usize,
    /// Largest absolute scalar difference.
    pub max_abs_error: f64,
    /// Mean absolute scalar difference.
    pub mean_abs_error: f64,
    /// Root-mean-square scalar difference.
    pub root_mean_square_error: f64,
    /// Cosine similarity across all flattened logits.
    pub cosine_similarity: f64,
    /// Number of token positions with the same top-1 token.
    pub top1_matches: usize,
    /// Number of token positions compared.
    pub top1_total: usize,
    /// Fraction of token positions with the same top-1 token.
    pub top1_agreement: f64,
    /// Rust top-1 token at every position.
    pub rust_top1_token_ids: Vec<u32>,
    /// JAX top-1 token at every position.
    pub reference_top1_token_ids: Vec<u32>,
    /// Whether Rust and JAX choose the same final-position next token.
    pub final_position_top1_match: bool,
}

impl ParityMetrics {
    /// Compare equal-shaped row-major logits.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for zero dimensions, inconsistent slice
    /// lengths, non-finite values, or dimensions that overflow.
    pub fn compare(
        actual: &[f32],
        expected: &[f32],
        batch_size: usize,
        sequence_length: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        if batch_size == 0 || sequence_length == 0 || vocab_size == 0 {
            return Err(EngineError::invalid_input(
                "logit parity",
                "batch, sequence, and vocabulary dimensions must be non-zero",
            ));
        }
        let rows = batch_size.checked_mul(sequence_length).ok_or_else(|| {
            EngineError::invalid_input("logit parity", "row count overflows usize")
        })?;
        let expected_count = rows.checked_mul(vocab_size).ok_or_else(|| {
            EngineError::invalid_input("logit parity", "element count overflows usize")
        })?;
        if actual.len() != expected_count || expected.len() != expected_count {
            return Err(EngineError::invalid_input(
                "logit parity",
                format!(
                    "actual and reference lengths must both equal {expected_count}; found {} and {}",
                    actual.len(),
                    expected.len()
                ),
            ));
        }
        if actual
            .iter()
            .chain(expected)
            .any(|value| !value.is_finite())
        {
            return Err(EngineError::invalid_input(
                "logit parity",
                "all compared logits must be finite",
            ));
        }

        let mut max_abs_error = 0.0_f64;
        let mut absolute_error_sum = 0.0_f64;
        let mut squared_error_sum = 0.0_f64;
        let mut dot_product = 0.0_f64;
        let mut actual_square_sum = 0.0_f64;
        let mut expected_square_sum = 0.0_f64;

        for (&actual_value, &expected_value) in actual.iter().zip(expected) {
            let actual_value = f64::from(actual_value);
            let expected_value = f64::from(expected_value);
            let difference = actual_value - expected_value;
            let absolute = difference.abs();
            max_abs_error = max_abs_error.max(absolute);
            absolute_error_sum += absolute;
            squared_error_sum += difference * difference;
            dot_product += actual_value * expected_value;
            actual_square_sum += actual_value * actual_value;
            expected_square_sum += expected_value * expected_value;
        }

        let element_count_f64 = expected_count as f64;
        let mean_abs_error = absolute_error_sum / element_count_f64;
        let root_mean_square_error = (squared_error_sum / element_count_f64).sqrt();
        let cosine_denominator = (actual_square_sum * expected_square_sum).sqrt();
        let cosine_similarity = if cosine_denominator == 0.0 {
            if actual_square_sum == 0.0 && expected_square_sum == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            dot_product / cosine_denominator
        };

        let rust_top1_token_ids = actual
            .chunks_exact(vocab_size)
            .map(argmax_token)
            .collect::<Vec<_>>();
        let reference_top1_token_ids = expected
            .chunks_exact(vocab_size)
            .map(argmax_token)
            .collect::<Vec<_>>();
        let top1_matches = rust_top1_token_ids
            .iter()
            .zip(&reference_top1_token_ids)
            .filter(|(actual_token, expected_token)| actual_token == expected_token)
            .count();
        let final_position_top1_match =
            rust_top1_token_ids.last() == reference_top1_token_ids.last();

        Ok(Self {
            element_count: expected_count,
            max_abs_error,
            mean_abs_error,
            root_mean_square_error,
            cosine_similarity,
            top1_matches,
            top1_total: rows,
            top1_agreement: top1_matches as f64 / rows as f64,
            rust_top1_token_ids,
            reference_top1_token_ids,
            final_position_top1_match,
        })
    }
}

fn argmax_token(row: &[f32]) -> u32 {
    let mut best_index = 0_usize;
    let mut best_value = row[0];
    for (index, &value) in row.iter().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index as u32
}

#[cfg(test)]
mod tests {
    use super::ParityMetrics;

    #[test]
    fn computes_scalar_and_top1_metrics() {
        let expected = [0.0, 1.0, 2.0, 3.0, 1.0, 0.0];
        let actual = [0.0, 1.0, 2.0, 3.0, 0.5, 0.0];
        let metrics =
            ParityMetrics::compare(&actual, &expected, 1, 2, 3).expect("valid comparison");

        assert_eq!(metrics.element_count, 6);
        assert_eq!(metrics.max_abs_error, 0.5);
        assert_eq!(metrics.top1_matches, 2);
        assert_eq!(metrics.top1_agreement, 1.0);
        assert_eq!(metrics.rust_top1_token_ids, vec![2, 0]);
        assert_eq!(metrics.reference_top1_token_ids, vec![2, 0]);
        assert!(metrics.final_position_top1_match);
    }

    #[test]
    fn rejects_non_finite_and_inconsistent_logits() {
        assert!(ParityMetrics::compare(&[0.0], &[0.0, 1.0], 1, 1, 2).is_err());
        assert!(ParityMetrics::compare(&[f32::NAN], &[0.0], 1, 1, 1).is_err());
    }
}
