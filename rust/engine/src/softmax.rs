//! Numerically stable FP32 softmax operations.

use crate::error::{EngineError, Result};
use crate::tensor::validate_matrix;

/// Apply numerically stable softmax to one vector in place.
///
/// # Errors
///
/// Returns [`EngineError::InvalidInput`] for an empty vector, non-finite input,
/// or a non-finite/zero normalization sum.
pub fn softmax_in_place(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(EngineError::invalid_input(
            "softmax",
            "at least one value is required",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::invalid_input(
            "softmax",
            "all input values must be finite",
        ));
    }

    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }

    if !sum.is_finite() || sum <= 0.0 {
        return Err(EngineError::invalid_input(
            "softmax",
            "normalization sum must be finite and greater than zero",
        ));
    }

    for value in values {
        *value /= sum;
    }
    Ok(())
}

/// Apply softmax independently to every row of a row-major matrix.
///
/// # Errors
///
/// Returns an invalid-input error for an inconsistent matrix shape or any row
/// that cannot be normalized.
pub fn softmax_rows(values: &mut [f32], rows: usize, columns: usize) -> Result<()> {
    validate_matrix("softmax matrix", values, rows, columns)?;
    for row in values.chunks_exact_mut(columns) {
        softmax_in_place(row)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{softmax_in_place, softmax_rows};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn remains_stable_for_large_logits() {
        let mut values = [1_000.0, 1_001.0];
        softmax_in_place(&mut values).expect("valid softmax");

        let first = 1.0_f32 / (1.0 + 1.0_f32.exp());
        assert_close(values[0], first);
        assert_close(values[1], 1.0 - first);
        assert_close(values.iter().sum(), 1.0);
    }

    #[test]
    fn normalizes_rows_independently() {
        let mut values = [0.0, 0.0, 2.0, 2.0];
        softmax_rows(&mut values, 2, 2).expect("valid matrix");

        for row in values.chunks_exact(2) {
            assert_close(row.iter().sum(), 1.0);
            assert_close(row[0], 0.5);
            assert_close(row[1], 0.5);
        }
    }

    #[test]
    fn rejects_empty_and_non_finite_inputs() {
        assert!(softmax_in_place(&mut []).is_err());
        assert!(softmax_in_place(&mut [0.0, f32::NAN]).is_err());
    }
}
