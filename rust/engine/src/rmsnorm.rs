//! FP32 RMSNorm matching the JAX reference operation.

use crate::error::{EngineError, Result};
use crate::tensor::validate_matrix;

/// Shape and numerical settings for one RMSNorm operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNorm {
    hidden_size: usize,
    epsilon: f32,
}

impl RmsNorm {
    /// Create an RMSNorm operator after validating its settings.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidConfiguration`] for a zero width or an
    /// invalid epsilon.
    pub fn new(hidden_size: usize, epsilon: f32) -> Result<Self> {
        if hidden_size == 0 {
            return Err(EngineError::invalid_configuration(
                "RMSNorm hidden_size must be greater than zero",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(EngineError::invalid_configuration(
                "RMSNorm epsilon must be finite and greater than zero",
            ));
        }
        Ok(Self {
            hidden_size,
            epsilon,
        })
    }

    /// Return the normalized feature width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Return the numerical stability epsilon.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Normalize one hidden-state row using the supplied scale weights.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidInput`] when either slice length differs
    /// from the configured hidden size.
    pub fn forward(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        self.forward_rows(input, 1, weight)
    }

    /// Normalize a row-major matrix shaped `[rows, hidden_size]`.
    ///
    /// Mean-square accumulation, reciprocal square root, and scaling are
    /// performed in FP32, matching the JAX reference implementation.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidInput`] for a zero row count or
    /// inconsistent input/weight lengths.
    pub fn forward_rows(&self, input: &[f32], rows: usize, weight: &[f32]) -> Result<Vec<f32>> {
        validate_matrix("RMSNorm input", input, rows, self.hidden_size)?;
        if weight.len() != self.hidden_size {
            return Err(EngineError::invalid_input(
                "RMSNorm",
                format!("weight length must equal hidden size {}", self.hidden_size),
            ));
        }

        let mut output = Vec::with_capacity(input.len());
        let width = self.hidden_size as f32;

        for row in input.chunks_exact(self.hidden_size) {
            let square_sum = row
                .iter()
                .fold(0.0_f32, |sum, &value| value.mul_add(value, sum));
            let inverse_rms = (square_sum / width + self.epsilon).sqrt().recip();

            output.extend(
                row.iter()
                    .zip(weight)
                    .map(|(&value, &scale)| value * inverse_rms * scale),
            );
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::RmsNorm;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn normalizes_one_row_and_applies_weights() {
        let norm = RmsNorm::new(4, 1.0e-5).expect("valid RMSNorm");
        let output = norm
            .forward(&[1.0, -1.0, 1.0, -1.0], &[1.0, 2.0, 3.0, 4.0])
            .expect("valid input");
        let inverse_rms = (1.0_f32 + 1.0e-5).sqrt().recip();
        let expected = [
            inverse_rms,
            -2.0 * inverse_rms,
            3.0 * inverse_rms,
            -4.0 * inverse_rms,
        ];

        for (actual, expected) in output.into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn normalizes_multiple_rows_independently() {
        let norm = RmsNorm::new(2, 1.0e-5).expect("valid RMSNorm");
        let output = norm
            .forward_rows(&[1.0, 1.0, 2.0, 2.0], 2, &[1.0, 1.0])
            .expect("valid matrix");

        assert_close(output[0], (1.0_f32 + 1.0e-5).sqrt().recip());
        assert_close(output[2], 2.0 * (4.0_f32 + 1.0e-5).sqrt().recip());
    }

    #[test]
    fn rejects_wrong_shapes() {
        let norm = RmsNorm::new(4, 1.0e-5).expect("valid RMSNorm");
        assert!(norm.forward(&[1.0; 3], &[1.0; 4]).is_err());
        assert!(norm.forward_rows(&[1.0; 8], 2, &[1.0; 3]).is_err());
    }
}
