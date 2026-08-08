//! Interleaved rotary positional embeddings for attention heads.

use crate::error::{EngineError, Result};

/// Rotary embedding settings and cached inverse frequencies.
#[derive(Clone, Debug, PartialEq)]
pub struct RotaryEmbedding {
    head_dimension: usize,
    theta: f32,
    inverse_frequencies: Vec<f32>,
}

impl RotaryEmbedding {
    /// Create rotary embedding settings and cache one inverse frequency per
    /// interleaved even/odd feature pair.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidConfiguration`] when the head dimension is
    /// zero or odd, or when theta is not finite and positive.
    pub fn new(head_dimension: usize, theta: f32) -> Result<Self> {
        if head_dimension == 0 || head_dimension % 2 != 0 {
            return Err(EngineError::invalid_configuration(
                "RoPE head_dimension must be positive and even",
            ));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(EngineError::invalid_configuration(
                "RoPE theta must be finite and greater than zero",
            ));
        }

        let denominator = head_dimension as f32;
        let inverse_frequencies = (0..head_dimension / 2)
            .map(|pair| theta.powf(-(2.0 * pair as f32) / denominator))
            .collect();

        Ok(Self {
            head_dimension,
            theta,
            inverse_frequencies,
        })
    }

    /// Return the attention-head width.
    #[must_use]
    pub const fn head_dimension(&self) -> usize {
        self.head_dimension
    }

    /// Return the rotary base frequency.
    #[must_use]
    pub const fn theta(&self) -> f32 {
        self.theta
    }

    /// Return cached inverse frequencies in even/odd pair order.
    #[must_use]
    pub fn inverse_frequencies(&self) -> &[f32] {
        &self.inverse_frequencies
    }

    /// Return one angle per even/odd feature pair for an absolute position.
    #[must_use]
    pub fn angles(&self, position: usize) -> Vec<f32> {
        let position = position as f32;
        self.inverse_frequencies
            .iter()
            .map(|&frequency| position * frequency)
            .collect()
    }

    /// Rotate one interleaved attention head in place.
    ///
    /// Features `(0, 1)`, `(2, 3)`, and so on are treated as even/odd pairs,
    /// matching the JAX reference implementation.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidInput`] when the head width differs from
    /// the configured dimension.
    pub fn apply_head_in_place(&self, head: &mut [f32], position: usize) -> Result<()> {
        if head.len() != self.head_dimension {
            return Err(EngineError::invalid_input(
                "RoPE",
                format!("head length must equal {}", self.head_dimension),
            ));
        }

        let position = position as f32;
        for (pair, &inverse_frequency) in head.chunks_exact_mut(2).zip(&self.inverse_frequencies) {
            let angle = position * inverse_frequency;
            let (sin, cos) = angle.sin_cos();
            let even = pair[0];
            let odd = pair[1];
            pair[0] = even * cos - odd * sin;
            pair[1] = even * sin + odd * cos;
        }
        Ok(())
    }

    /// Rotate a row-major collection of heads at one absolute position.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when the number of values is not exactly
    /// `num_heads * head_dimension` or when `num_heads` is zero.
    pub fn apply_heads_in_place(
        &self,
        values: &mut [f32],
        num_heads: usize,
        position: usize,
    ) -> Result<()> {
        if num_heads == 0 {
            return Err(EngineError::invalid_input(
                "RoPE",
                "num_heads must be greater than zero",
            ));
        }
        let expected = num_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| EngineError::invalid_input("RoPE", "head shape overflows usize"))?;
        if values.len() != expected {
            return Err(EngineError::invalid_input(
                "RoPE",
                format!("received {} values, expected {expected}", values.len()),
            ));
        }

        for head in values.chunks_exact_mut(self.head_dimension) {
            self.apply_head_in_place(head, position)?;
        }
        Ok(())
    }

    /// Rotate one query and key head at the requested token position.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidInput`] for inconsistent head lengths.
    pub fn apply(&self, query: &mut [f32], key: &mut [f32], position: usize) -> Result<()> {
        self.apply_head_in_place(query, position)?;
        self.apply_head_in_place(key, position)
    }
}

#[cfg(test)]
mod tests {
    use super::RotaryEmbedding;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn caches_reference_inverse_frequencies() {
        let rotary = RotaryEmbedding::new(4, 10_000.0).expect("valid RoPE");
        assert_eq!(rotary.inverse_frequencies().len(), 2);
        assert_close(rotary.inverse_frequencies()[0], 1.0);
        assert_close(rotary.inverse_frequencies()[1], 0.01);
    }

    #[test]
    fn position_zero_is_identity() {
        let rotary = RotaryEmbedding::new(4, 10_000.0).expect("valid RoPE");
        let mut head = [1.0, 2.0, 3.0, 4.0];
        rotary
            .apply_head_in_place(&mut head, 0)
            .expect("valid head");
        assert_eq!(head, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rotates_interleaved_even_odd_pairs() {
        let rotary = RotaryEmbedding::new(2, 10_000.0).expect("valid RoPE");
        let mut query = [1.0, 0.0];
        let mut key = [0.0, 1.0];
        rotary.apply(&mut query, &mut key, 1).expect("valid heads");

        let (sin, cos) = 1.0_f32.sin_cos();
        assert_close(query[0], cos);
        assert_close(query[1], sin);
        assert_close(key[0], -sin);
        assert_close(key[1], cos);
    }

    #[test]
    fn rotates_multiple_heads_and_rejects_bad_shapes() {
        let rotary = RotaryEmbedding::new(2, 10_000.0).expect("valid RoPE");
        let mut heads = [1.0, 0.0, 0.0, 1.0];
        rotary
            .apply_heads_in_place(&mut heads, 2, 1)
            .expect("valid heads");
        assert!(rotary.apply_heads_in_place(&mut heads, 3, 1).is_err());
    }
}
