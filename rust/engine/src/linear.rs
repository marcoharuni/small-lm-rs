//! Bias-free linear projection using exported `[out, in]` weights.

use crate::error::Result;
use crate::tensor::{checked_matrix_len, validate_matrix};

/// Round an FP32 value to the nearest representable bfloat16 value and return
/// it as FP32.
///
/// This mirrors the JAX reference model's conversion of activations and
/// weights to bfloat16 before dot products.
#[must_use]
pub(crate) fn round_to_bfloat16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }

    let bits = value.to_bits();
    let least_significant_kept_bit = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + least_significant_kept_bit);
    f32::from_bits(rounded & 0xffff_0000)
}

/// Apply a bias-free linear projection to a row-major input matrix.
///
/// `input` has shape `[rows, in_features]`. `weight` follows the exported
/// NileMini layout `[out_features, in_features]`. The returned vector has
/// shape `[rows, out_features]`.
///
/// Inputs and weights are rounded to bfloat16 before multiplication, while
/// accumulation remains FP32 to follow the JAX reference computation.
///
/// # Errors
///
/// Returns an invalid-input error when any dimension or flat slice length is
/// inconsistent.
pub fn linear(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    validate_matrix("linear input", input, rows, in_features)?;
    validate_matrix("linear weight", weight, out_features, in_features)?;

    let output_len = checked_matrix_len("linear output", rows, out_features)?;
    let mut output = vec![0.0_f32; output_len];

    for (input_row, output_row) in input
        .chunks_exact(in_features)
        .zip(output.chunks_exact_mut(out_features))
    {
        for (output_index, output_value) in output_row.iter_mut().enumerate() {
            let start = output_index * in_features;
            let weight_row = &weight[start..start + in_features];

            *output_value = input_row.iter().zip(weight_row).fold(
                0.0_f32,
                |sum, (&input_value, &weight_value)| {
                    round_to_bfloat16(input_value).mul_add(round_to_bfloat16(weight_value), sum)
                },
            );
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{linear, round_to_bfloat16};

    #[test]
    fn bfloat16_rounding_uses_round_to_nearest_even() {
        assert_eq!(round_to_bfloat16(1.0), 1.0);
        assert_eq!(round_to_bfloat16(-2.0), -2.0);
        assert_eq!(round_to_bfloat16(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn projection_uses_out_features_by_in_features_layout() {
        let input = [1.0, 2.0];
        let weight = [
            3.0, 4.0, //
            5.0, 6.0,
        ];
        let output = linear(&input, 1, 2, &weight, 2).expect("valid projection");
        assert_eq!(output, vec![11.0, 17.0]);
    }

    #[test]
    fn rejects_incorrect_weight_length() {
        assert!(linear(&[1.0, 2.0], 1, 2, &[1.0, 2.0, 3.0], 2).is_err());
    }
}
