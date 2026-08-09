//! Bias-free linear projections using exported `[out, in]` weights.

use rayon::prelude::*;

use crate::error::Result;
use crate::tensor::{checked_matrix_len, validate_matrix};

/// Round an FP32 value to the nearest representable bfloat16 value and return
/// it as FP32.
///
/// This mirrors the JAX transformer projection path, which converts
/// activations and weights to bfloat16 before dot products.
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

/// Apply a bias-free transformer projection with BF16-equivalent operands and
/// FP32 accumulation.
///
/// `input` has shape `[rows, in_features]`. `weight` follows the exported
/// layout `[out_features, in_features]`. The returned vector has shape
/// `[rows, out_features]`.
///
/// Output elements are independent, so release inference distributes them
/// across Rayon's CPU worker pool. Inputs are rounded once per projection and
/// reused by every output dot product instead of repeating the same conversion
/// for each output neuron. Weights are rounded at multiplication time. The
/// accumulation order inside each dot product is unchanged.
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

    let rounded_input = input
        .iter()
        .copied()
        .map(round_to_bfloat16)
        .collect::<Vec<_>>();
    let output_len = checked_matrix_len("linear output", rows, out_features)?;
    let mut output = vec![0.0_f32; output_len];

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_index, output_value)| {
            let row_index = flat_index / out_features;
            let output_index = flat_index % out_features;
            let input_start = row_index * in_features;
            let input_row = &rounded_input[input_start..input_start + in_features];
            let weight_start = output_index * in_features;
            let weight_row = &weight[weight_start..weight_start + in_features];

            *output_value = input_row.iter().zip(weight_row).fold(
                0.0_f32,
                |sum, (&input_value, &weight_value)| {
                    input_value.mul_add(round_to_bfloat16(weight_value), sum)
                },
            );
        });

    Ok(output)
}

/// Apply a bias-free FP32 projection without BF16 operand rounding.
///
/// This is used for the tied vocabulary projection because the JAX reference
/// computes its final `hidden @ embedding.T` einsum with FP32 operands and FP32
/// accumulation, unlike the transformer projection matrices.
///
/// # Errors
///
/// Returns an invalid-input error when any dimension or flat slice length is
/// inconsistent.
pub fn linear_fp32(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    validate_matrix("FP32 linear input", input, rows, in_features)?;
    validate_matrix("FP32 linear weight", weight, out_features, in_features)?;

    let output_len = checked_matrix_len("FP32 linear output", rows, out_features)?;
    let mut output = vec![0.0_f32; output_len];

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_index, output_value)| {
            let row_index = flat_index / out_features;
            let output_index = flat_index % out_features;
            let input_start = row_index * in_features;
            let input_row = &input[input_start..input_start + in_features];
            let weight_start = output_index * in_features;
            let weight_row = &weight[weight_start..weight_start + in_features];

            *output_value = input_row
                .iter()
                .zip(weight_row)
                .fold(0.0_f32, |sum, (&input_value, &weight_value)| {
                    input_value.mul_add(weight_value, sum)
                });
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{linear, linear_fp32, round_to_bfloat16};

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
    fn projection_reuses_bfloat16_rounded_inputs_without_changing_results() {
        let input = [1.003_906_2_f32, -0.996_093_75_f32];
        let weight = [
            0.333_984_38_f32,
            0.667_968_75_f32, //
            -0.25_f32,
            0.125_f32,
        ];
        let output = linear(&input, 1, 2, &weight, 2).expect("valid projection");

        let expected = weight
            .chunks_exact(2)
            .map(|row| {
                let mut sum = 0.0_f32;
                for index in 0..2 {
                    sum =
                        round_to_bfloat16(input[index]).mul_add(round_to_bfloat16(row[index]), sum);
                }
                sum
            })
            .collect::<Vec<_>>();

        assert_eq!(output, expected);
    }

    #[test]
    fn fp32_projection_does_not_apply_bfloat16_rounding() {
        let input = [1.003_906_2_f32, -0.996_093_75_f32];
        let weight = [0.333_984_38_f32, 0.667_968_75_f32];
        let output = linear_fp32(&input, 1, 2, &weight, 1).expect("valid FP32 projection");

        let expected = input[0].mul_add(weight[0], input[1] * weight[1]);
        let bf16 = linear(&input, 1, 2, &weight, 1).expect("valid BF16 projection");

        assert_eq!(output, vec![expected]);
        assert_ne!(output, bf16);
    }

    #[test]
    fn rejects_incorrect_weight_length() {
        assert!(linear(&[1.0, 2.0], 1, 2, &[1.0, 2.0, 3.0], 2).is_err());
        assert!(linear_fp32(&[1.0, 2.0], 1, 2, &[1.0, 2.0, 3.0], 2).is_err());
    }
}
