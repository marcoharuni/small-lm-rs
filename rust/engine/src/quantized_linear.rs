//! Reference weight-only INT8 projection with FP32 accumulation.

use rayon::prelude::*;

use crate::error::{EngineError, Result};
use crate::linear::round_to_bfloat16;
use crate::quantized_weights::QuantizedMatrix;
use crate::tensor::{checked_matrix_len, validate_matrix};

/// Apply one per-output-channel INT8 projection.
///
/// Activations are rounded to BF16-equivalent values to match the transformer
/// compute contract. Each quantized weight is dequantized with its output-row
/// scale, rounded to BF16-equivalent precision, and accumulated in FP32.
///
/// This is the correctness/reference path. Dedicated SIMD kernels are a later
/// optimization and must preserve the same numerical contract.
///
/// # Errors
///
/// Returns an invalid-input error for inconsistent matrix dimensions, scales,
/// or quantized weight storage.
pub fn linear_int8(
    input: &[f32],
    rows: usize,
    in_features: usize,
    weight: &QuantizedMatrix,
    out_features: usize,
) -> Result<Vec<f32>> {
    validate_matrix("INT8 linear input", input, rows, in_features)?;
    if weight.shape() != [out_features, in_features] {
        return Err(EngineError::invalid_input(
            "INT8 linear weight",
            format!(
                "weight has shape {:?}, expected [{out_features}, {in_features}]",
                weight.shape()
            ),
        ));
    }
    let expected_weights = checked_matrix_len("INT8 linear weight", out_features, in_features)?;
    if weight.values().len() != expected_weights {
        return Err(EngineError::invalid_input(
            "INT8 linear weight",
            format!(
                "weight contains {} values, expected {expected_weights}",
                weight.values().len()
            ),
        ));
    }
    if weight.scales().len() != out_features {
        return Err(EngineError::invalid_input(
            "INT8 linear scales",
            format!(
                "received {} scales, expected {out_features}",
                weight.scales().len()
            ),
        ));
    }
    if weight
        .scales()
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(EngineError::invalid_input(
            "INT8 linear scales",
            "all scales must be finite and positive",
        ));
    }

    let rounded_input = input
        .iter()
        .copied()
        .map(round_to_bfloat16)
        .collect::<Vec<_>>();
    let output_len = checked_matrix_len("INT8 linear output", rows, out_features)?;
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
            let weight_row = &weight.values()[weight_start..weight_start + in_features];
            let scale = weight.scales()[output_index];

            *output_value = input_row.iter().zip(weight_row).fold(
                0.0_f32,
                |sum, (&input_value, &quantized_weight)| {
                    let dequantized = round_to_bfloat16(f32::from(quantized_weight) * scale);
                    input_value.mul_add(dequantized, sum)
                },
            );
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::linear_int8;
    use crate::quantized_weights::QuantizedMatrix;

    fn matrix(shape: [usize; 2], values: Vec<i8>, scales: Vec<f32>) -> QuantizedMatrix {
        QuantizedMatrix::from_parts_for_test(shape.to_vec(), values, scales)
    }

    #[test]
    fn projection_applies_one_scale_per_output_row() {
        let weight = matrix([2, 2], vec![127, -127, 64, 32], vec![1.0 / 127.0, 0.5 / 64.0]);
        let output = linear_int8(&[2.0, 1.0], 1, 2, &weight, 2).expect("valid projection");

        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output[0] > 0.9 && output[0] < 1.1);
        assert!(output[1] > 1.20 && output[1] < 1.30);
    }

    #[test]
    fn projection_validates_shape_and_scale_count() {
        let bad_shape = matrix([1, 2], vec![1, 2], vec![1.0]);
        assert!(linear_int8(&[1.0, 2.0], 1, 2, &bad_shape, 2).is_err());

        let bad_scales = matrix([2, 2], vec![1, 2, 3, 4], vec![1.0]);
        assert!(linear_int8(&[1.0, 2.0], 1, 2, &bad_scales, 2).is_err());
    }
}
