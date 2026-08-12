//! Weight-only INT8 projection with runtime-dispatched AVX2 acceleration.

use rayon::prelude::*;

use crate::avx2::{dequantize_bf16_row, dot_bf16_int8};
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
/// On x86_64 CPUs with AVX2, dequantization processes eight INT8 weights at a
/// time. Multi-row prefill dequantizes each projection matrix once and reuses
/// it across input rows. Single-row cached decode keeps an allocation-free
/// ordered dot-product path. Other CPUs retain the scalar fallback.
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

    if rows > 1 {
        let mut dequantized_weight = vec![0.0_f32; expected_weights];
        dequantized_weight
            .par_chunks_mut(in_features)
            .enumerate()
            .for_each(|(output_index, destination)| {
                let start = output_index * in_features;
                let source = &weight.values()[start..start + in_features];
                dequantize_bf16_row(source, weight.scales()[output_index], destination);
            });

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(flat_index, output_value)| {
                let row_index = flat_index / out_features;
                let output_index = flat_index % out_features;
                let input_start = row_index * in_features;
                let input_row = &rounded_input[input_start..input_start + in_features];
                let weight_start = output_index * in_features;
                let weight_row =
                    &dequantized_weight[weight_start..weight_start + in_features];

                *output_value = input_row
                    .iter()
                    .zip(weight_row)
                    .fold(0.0_f32, |sum, (&input_value, &weight_value)| {
                        input_value.mul_add(weight_value, sum)
                    });
            });
    } else {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(output_index, output_value)| {
                let weight_start = output_index * in_features;
                let weight_row = &weight.values()[weight_start..weight_start + in_features];
                *output_value = dot_bf16_int8(
                    &rounded_input[..in_features],
                    weight_row,
                    weight.scales()[output_index],
                );
            });
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::linear_int8;
    use crate::linear::round_to_bfloat16;
    use crate::quantized_weights::QuantizedMatrix;

    fn matrix(shape: [usize; 2], values: Vec<i8>, scales: Vec<f32>) -> QuantizedMatrix {
        QuantizedMatrix::from_parts(shape.to_vec(), values, scales).expect("valid INT8 matrix")
    }

    #[test]
    fn projection_applies_one_scale_per_output_row() {
        let weight = matrix(
            [2, 2],
            vec![127, -127, 64, 32],
            vec![1.0 / 127.0, 0.5 / 64.0],
        );
        let output = linear_int8(&[2.0, 1.0], 1, 2, &weight, 2).expect("valid projection");

        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output[0] > 0.9 && output[0] < 1.1);
        assert!(output[1] > 1.20 && output[1] < 1.30);
    }

    #[test]
    fn multirow_projection_matches_scalar_reference() {
        let input = [
            1.003_906_2_f32,
            -0.996_093_75_f32,
            0.503_906_25_f32,
            1.996_093_8_f32,
        ];
        let values = vec![127_i8, -64, 32, 11];
        let scales = vec![0.003_75_f32, 0.007_5_f32];
        let weight = matrix([2, 2], values.clone(), scales.clone());
        let actual = linear_int8(&input, 2, 2, &weight, 2).expect("valid projection");

        let rounded_input = input
            .iter()
            .copied()
            .map(round_to_bfloat16)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for input_row in rounded_input.chunks_exact(2) {
            for output_index in 0..2 {
                let weight_row = &values[output_index * 2..output_index * 2 + 2];
                let sum = input_row.iter().zip(weight_row).fold(
                    0.0_f32,
                    |sum, (&input_value, &quantized)| {
                        input_value.mul_add(
                            round_to_bfloat16(f32::from(quantized) * scales[output_index]),
                            sum,
                        )
                    },
                );
                expected.push(sum);
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn projection_rejects_shape_mismatch() {
        let weight = matrix([1, 2], vec![1, 2], vec![1.0]);
        assert!(linear_int8(&[1.0, 2.0], 1, 2, &weight, 2).is_err());
    }
}
