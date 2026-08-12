//! x86_64 AVX2 helpers for weight-only INT8 execution.
//!
//! Unsafe code is intentionally isolated in this module. Public engine paths
//! runtime-detect AVX2 before calling target-feature functions.

#![allow(unsafe_code)]

use crate::linear::round_to_bfloat16;

#[must_use]
pub(crate) fn is_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub(crate) fn dequantize_bf16_row(values: &[i8], scale: f32, output: &mut [f32]) {
    debug_assert_eq!(values.len(), output.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_available() {
            // SAFETY: AVX2 support is checked immediately above. Input/output
            // slices are equal-length and the implementation bounds-checks all
            // vector loads/stores before using raw pointers.
            unsafe {
                dequantize_bf16_row_avx2(values, scale, output);
            }
            return;
        }
    }

    for (destination, &quantized) in output.iter_mut().zip(values) {
        *destination = round_to_bfloat16(f32::from(quantized) * scale);
    }
}

pub(crate) fn dot_bf16_int8(input: &[f32], values: &[i8], scale: f32) -> f32 {
    debug_assert_eq!(input.len(), values.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_available() {
            // SAFETY: AVX2 support is checked immediately above. The function
            // only reads within the provided equal-length slices.
            unsafe {
                return dot_bf16_int8_avx2(input, values, scale);
            }
        }
    }

    input
        .iter()
        .zip(values)
        .fold(0.0_f32, |sum, (&input_value, &quantized)| {
            input_value.mul_add(round_to_bfloat16(f32::from(quantized) * scale), sum)
        })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_bf16_row_avx2(values: &[i8], scale: f32, output: &mut [f32]) {
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_storeu_ps, _mm_loadl_epi64,
    };

    let scale_vector = _mm256_set1_ps(scale);
    let mut index = 0;
    let mut lanes = [0.0_f32; 8];

    while index + 8 <= values.len() {
        let packed = unsafe { _mm_loadl_epi64(values.as_ptr().add(index).cast::<__m128i>()) };
        let integers = _mm256_cvtepi8_epi32(packed);
        let floats = _mm256_cvtepi32_ps(integers);
        let scaled = _mm256_mul_ps(floats, scale_vector);
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), scaled) };

        for lane in 0..8 {
            output[index + lane] = round_to_bfloat16(lanes[lane]);
        }
        index += 8;
    }

    for tail in index..values.len() {
        output[tail] = round_to_bfloat16(f32::from(values[tail]) * scale);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_bf16_int8_avx2(input: &[f32], values: &[i8], scale: f32) -> f32 {
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_storeu_ps, _mm_loadl_epi64,
    };

    let scale_vector = _mm256_set1_ps(scale);
    let mut index = 0;
    let mut sum = 0.0_f32;
    let mut lanes = [0.0_f32; 8];

    while index + 8 <= values.len() {
        let packed = unsafe { _mm_loadl_epi64(values.as_ptr().add(index).cast::<__m128i>()) };
        let integers = _mm256_cvtepi8_epi32(packed);
        let floats = _mm256_cvtepi32_ps(integers);
        let scaled = _mm256_mul_ps(floats, scale_vector);
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), scaled) };

        for lane in 0..8 {
            sum = input[index + lane].mul_add(round_to_bfloat16(lanes[lane]), sum);
        }
        index += 8;
    }

    for tail in index..values.len() {
        sum = input[tail].mul_add(round_to_bfloat16(f32::from(values[tail]) * scale), sum);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::{dequantize_bf16_row, dot_bf16_int8};
    use crate::linear::round_to_bfloat16;

    #[test]
    fn dequantization_matches_scalar_reference() {
        let values = [-127_i8, -64, -1, 0, 1, 32, 64, 127, 11];
        let scale = 0.003_75_f32;
        let mut actual = vec![0.0_f32; values.len()];
        dequantize_bf16_row(&values, scale, &mut actual);
        let expected = values
            .iter()
            .map(|&value| round_to_bfloat16(f32::from(value) * scale))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn dot_matches_scalar_accumulation_order() {
        let input = [1.0_f32, -2.0, 0.5, 3.0, -4.0, 0.25, 2.5, -1.5, 0.75];
        let values = [-127_i8, -64, -1, 0, 1, 32, 64, 127, 11];
        let scale = 0.003_75_f32;
        let expected = input.iter().zip(values).fold(0.0_f32, |sum, (&x, q)| {
            x.mul_add(round_to_bfloat16(f32::from(q) * scale), sum)
        });
        assert_eq!(dot_bf16_int8(&input, &values, scale), expected);
    }
}
