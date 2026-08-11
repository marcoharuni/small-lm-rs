//! Grouped-query attention backed by weight-only INT8 projections.

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::kv_cache::KvCache;
use crate::quantized_linear::linear_int8;
use crate::quantized_weights::QuantizedMatrix;
use crate::rope::RotaryEmbedding;
use crate::softmax::softmax_rows;
use crate::tensor::validate_matrix;

/// Borrowed INT8 projection weights for one attention layer.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedAttentionWeights<'a> {
    query: &'a QuantizedMatrix,
    key: &'a QuantizedMatrix,
    value: &'a QuantizedMatrix,
    output: &'a QuantizedMatrix,
}

impl<'a> QuantizedAttentionWeights<'a> {
    /// Bundle query, key, value, and output projection matrices.
    #[must_use]
    pub const fn new(
        query: &'a QuantizedMatrix,
        key: &'a QuantizedMatrix,
        value: &'a QuantizedMatrix,
        output: &'a QuantizedMatrix,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
        }
    }
}

/// Quantized grouped-query causal self-attention.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedGroupedQueryAttention {
    hidden_size: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dimension: usize,
    rotary: RotaryEmbedding,
}

impl QuantizedGroupedQueryAttention {
    /// Build attention metadata from model configuration.
    ///
    /// # Errors
    ///
    /// Returns invalid model-configuration errors.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            hidden_size: config.hidden_size,
            num_query_heads: config.num_query_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dimension: config.head_dimension,
            rotary: RotaryEmbedding::new(config.head_dimension, config.rope_theta)?,
        })
    }

    fn kv_width(&self) -> Result<usize> {
        self.num_key_value_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| {
                EngineError::invalid_input("quantized attention", "KV width overflows usize")
            })
    }

    fn group_size(&self) -> usize {
        self.num_query_heads / self.num_key_value_heads
    }

    fn project_qkv(
        &self,
        hidden_states: &[f32],
        rows: usize,
        weights: QuantizedAttentionWeights<'_>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        validate_matrix(
            "quantized attention hidden states",
            hidden_states,
            rows,
            self.hidden_size,
        )?;
        let kv_width = self.kv_width()?;
        let query = linear_int8(
            hidden_states,
            rows,
            self.hidden_size,
            weights.query,
            self.hidden_size,
        )?;
        let key = linear_int8(hidden_states, rows, self.hidden_size, weights.key, kv_width)?;
        let value = linear_int8(
            hidden_states,
            rows,
            self.hidden_size,
            weights.value,
            kv_width,
        )?;
        Ok((query, key, value))
    }

    fn apply_rotary(
        &self,
        query: &mut [f32],
        key: &mut [f32],
        rows: usize,
        position_offset: usize,
    ) -> Result<()> {
        let query_stride = self.hidden_size;
        let key_stride = self.kv_width()?;
        validate_matrix("quantized attention query", query, rows, query_stride)?;
        validate_matrix("quantized attention key", key, rows, key_stride)?;

        for row in 0..rows {
            let position = position_offset.checked_add(row).ok_or_else(|| {
                EngineError::invalid_input("quantized attention RoPE", "position overflows usize")
            })?;
            let query_start = row * query_stride;
            let key_start = row * key_stride;
            self.rotary.apply_heads_in_place(
                &mut query[query_start..query_start + query_stride],
                self.num_query_heads,
                position,
            )?;
            self.rotary.apply_heads_in_place(
                &mut key[key_start..key_start + key_stride],
                self.num_key_value_heads,
                position,
            )?;
        }
        Ok(())
    }

    fn attend_prefill(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let kv_width = self.kv_width()?;
        validate_matrix("quantized attention query", query, rows, self.hidden_size)?;
        validate_matrix("quantized attention key", key, rows, kv_width)?;
        validate_matrix("quantized attention value", value, rows, kv_width)?;

        let score_rows = self.num_query_heads.checked_mul(rows).ok_or_else(|| {
            EngineError::invalid_input("quantized attention", "score row count overflows usize")
        })?;
        let score_count = score_rows.checked_mul(rows).ok_or_else(|| {
            EngineError::invalid_input("quantized attention", "score count overflows usize")
        })?;
        let mut probabilities = vec![f32::MIN; score_count];
        let scale = (self.head_dimension as f32).sqrt().recip();

        for query_head in 0..self.num_query_heads {
            let kv_head = query_head / self.group_size();
            for query_position in 0..rows {
                let probability_row = (query_head * rows + query_position) * rows;
                let query_start =
                    query_position * self.hidden_size + query_head * self.head_dimension;
                let query_values = &query[query_start..query_start + self.head_dimension];
                for source_position in 0..=query_position {
                    let key_start = source_position * kv_width + kv_head * self.head_dimension;
                    let key_values = &key[key_start..key_start + self.head_dimension];
                    let dot = query_values
                        .iter()
                        .zip(key_values)
                        .fold(0.0_f32, |sum, (&q, &k)| q.mul_add(k, sum));
                    probabilities[probability_row + source_position] = dot * scale;
                }
            }
        }
        softmax_rows(&mut probabilities, score_rows, rows)?;

        let mut attended = vec![0.0_f32; rows * self.hidden_size];
        for query_position in 0..rows {
            for query_head in 0..self.num_query_heads {
                let kv_head = query_head / self.group_size();
                let probability_row = (query_head * rows + query_position) * rows;
                let output_start =
                    query_position * self.hidden_size + query_head * self.head_dimension;
                let output_head = &mut attended[output_start..output_start + self.head_dimension];
                for source_position in 0..rows {
                    let probability = probabilities[probability_row + source_position];
                    if probability == 0.0 {
                        continue;
                    }
                    let value_start = source_position * kv_width + kv_head * self.head_dimension;
                    let value_head = &value[value_start..value_start + self.head_dimension];
                    for (output, &source) in output_head.iter_mut().zip(value_head) {
                        *output = source.mul_add(probability, *output);
                    }
                }
            }
        }
        Ok(attended)
    }

    fn attend_cached_query(
        &self,
        query: &[f32],
        layer_index: usize,
        cache: &KvCache,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "quantized cached attention query",
            query,
            self.num_query_heads,
            self.head_dimension,
        )?;
        let source_length = cache.layer_sequence_length(layer_index)?;
        if source_length == 0 {
            return Err(EngineError::invalid_input(
                "quantized cached attention",
                "at least one cached position is required",
            ));
        }

        let mut probabilities = vec![0.0_f32; self.num_query_heads * source_length];
        let scale = (self.head_dimension as f32).sqrt().recip();
        for query_head in 0..self.num_query_heads {
            let kv_head = query_head / self.group_size();
            let query_start = query_head * self.head_dimension;
            let query_values = &query[query_start..query_start + self.head_dimension];
            let row =
                &mut probabilities[query_head * source_length..(query_head + 1) * source_length];
            for (source_position, score) in row.iter_mut().enumerate() {
                let key = cache.key_head(layer_index, source_position, kv_head)?;
                let dot = query_values
                    .iter()
                    .zip(key)
                    .fold(0.0_f32, |sum, (&q, &k)| q.mul_add(k, sum));
                *score = dot * scale;
            }
        }
        softmax_rows(&mut probabilities, self.num_query_heads, source_length)?;

        let mut attended = vec![0.0_f32; self.hidden_size];
        for query_head in 0..self.num_query_heads {
            let kv_head = query_head / self.group_size();
            let output_start = query_head * self.head_dimension;
            let output_head = &mut attended[output_start..output_start + self.head_dimension];
            let row = &probabilities[query_head * source_length..(query_head + 1) * source_length];
            for (source_position, &probability) in row.iter().enumerate() {
                let value = cache.value_head(layer_index, source_position, kv_head)?;
                for (output, &source) in output_head.iter_mut().zip(value) {
                    *output = source.mul_add(probability, *output);
                }
            }
        }
        Ok(attended)
    }

    /// Run fresh-sequence attention and populate one empty KV-cache layer.
    ///
    /// # Errors
    ///
    /// Returns projection, RoPE, cache, softmax, or shape errors.
    pub fn forward_prefill_with_cache(
        &self,
        hidden_states: &[f32],
        rows: usize,
        layer_index: usize,
        cache: &mut KvCache,
        weights: QuantizedAttentionWeights<'_>,
    ) -> Result<Vec<f32>> {
        if cache.layer_sequence_length(layer_index)? != 0 {
            return Err(EngineError::invalid_input(
                "quantized attention prefill",
                "cache layer must be empty before prefill",
            ));
        }
        let (mut query, mut key, value) = self.project_qkv(hidden_states, rows, weights)?;
        self.apply_rotary(&mut query, &mut key, rows, 0)?;
        let attended = self.attend_prefill(&query, &key, &value, rows)?;
        let output = linear_int8(
            &attended,
            rows,
            self.hidden_size,
            weights.output,
            self.hidden_size,
        )?;
        cache.append_layer(layer_index, &key, &value, rows)?;
        Ok(output)
    }

    /// Run one cached decode token and append its key/value state.
    ///
    /// # Errors
    ///
    /// Returns projection, RoPE, cache, softmax, or shape errors. Cache state
    /// is rolled back if attention/output projection fails after append.
    pub fn forward_cached_token(
        &self,
        hidden_state: &[f32],
        layer_index: usize,
        cache: &mut KvCache,
        weights: QuantizedAttentionWeights<'_>,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "quantized cached hidden state",
            hidden_state,
            1,
            self.hidden_size,
        )?;
        let original_length = cache.layer_sequence_length(layer_index)?;
        let (mut query, mut key, value) = self.project_qkv(hidden_state, 1, weights)?;
        self.apply_rotary(&mut query, &mut key, 1, original_length)?;
        cache.append_layer(layer_index, &key, &value, 1)?;

        let result = (|| {
            let attended = self.attend_cached_query(&query, layer_index, cache)?;
            linear_int8(
                &attended,
                1,
                self.hidden_size,
                weights.output,
                self.hidden_size,
            )
        })();

        match result {
            Ok(output) => Ok(output),
            Err(error) => {
                cache.truncate_layer(layer_index, original_length)?;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QuantizedGroupedQueryAttention;
    use crate::config::ModelConfig;

    #[test]
    fn quantized_attention_builds_from_configuration() {
        let config = ModelConfig {
            model_name: "tiny".to_owned(),
            architecture: "decoder-only-transformer".to_owned(),
            normalization: "RMSNorm".to_owned(),
            position_encoding: "RoPE".to_owned(),
            activation: "SiTU-GLU".to_owned(),
            weight_layout: "out_features,in_features".to_owned(),
            vocab_size: 8,
            context_length: 8,
            num_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_query_heads: 2,
            num_key_value_heads: 1,
            head_dimension: 2,
            rms_norm_epsilon: 1.0e-5,
            rope_theta: 10_000.0,
            situ_beta_gate: 4.0,
            situ_beta_up: 25.0,
            tie_word_embeddings: true,
            use_bias: false,
            dropout: 0.0,
            expected_parameter_count: 1,
        };
        let attention =
            QuantizedGroupedQueryAttention::from_config(&config).expect("valid attention");
        assert_eq!(attention.hidden_size, 4);
    }
}
