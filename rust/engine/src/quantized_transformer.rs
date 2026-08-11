//! Pre-norm transformer execution backed by INT8 projection weights.

use crate::config::ModelConfig;
use crate::error::Result;
use crate::kv_cache::KvCache;
use crate::quantized_attention::{QuantizedAttentionWeights, QuantizedGroupedQueryAttention};
use crate::quantized_situ_glu::QuantizedSituGlu;
use crate::quantized_weights::{QuantizedMatrix, QuantizedModelWeights};
use crate::rmsnorm::RmsNorm;
use crate::transformer::add_residual_in_place;

/// Borrowed mixed-precision tensors required by one quantized transformer block.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedTransformerBlockWeights<'a> {
    attention_norm: &'a [f32],
    query: &'a QuantizedMatrix,
    key: &'a QuantizedMatrix,
    value: &'a QuantizedMatrix,
    output: &'a QuantizedMatrix,
    ffn_norm: &'a [f32],
    gate: &'a QuantizedMatrix,
    up: &'a QuantizedMatrix,
    down: &'a QuantizedMatrix,
}

impl<'a> QuantizedTransformerBlockWeights<'a> {
    /// Borrow one layer from validated INT8 model weights.
    ///
    /// # Errors
    ///
    /// Returns an invalid-weight error when a required tensor is absent.
    pub fn from_model_weights(
        weights: &'a QuantizedModelWeights,
        layer_index: usize,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer_index}");
        Ok(Self {
            attention_norm: weights
                .required_float(&format!("{prefix}.attention_norm.weight"))?
                .values(),
            query: weights.required_projection(&format!("{prefix}.q_proj.weight"))?,
            key: weights.required_projection(&format!("{prefix}.k_proj.weight"))?,
            value: weights.required_projection(&format!("{prefix}.v_proj.weight"))?,
            output: weights.required_projection(&format!("{prefix}.o_proj.weight"))?,
            ffn_norm: weights
                .required_float(&format!("{prefix}.ffn_norm.weight"))?
                .values(),
            gate: weights.required_projection(&format!("{prefix}.gate_proj.weight"))?,
            up: weights.required_projection(&format!("{prefix}.up_proj.weight"))?,
            down: weights.required_projection(&format!("{prefix}.down_proj.weight"))?,
        })
    }
}

/// Executable metadata for one quantized SmallLM transformer block.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedTransformerBlock {
    hidden_size: usize,
    attention_norm: RmsNorm,
    attention: QuantizedGroupedQueryAttention,
    ffn_norm: RmsNorm,
    ffn: QuantizedSituGlu,
}

impl QuantizedTransformerBlock {
    /// Build block operations from validated model configuration.
    ///
    /// # Errors
    ///
    /// Returns invalid model-configuration errors.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            hidden_size: config.hidden_size,
            attention_norm: RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)?,
            attention: QuantizedGroupedQueryAttention::from_config(config)?,
            ffn_norm: RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)?,
            ffn: QuantizedSituGlu::from_config(config)?,
        })
    }

    fn ffn_residual(
        &self,
        hidden_states: &[f32],
        rows: usize,
        weights: QuantizedTransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        let normalized = self
            .ffn_norm
            .forward_rows(hidden_states, rows, weights.ffn_norm)?;
        let update = self
            .ffn
            .forward(&normalized, weights.gate, weights.up, weights.down)?;
        let mut output = hidden_states.to_vec();
        add_residual_in_place(&mut output, &update)?;
        Ok(output)
    }

    /// Run prompt rows while populating one empty KV-cache layer.
    ///
    /// # Errors
    ///
    /// Returns norm, quantized-attention, cache, feed-forward, or residual errors.
    pub fn forward_prefill_with_cache(
        &self,
        hidden_states: &[f32],
        rows: usize,
        layer_index: usize,
        cache: &mut KvCache,
        weights: QuantizedTransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        let normalized = self
            .attention_norm
            .forward_rows(hidden_states, rows, weights.attention_norm)?;
        let attention_weights =
            QuantizedAttentionWeights::new(weights.query, weights.key, weights.value, weights.output);
        let attention_output = self.attention.forward_prefill_with_cache(
            &normalized,
            rows,
            layer_index,
            cache,
            attention_weights,
        )?;
        let mut after_attention = hidden_states.to_vec();
        add_residual_in_place(&mut after_attention, &attention_output)?;
        self.ffn_residual(&after_attention, rows, weights)
    }

    /// Run one cached decode token through this block.
    ///
    /// # Errors
    ///
    /// Returns norm, quantized-attention, cache, feed-forward, or residual errors.
    pub fn forward_cached_token(
        &self,
        hidden_state: &[f32],
        layer_index: usize,
        cache: &mut KvCache,
        weights: QuantizedTransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        let normalized = self
            .attention_norm
            .forward_rows(hidden_state, 1, weights.attention_norm)?;
        let attention_weights =
            QuantizedAttentionWeights::new(weights.query, weights.key, weights.value, weights.output);
        let attention_output = self.attention.forward_cached_token(
            &normalized,
            layer_index,
            cache,
            attention_weights,
        )?;
        let mut after_attention = hidden_state.to_vec();
        add_residual_in_place(&mut after_attention, &attention_output)?;
        self.ffn_residual(&after_attention, 1, weights)
    }

    /// Return hidden width for one row.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::QuantizedTransformerBlock;
    use crate::config::ModelConfig;

    #[test]
    fn quantized_block_builds_from_configuration() {
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
        let block = QuantizedTransformerBlock::from_config(&config).expect("valid block");
        assert_eq!(block.hidden_size(), 4);
    }
}
