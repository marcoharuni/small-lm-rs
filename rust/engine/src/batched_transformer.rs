//! Batched cached transformer-block execution for active decode requests.

use crate::batched_attention::forward_cached_batch;
use crate::config::ModelConfig;
use crate::error::Result;
use crate::kv_cache::KvCache;
use crate::rmsnorm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::situ_glu::SituGlu;
use crate::transformer::{add_residual_in_place, TransformerBlockWeights};
use crate::attention::{AttentionWeights, GroupedQueryAttention};

/// Batched form of one pre-norm SmallLM transformer block.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchedTransformerBlock {
    hidden_size: usize,
    attention_norm: RmsNorm,
    attention: GroupedQueryAttention,
    rotary: RotaryEmbedding,
    ffn_norm: RmsNorm,
    ffn: SituGlu,
}

impl BatchedTransformerBlock {
    /// Build batched block operations from validated model configuration.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when model invariants fail.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            hidden_size: config.hidden_size,
            attention_norm: RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)?,
            attention: GroupedQueryAttention::from_config(config)?,
            rotary: RotaryEmbedding::new(config.head_dimension, config.rope_theta)?,
            ffn_norm: RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)?,
            ffn: SituGlu::from_config(config)?,
        })
    }

    /// Run one cached decode token for every request in the batch.
    ///
    /// Normalization, Q/K/V projection, attention output projection, and all
    /// feed-forward projections operate on the complete row batch. Attention
    /// reductions and KV writes remain request-local.
    ///
    /// # Errors
    ///
    /// Propagates norm, projection, attention, cache, residual, and
    /// feed-forward errors.
    pub fn forward_cached_batch(
        &self,
        hidden_states: &[f32],
        layer_index: usize,
        caches: &mut [&mut KvCache],
        weights: TransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        let batch_size = caches.len();
        let normalized = self.attention_norm.forward_rows(
            hidden_states,
            batch_size,
            weights.attention_norm(),
        )?;
        let attention_weights = AttentionWeights::new(
            weights.query(),
            weights.key(),
            weights.value(),
            weights.output(),
        );
        let attention_output = forward_cached_batch(
            &self.attention,
            &normalized,
            layer_index,
            caches,
            attention_weights,
            &self.rotary,
        )?;
        let mut after_attention = hidden_states.to_vec();
        add_residual_in_place(&mut after_attention, &attention_output)?;

        let normalized_ffn = self.ffn_norm.forward_rows(
            &after_attention,
            batch_size,
            weights.ffn_norm(),
        )?;
        let ffn_output = self.ffn.forward(
            &normalized_ffn,
            weights.gate(),
            weights.up(),
            weights.down(),
        )?;
        add_residual_in_place(&mut after_attention, &ffn_output)?;
        Ok(after_attention)
    }

    /// Return the hidden width consumed by each batch row.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::BatchedTransformerBlock;
    use crate::config::ModelConfig;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
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
        }
    }

    #[test]
    fn block_uses_model_hidden_width() {
        let block = BatchedTransformerBlock::from_config(&tiny_config()).expect("valid block");
        assert_eq!(block.hidden_size(), 4);
    }
}
