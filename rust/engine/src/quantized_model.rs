//! Full-model execution using mixed FP32/INT8 model weights.

use std::path::Path;

use crate::config::ModelConfig;
use crate::embedding::embedding_lookup;
use crate::error::{EngineError, Result};
use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::linear::linear_fp32;
use crate::quantized_transformer::{
    QuantizedTransformerBlock, QuantizedTransformerBlockWeights,
};
use crate::quantized_weights::QuantizedModelWeights;
use crate::rmsnorm::RmsNorm;

/// Decoder-only model backed by weight-only INT8 transformer projections.
#[derive(Debug)]
pub struct QuantizedDecodeModel {
    config: ModelConfig,
    blocks: Vec<QuantizedTransformerBlock>,
    final_norm: RmsNorm,
    weights: Option<QuantizedModelWeights>,
}

impl QuantizedDecodeModel {
    /// Construct an unweighted INT8 model from validated configuration.
    ///
    /// # Errors
    ///
    /// Returns invalid model-configuration errors.
    pub fn from_config(config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let blocks = (0..config.num_layers)
            .map(|_| QuantizedTransformerBlock::from_config(&config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)?;
        Ok(Self {
            config,
            blocks,
            final_norm,
            weights: None,
        })
    }

    /// Return architecture configuration.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Return whether quantized weights are loaded.
    #[must_use]
    pub const fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Load one validated mixed FP32/INT8 SafeTensors artifact.
    ///
    /// # Errors
    ///
    /// Returns artifact or weight-layout validation errors.
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.weights = Some(QuantizedModelWeights::load(path, &self.config)?);
        Ok(())
    }

    /// Allocate one request-local dense KV cache.
    ///
    /// # Errors
    ///
    /// Returns invalid cache-capacity errors.
    pub fn allocate_kv_cache(&self, max_sequence_length: usize) -> Result<KvCache> {
        KvCache::allocate(KvCacheConfig::from_model(&self.config, max_sequence_length)?)
    }

    fn loaded_weights(&self) -> Result<&QuantizedModelWeights> {
        self.weights.as_ref().ok_or_else(|| {
            EngineError::invalid_input("INT8 model execution", "quantized weights are not loaded")
        })
    }

    fn validate_token_ids(&self, token_ids: &[u32]) -> Result<()> {
        if token_ids.is_empty() {
            return Err(EngineError::invalid_input(
                "INT8 model forward",
                "at least one token is required",
            ));
        }
        if let Some(token_id) = token_ids
            .iter()
            .copied()
            .find(|token_id| *token_id as usize >= self.config.vocab_size)
        {
            return Err(EngineError::invalid_input(
                "INT8 model forward",
                format!("token id {token_id} is outside the vocabulary"),
            ));
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &KvCache) -> Result<()> {
        let config = cache.config();
        if config.num_layers != self.config.num_layers
            || config.num_key_value_heads != self.config.num_key_value_heads
            || config.head_dimension != self.config.head_dimension
            || config.max_sequence_length > self.config.context_length
        {
            return Err(EngineError::invalid_input(
                "INT8 model KV cache",
                "cache dimensions do not match model configuration",
            ));
        }
        Ok(())
    }

    fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        self.validate_token_ids(token_ids)?;
        let embedding = self
            .loaded_weights()?
            .required_float("token_embedding.weight")?;
        embedding_lookup(
            token_ids,
            embedding.values(),
            self.config.vocab_size,
            self.config.hidden_size,
        )
    }

    fn final_logits(&self, hidden_states: &[f32], rows: usize) -> Result<Vec<f32>> {
        let weights = self.loaded_weights()?;
        let norm = weights.required_float("final_norm.weight")?;
        let normalized = self
            .final_norm
            .forward_rows(hidden_states, rows, norm.values())?;
        let embedding = weights.required_float("token_embedding.weight")?;
        linear_fp32(
            &normalized,
            rows,
            self.config.hidden_size,
            embedding.values(),
            self.config.vocab_size,
        )
    }

    /// Run prompt prefill and populate the request cache.
    ///
    /// # Errors
    ///
    /// Returns token, weight, norm, quantized projection, attention, or cache errors.
    pub fn forward_prefill_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        self.validate_token_ids(token_ids)?;
        self.validate_cache(cache)?;
        if !cache.is_empty() {
            return Err(EngineError::invalid_input(
                "INT8 model prefill",
                "prompt prefill requires an empty cache",
            ));
        }
        if token_ids.len() > cache.config().max_sequence_length {
            return Err(EngineError::invalid_input(
                "INT8 model prefill",
                "prompt exceeds KV-cache capacity",
            ));
        }

        let result = (|| {
            let weights = self.loaded_weights()?;
            let mut hidden_states = self.embed_tokens(token_ids)?;
            for (layer_index, block) in self.blocks.iter().enumerate() {
                let block_weights =
                    QuantizedTransformerBlockWeights::from_model_weights(weights, layer_index)?;
                hidden_states = block.forward_prefill_with_cache(
                    &hidden_states,
                    token_ids.len(),
                    layer_index,
                    cache,
                    block_weights,
                )?;
            }
            if !cache.is_synchronized() || cache.sequence_length() != token_ids.len() {
                return Err(EngineError::invalid_input(
                    "INT8 model prefill",
                    "all KV-cache layers must finish at prompt length",
                ));
            }
            self.final_logits(&hidden_states, token_ids.len())
        })();

        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                cache.clear();
                Err(error)
            }
        }
    }

    /// Decode one token against an existing request-local KV cache.
    ///
    /// # Errors
    ///
    /// Returns token, weight, norm, quantized projection, attention, or cache errors.
    pub fn forward_cached_token(&self, token_id: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.validate_token_ids(&[token_id])?;
        self.validate_cache(cache)?;
        if !cache.is_synchronized() || cache.sequence_length() == 0 {
            return Err(EngineError::invalid_input(
                "INT8 cached decode",
                "prefill must populate a synchronized cache before decode",
            ));
        }
        if cache.remaining_capacity() == 0 {
            return Err(EngineError::invalid_input(
                "INT8 cached decode",
                "KV cache has no remaining token capacity",
            ));
        }

        let original_length = cache.sequence_length();
        let result = (|| {
            let weights = self.loaded_weights()?;
            let mut hidden_state = self.embed_tokens(&[token_id])?;
            for (layer_index, block) in self.blocks.iter().enumerate() {
                let block_weights =
                    QuantizedTransformerBlockWeights::from_model_weights(weights, layer_index)?;
                hidden_state = block.forward_cached_token(
                    &hidden_state,
                    layer_index,
                    cache,
                    block_weights,
                )?;
            }
            let expected_length = original_length.checked_add(1).ok_or_else(|| {
                EngineError::invalid_input("INT8 cached decode", "sequence length overflows usize")
            })?;
            if !cache.is_synchronized() || cache.sequence_length() != expected_length {
                return Err(EngineError::invalid_input(
                    "INT8 cached decode",
                    "every KV-cache layer must advance by exactly one token",
                ));
            }
            self.final_logits(&hidden_state, 1)
        })();

        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                cache.truncate_all(original_length)?;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QuantizedDecodeModel;
    use crate::config::ModelConfig;

    #[test]
    fn quantized_model_builds_without_weights() {
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
        let model = QuantizedDecodeModel::from_config(config).expect("valid quantized model");
        assert_eq!(model.config().hidden_size, 4);
        assert!(!model.has_weights());
    }
}
