//! Full-model execution for batched cached decode.

use std::path::Path;

use crate::batched_transformer::BatchedTransformerBlock;
use crate::config::ModelConfig;
use crate::embedding::embedding_lookup;
use crate::error::{EngineError, Result};
use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::linear::linear_fp32;
use crate::rmsnorm::RmsNorm;
use crate::transformer::{TransformerBlock, TransformerBlockWeights};
use crate::weights::ModelWeights;

/// Model executor that batches one decode row from each active request.
///
/// Prompt prefill uses the established single-sequence transformer path.
/// Cached decode uses [`BatchedTransformerBlock`] so dense projections operate
/// across all active request rows while each request retains its own KV cache.
#[derive(Debug)]
pub struct BatchedDecodeModel {
    config: ModelConfig,
    prefill_blocks: Vec<TransformerBlock>,
    decode_blocks: Vec<BatchedTransformerBlock>,
    final_norm: RmsNorm,
    weights: Option<ModelWeights>,
}

impl BatchedDecodeModel {
    /// Construct an unweighted batched decoder from validated configuration.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when model invariants fail.
    pub fn from_config(config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let prefill_blocks = (0..config.num_layers)
            .map(|_| TransformerBlock::from_config(&config))
            .collect::<Result<Vec<_>>>()?;
        let decode_blocks = (0..config.num_layers)
            .map(|_| BatchedTransformerBlock::from_config(&config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)?;
        Ok(Self {
            config,
            prefill_blocks,
            decode_blocks,
            final_norm,
            weights: None,
        })
    }

    /// Return the immutable architecture configuration.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Return whether validated model weights have been loaded.
    #[must_use]
    pub const fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Load and validate the exported FP32 SafeTensors weights.
    ///
    /// # Errors
    ///
    /// Returns artifact or model-layout validation errors.
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.weights = Some(ModelWeights::load(path, &self.config)?);
        Ok(())
    }

    /// Allocate one request-local KV cache.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for an invalid requested
    /// sequence capacity.
    pub fn allocate_kv_cache(&self, max_sequence_length: usize) -> Result<KvCache> {
        let config = KvCacheConfig::from_model(&self.config, max_sequence_length)?;
        KvCache::allocate(config)
    }

    fn loaded_weights(&self) -> Result<&ModelWeights> {
        self.weights.as_ref().ok_or_else(|| {
            EngineError::invalid_input("batched model execution", "model weights are not loaded")
        })
    }

    fn validate_token_ids(&self, token_ids: &[u32]) -> Result<()> {
        if token_ids.is_empty() {
            return Err(EngineError::invalid_input(
                "batched model forward",
                "at least one token is required",
            ));
        }
        if let Some(token_id) = token_ids
            .iter()
            .copied()
            .find(|token_id| *token_id as usize >= self.config.vocab_size)
        {
            return Err(EngineError::invalid_input(
                "batched model forward",
                format!("token id {token_id} is outside the vocabulary"),
            ));
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &KvCache) -> Result<()> {
        let cache_config = cache.config();
        if cache_config.num_layers != self.config.num_layers
            || cache_config.num_key_value_heads != self.config.num_key_value_heads
            || cache_config.head_dimension != self.config.head_dimension
            || cache_config.max_sequence_length > self.config.context_length
        {
            return Err(EngineError::invalid_input(
                "batched model KV cache",
                "cache dimensions do not match the model configuration",
            ));
        }
        Ok(())
    }

    fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        self.validate_token_ids(token_ids)?;
        let table = self
            .loaded_weights()?
            .required_tensor_values("token_embedding.weight")?;
        embedding_lookup(
            token_ids,
            table,
            self.config.vocab_size,
            self.config.hidden_size,
        )
    }

    fn apply_final_norm(&self, hidden_states: &[f32], rows: usize) -> Result<Vec<f32>> {
        let scale = self
            .loaded_weights()?
            .required_tensor_values("final_norm.weight")?;
        self.final_norm.forward_rows(hidden_states, rows, scale)
    }

    fn project_tied_logits(&self, hidden_states: &[f32], rows: usize) -> Result<Vec<f32>> {
        let embedding = self
            .loaded_weights()?
            .required_tensor_values("token_embedding.weight")?;
        linear_fp32(
            hidden_states,
            rows,
            self.config.hidden_size,
            embedding,
            self.config.vocab_size,
        )
    }

    /// Run prompt prefill for one request while populating its cache.
    ///
    /// # Errors
    ///
    /// Returns token, cache, weight, transformer, norm, or projection errors.
    pub fn forward_prefill_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        self.validate_token_ids(token_ids)?;
        self.validate_cache(cache)?;
        if token_ids.len() > self.config.context_length {
            return Err(EngineError::invalid_input(
                "batched model prefill",
                "prompt exceeds the model context length",
            ));
        }
        if !cache.is_empty() {
            return Err(EngineError::invalid_input(
                "batched model prefill",
                "prompt prefill requires an empty cache",
            ));
        }
        if token_ids.len() > cache.config().max_sequence_length {
            return Err(EngineError::invalid_input(
                "batched model prefill",
                "prompt exceeds KV-cache capacity",
            ));
        }

        let result = (|| {
            let weights = self.loaded_weights()?;
            let mut hidden_states = self.embed_tokens(token_ids)?;
            for (layer_index, block) in self.prefill_blocks.iter().enumerate() {
                let block_weights =
                    TransformerBlockWeights::from_model_weights(weights, layer_index)?;
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
                    "batched model prefill",
                    "all cache layers must finish prefill at the prompt length",
                ));
            }
            let normalized = self.apply_final_norm(&hidden_states, token_ids.len())?;
            self.project_tied_logits(&normalized, token_ids.len())
        })();

        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                cache.clear();
                Err(error)
            }
        }
    }

    /// Decode one token for every active request in one full-model batch.
    ///
    /// Input token `i` is evaluated against cache `i`. All dense transformer
    /// and vocabulary projections use the complete batch of active rows.
    ///
    /// # Errors
    ///
    /// Returns token, cache, model, transformer, norm, or projection errors.
    /// Any failure restores every request cache to its length before the batch.
    pub fn forward_cached_batch(
        &self,
        token_ids: &[u32],
        caches: &mut [&mut KvCache],
    ) -> Result<Vec<f32>> {
        self.validate_token_ids(token_ids)?;
        if token_ids.len() != caches.len() {
            return Err(EngineError::invalid_input(
                "batched model decode",
                format!(
                    "received {} tokens for {} request caches",
                    token_ids.len(),
                    caches.len()
                ),
            ));
        }

        let mut original_lengths = Vec::with_capacity(caches.len());
        for cache in caches.iter() {
            self.validate_cache(cache)?;
            if !cache.is_synchronized() {
                return Err(EngineError::invalid_input(
                    "batched model decode",
                    "every request cache must be synchronized before decode",
                ));
            }
            let length = cache.sequence_length();
            if length == 0 {
                return Err(EngineError::invalid_input(
                    "batched model decode",
                    "prompt prefill must populate every request cache before decode",
                ));
            }
            if cache.remaining_capacity() == 0 {
                return Err(EngineError::invalid_input(
                    "batched model decode",
                    "a request KV cache has no remaining token capacity",
                ));
            }
            original_lengths.push(length);
        }

        let result = (|| {
            let weights = self.loaded_weights()?;
            let mut hidden_states = self.embed_tokens(token_ids)?;
            for (layer_index, block) in self.decode_blocks.iter().enumerate() {
                let block_weights =
                    TransformerBlockWeights::from_model_weights(weights, layer_index)?;
                hidden_states = block.forward_cached_batch(
                    &hidden_states,
                    layer_index,
                    caches,
                    block_weights,
                )?;
            }

            for (cache, &previous_length) in caches.iter().zip(&original_lengths) {
                let expected_length = previous_length.checked_add(1).ok_or_else(|| {
                    EngineError::invalid_input(
                        "batched model decode",
                        "cache sequence length overflows usize",
                    )
                })?;
                if !cache.is_synchronized() || cache.sequence_length() != expected_length {
                    return Err(EngineError::invalid_input(
                        "batched model decode",
                        "every request cache must advance by exactly one token",
                    ));
                }
            }

            let normalized = self.apply_final_norm(&hidden_states, token_ids.len())?;
            self.project_tied_logits(&normalized, token_ids.len())
        })();

        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                rollback_caches(caches, &original_lengths)?;
                Err(error)
            }
        }
    }

    /// Decode one token using the batched path with a batch size of one.
    ///
    /// # Errors
    ///
    /// Propagates full-model cached-decode errors.
    pub fn forward_cached_token(&self, token_id: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.forward_cached_batch(&[token_id], &mut [cache])
    }
}

fn rollback_caches(caches: &mut [&mut KvCache], original_lengths: &[usize]) -> Result<()> {
    for (cache, &length) in caches.iter_mut().zip(original_lengths) {
        cache.truncate_all(length)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::BatchedDecodeModel;
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
    fn decoder_builds_from_model_configuration() {
        let model = BatchedDecodeModel::from_config(tiny_config()).expect("valid decoder");
        assert_eq!(model.config().hidden_size, 4);
        assert!(!model.has_weights());
    }
}
