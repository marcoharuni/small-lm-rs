//! Decoder-only NileMini model interface.

use crate::config::ModelConfig;
use crate::embedding::embedding_lookup;
use crate::error::{EngineError, Result};
use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::linear::linear;
use crate::rmsnorm::RmsNorm;
use crate::tensor::validate_matrix;
use crate::transformer::{TransformerBlock, TransformerBlockWeights};
use crate::weights::ModelWeights;

/// Executable decoder-only NileMini transformer.
///
/// The model owns validated FP32 weights and executes fresh-sequence CPU
/// prefill through token embeddings, every transformer block, final RMSNorm,
/// and the tied vocabulary projection.
#[derive(Debug)]
pub struct NileMiniModel {
    config: ModelConfig,
    blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    weights: Option<ModelWeights>,
}

impl NileMiniModel {
    /// Create an unweighted model definition from validated configuration.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when model invariants fail.
    pub fn from_config(config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let blocks = (0..config.num_layers)
            .map(|_| TransformerBlock::from_config(&config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)?;
        Ok(Self {
            config,
            blocks,
            final_norm,
            weights: None,
        })
    }

    /// Return the immutable architecture configuration.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Return the number of executable transformer blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Allocate a KV cache compatible with this model.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when the requested maximum is
    /// zero, exceeds the model context, or produces an invalid cache shape.
    pub fn allocate_kv_cache(&self, max_sequence_length: usize) -> Result<KvCache> {
        let config = KvCacheConfig::from_model(&self.config, max_sequence_length)?;
        KvCache::allocate(config)
    }

    /// Return whether CPU-native weights have been attached.
    #[must_use]
    pub const fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Attach SafeTensors weights after validating their layout and dtypes.
    ///
    /// # Errors
    ///
    /// Returns artifact or model-layout validation errors.
    pub fn load_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let weights = ModelWeights::load(path, &self.config)?;
        self.weights = Some(weights);
        Ok(())
    }

    fn loaded_weights(&self) -> Result<&ModelWeights> {
        self.weights.as_ref().ok_or_else(|| {
            EngineError::invalid_input("model execution", "model weights are not loaded")
        })
    }

    fn validate_token_ids(&self, token_ids: &[u32]) -> Result<()> {
        if token_ids.is_empty() {
            return Err(EngineError::invalid_input(
                "model forward",
                "at least one token is required",
            ));
        }
        if token_ids.len() > self.config.context_length {
            return Err(EngineError::invalid_input(
                "model forward",
                format!(
                    "sequence length exceeds context length {}",
                    self.config.context_length
                ),
            ));
        }
        if let Some(token_id) = token_ids
            .iter()
            .copied()
            .find(|token_id| *token_id as usize >= self.config.vocab_size)
        {
            return Err(EngineError::invalid_input(
                "model forward",
                format!("token id {token_id} is outside the vocabulary"),
            ));
        }
        Ok(())
    }

    /// Gather row-major token embeddings for one validated sequence.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for invalid tokens or missing model
    /// weights, and propagates embedding-table shape errors.
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
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

    /// Execute every transformer block in layer order.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for inconsistent hidden-state dimensions
    /// or missing weights, and propagates block-execution failures.
    pub fn run_transformer_blocks(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "model hidden states",
            hidden_states,
            sequence_length,
            self.config.hidden_size,
        )?;
        let weights = self.loaded_weights()?;
        let mut output = hidden_states.to_vec();

        for (layer_index, block) in self.blocks.iter().enumerate() {
            let block_weights = TransformerBlockWeights::from_model_weights(weights, layer_index)?;
            output = block.forward_prefill(&output, sequence_length, block_weights)?;
        }

        Ok(output)
    }

    /// Apply the exported final RMSNorm to all sequence rows.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for inconsistent hidden-state dimensions
    /// or missing weights, and propagates RMSNorm failures.
    pub fn apply_final_norm(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "final model hidden states",
            hidden_states,
            sequence_length,
            self.config.hidden_size,
        )?;
        let scale = self
            .loaded_weights()?
            .required_tensor_values("final_norm.weight")?;
        self.final_norm
            .forward_rows(hidden_states, sequence_length, scale)
    }

    /// Project normalized hidden states to vocabulary logits using the tied
    /// token-embedding matrix.
    ///
    /// The returned row-major tensor has shape
    /// `[sequence_length, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for inconsistent hidden-state dimensions
    /// or missing weights, and propagates linear-projection failures.
    pub fn project_tied_logits(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "logits input",
            hidden_states,
            sequence_length,
            self.config.hidden_size,
        )?;
        let embedding = self
            .loaded_weights()?
            .required_tensor_values("token_embedding.weight")?;
        linear(
            hidden_states,
            sequence_length,
            self.config.hidden_size,
            embedding,
            self.config.vocab_size,
        )
    }

    /// Run embeddings, all transformer blocks, and final RMSNorm.
    ///
    /// The returned row-major tensor has shape
    /// `[token_ids.len(), hidden_size]`.
    ///
    /// # Errors
    ///
    /// Returns validation, missing-weight, or model-operation failures.
    pub fn forward_hidden_states(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let sequence_length = token_ids.len();
        let embeddings = self.embed_tokens(token_ids)?;
        let transformed = self.run_transformer_blocks(&embeddings, sequence_length)?;
        self.apply_final_norm(&transformed, sequence_length)
    }

    fn validate_kv_cache(&self, cache: &KvCache) -> Result<()> {
        let cache_config = cache.config();
        if cache_config.num_layers != self.config.num_layers
            || cache_config.num_key_value_heads != self.config.num_key_value_heads
            || cache_config.head_dimension != self.config.head_dimension
            || cache_config.max_sequence_length > self.config.context_length
        {
            return Err(EngineError::invalid_input(
                "model KV cache",
                "cache dimensions do not match the model configuration",
            ));
        }
        Ok(())
    }

    fn execute_prefill_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
        mut hidden_states: Vec<f32>,
    ) -> Result<Vec<f32>> {
        let weights = self.loaded_weights()?;
        for (layer_index, block) in self.blocks.iter().enumerate() {
            let block_weights = TransformerBlockWeights::from_model_weights(weights, layer_index)?;
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
                "model KV cache",
                "all cache layers must finish prompt prefill at the prompt length",
            ));
        }
        let normalized = self.apply_final_norm(&hidden_states, token_ids.len())?;
        self.project_tied_logits(&normalized, token_ids.len())
    }

    /// Run prompt prefill through all layers while populating a synchronized KV
    /// cache and return full-sequence vocabulary logits.
    ///
    /// The supplied cache must be empty and have capacity for the whole prompt.
    /// On any failure after cache mutation begins, all layer lengths are reset
    /// to zero so a partial prompt state is never exposed.
    ///
    /// # Errors
    ///
    /// Returns token, weight, cache-shape, capacity, transformer, final-norm,
    /// or tied-projection errors.
    pub fn forward_prefill_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        self.validate_token_ids(token_ids)?;
        self.validate_kv_cache(cache)?;
        if !cache.is_empty() {
            return Err(EngineError::invalid_input(
                "model KV cache",
                "prompt prefill requires an empty cache",
            ));
        }
        if token_ids.len() > cache.config().max_sequence_length {
            return Err(EngineError::invalid_input(
                "model KV cache",
                format!(
                    "prompt length {} exceeds cache capacity {}",
                    token_ids.len(),
                    cache.config().max_sequence_length
                ),
            ));
        }

        let embeddings = self.embed_tokens(token_ids)?;
        let result = self.execute_prefill_with_cache(token_ids, cache, embeddings);
        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                cache.clear();
                Err(error)
            }
        }
    }

    /// Decode one new input token against an existing synchronized KV cache.
    ///
    /// The cache must already contain prompt state. Every layer is extended by
    /// exactly one position, and the returned vector contains one vocabulary
    /// row. Any failure rolls every layer back to its original length.
    ///
    /// # Errors
    ///
    /// Returns token, cache-state, capacity, weight, transformer, norm, or
    /// tied-projection errors.
    pub fn forward_cached_token(&self, token_id: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.validate_token_ids(&[token_id])?;
        self.validate_kv_cache(cache)?;
        if !cache.is_synchronized() {
            return Err(EngineError::invalid_input(
                "model cached decode",
                "all cache layers must have the same length before decoding",
            ));
        }
        let previous_length = cache.sequence_length();
        if previous_length == 0 {
            return Err(EngineError::invalid_input(
                "model cached decode",
                "prompt prefill must populate the cache before token decoding",
            ));
        }
        if cache.remaining_capacity() == 0 {
            return Err(EngineError::invalid_input(
                "model cached decode",
                "KV cache has no remaining token capacity",
            ));
        }

        let result = (|| {
            let weights = self.loaded_weights()?;
            let mut hidden_state = self.embed_tokens(&[token_id])?;
            for (layer_index, block) in self.blocks.iter().enumerate() {
                let block_weights =
                    TransformerBlockWeights::from_model_weights(weights, layer_index)?;
                hidden_state =
                    block.forward_cached_token(&hidden_state, layer_index, cache, block_weights)?;
            }
            let expected_length = previous_length.checked_add(1).ok_or_else(|| {
                EngineError::invalid_input(
                    "model cached decode",
                    "cache sequence length overflows usize",
                )
            })?;
            if !cache.is_synchronized() || cache.sequence_length() != expected_length {
                return Err(EngineError::invalid_input(
                    "model cached decode",
                    "all cache layers must advance by exactly one token",
                ));
            }
            let normalized = self.apply_final_norm(&hidden_state, 1)?;
            self.project_tied_logits(&normalized, 1)
        })();

        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                cache.truncate_all(previous_length)?;
                Err(error)
            }
        }
    }

    /// Run a complete fresh-sequence forward pass and return all vocabulary
    /// logits in row-major `[sequence_length, vocab_size]` layout.
    ///
    /// # Errors
    ///
    /// Returns validation, missing-weight, or model-operation failures.
    pub fn forward_prefill(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden_states = self.forward_hidden_states(token_ids)?;
        self.project_tied_logits(&hidden_states, token_ids.len())
    }

    /// Compute logits either with ordinary fresh-sequence prefill or with
    /// cache-populating prompt prefill.
    ///
    /// # Errors
    ///
    /// Returns validation, cache, or model-operation errors.
    pub fn forward(&self, token_ids: &[u32], cache: Option<&mut KvCache>) -> Result<Vec<f32>> {
        match cache {
            Some(cache) => self.forward_prefill_with_cache(token_ids, cache),
            None => self.forward_prefill(token_ids),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NileMiniModel;
    use crate::config::ModelConfig;
    use crate::linear::linear;
    use crate::rmsnorm::RmsNorm;
    use crate::weights::ModelWeights;
    use std::path::PathBuf;

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
            num_layers: 2,
            hidden_size: 4,
            intermediate_size: 4,
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
            expected_parameter_count: 244,
        }
    }

    #[test]
    fn embedding_requires_attached_weights() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        let error = model.embed_tokens(&[1]).expect_err("weights are required");
        assert!(error.to_string().contains("weights are not loaded"));
    }

    #[test]
    fn embedding_rejects_invalid_tokens_before_weight_access() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        assert!(model.embed_tokens(&[]).is_err());
        assert!(model.embed_tokens(&[8]).is_err());
    }

    #[test]
    fn transformer_stack_requires_attached_weights() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        let error = model
            .run_transformer_blocks(&[0.0; 4], 1)
            .expect_err("weights are required");
        assert!(error.to_string().contains("weights are not loaded"));
    }

    #[test]
    fn transformer_stack_validates_hidden_state_shape() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        assert!(model.run_transformer_blocks(&[0.0; 3], 1).is_err());
    }

    #[test]
    fn final_norm_requires_attached_weights() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        let error = model
            .apply_final_norm(&[0.0; 4], 1)
            .expect_err("weights are required");
        assert!(error.to_string().contains("weights are not loaded"));
    }

    #[test]
    fn tied_logits_require_attached_weights() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        let error = model
            .project_tied_logits(&[0.0; 4], 1)
            .expect_err("weights are required");
        assert!(error.to_string().contains("weights are not loaded"));
    }

    #[test]
    fn full_hidden_state_path_requires_attached_weights() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        let error = model
            .forward_hidden_states(&[1])
            .expect_err("weights are required");
        assert!(error.to_string().contains("weights are not loaded"));
    }

    fn synthetic_weights(config: &ModelConfig) -> ModelWeights {
        ModelWeights::synthetic(config, |name, shape| {
            let length = shape.iter().product::<usize>();
            if name == "token_embedding.weight" {
                (0..length)
                    .map(|index| (index as f32 - 12.0) / 16.0)
                    .collect()
            } else if name.ends_with("_norm.weight") || name == "final_norm.weight" {
                vec![1.0; length]
            } else {
                vec![0.0; length]
            }
        })
        .expect("valid synthetic model weights")
    }

    #[test]
    fn complete_prefill_runs_every_layer_and_returns_all_logits() {
        let config = tiny_config();
        let mut model = NileMiniModel::from_config(config.clone()).expect("valid model");
        model.weights = Some(synthetic_weights(&config));
        let token_ids = [1, 2];

        let embeddings = model.embed_tokens(&token_ids).expect("valid embeddings");
        let expected_hidden = RmsNorm::new(config.hidden_size, config.rms_norm_epsilon)
            .expect("valid final norm")
            .forward_rows(&embeddings, token_ids.len(), &[1.0; 4])
            .expect("valid normalized hidden states");
        let embedding_table = model
            .weights
            .as_ref()
            .expect("attached weights")
            .required_tensor_values("token_embedding.weight")
            .expect("embedding table");
        let expected_logits = linear(
            &expected_hidden,
            token_ids.len(),
            config.hidden_size,
            embedding_table,
            config.vocab_size,
        )
        .expect("valid tied projection");

        let hidden = model
            .forward_hidden_states(&token_ids)
            .expect("valid full hidden states");
        assert_eq!(hidden, expected_hidden);

        let logits = model.forward_prefill(&token_ids).expect("valid prefill");
        assert_eq!(logits.len(), token_ids.len() * config.vocab_size);
        assert_eq!(logits, expected_logits);
        assert_eq!(
            model.forward(&token_ids, None).expect("uncached forward"),
            logits
        );
    }

    #[test]
    fn cache_populating_prefill_matches_uncached_model_logits() {
        let config = tiny_config();
        let mut model = NileMiniModel::from_config(config.clone()).expect("valid model");
        model.weights = Some(synthetic_weights(&config));
        let token_ids = [1, 2, 3];
        let expected = model.forward_prefill(&token_ids).expect("uncached prefill");
        let mut cache = model.allocate_kv_cache(8).expect("allocated cache");

        let actual = model
            .forward(&token_ids, Some(&mut cache))
            .expect("cached prompt prefill");

        assert_eq!(actual, expected);
        assert_eq!(cache.sequence_length(), token_ids.len());
        assert!(cache.is_synchronized());
        for layer in 0..config.num_layers {
            assert_eq!(
                cache.layer_sequence_length(layer).expect("layer length"),
                token_ids.len()
            );
        }
        assert!(model
            .forward_prefill_with_cache(&token_ids, &mut cache)
            .is_err());
    }

    #[test]
    fn cached_token_logits_match_fresh_sequence_logits() {
        let config = tiny_config();
        let mut model = NileMiniModel::from_config(config.clone()).expect("valid model");
        model.weights = Some(synthetic_weights(&config));
        let mut cache = model.allocate_kv_cache(8).expect("allocated cache");
        model
            .forward_prefill_with_cache(&[1, 2], &mut cache)
            .expect("prompt prefill");

        let cached = model
            .forward_cached_token(3, &mut cache)
            .expect("cached decode");
        let fresh = model.forward_prefill(&[1, 2, 3]).expect("fresh prefill");
        let final_start = fresh.len() - config.vocab_size;

        assert_eq!(cached.as_slice(), &fresh[final_start..]);
        assert_eq!(cache.sequence_length(), 3);
        assert!(cache.is_synchronized());
    }

    #[test]
    fn cached_and_uncached_greedy_generation_match() {
        use crate::greedy::{greedy_generate_cached, greedy_generate_uncached};

        let config = tiny_config();
        let mut model = NileMiniModel::from_config(config.clone()).expect("valid model");
        model.weights = Some(synthetic_weights(&config));
        let uncached =
            greedy_generate_uncached(&model, &[1, 2], 3, None).expect("uncached generation");
        let cached = greedy_generate_cached(&model, &[1, 2], 3, None).expect("cached generation");
        assert_eq!(cached, uncached);
    }

    #[test]
    fn zero_temperature_generation_matches_cached_greedy() {
        use crate::generation::generate_token_ids;
        use crate::greedy::greedy_generate_cached;
        use crate::sampler::SamplingConfig;

        let config = tiny_config();
        let mut model = NileMiniModel::from_config(config.clone()).expect("valid model");
        model.weights = Some(synthetic_weights(&config));
        let expected = greedy_generate_cached(&model, &[1, 2], 3, None).expect("greedy generation");
        let actual = generate_token_ids(
            &model,
            &[1, 2],
            3,
            None,
            SamplingConfig {
                temperature: 0.0,
                ..SamplingConfig::default()
            },
        )
        .expect("sampled generation");
        assert_eq!(actual.generated_token_ids, expected);
        assert_eq!(actual.finish_reason, "length");
    }

    #[test]
    fn sampled_generation_stops_at_eos() {
        use crate::generation::generate_token_ids;
        use crate::greedy::greedy_generate_cached;
        use crate::sampler::SamplingConfig;

        let config = tiny_config();
        let mut model = NileMiniModel::from_config(config.clone()).expect("valid model");
        model.weights = Some(synthetic_weights(&config));
        let first_token =
            greedy_generate_cached(&model, &[1, 2], 1, None).expect("one greedy token")[0];
        let output = generate_token_ids(
            &model,
            &[1, 2],
            3,
            Some(first_token),
            SamplingConfig {
                temperature: 0.0,
                ..SamplingConfig::default()
            },
        )
        .expect("EOS generation");
        assert_eq!(output.generated_token_ids, vec![first_token]);
        assert_eq!(output.finish_reason, "eos");
    }

    #[test]
    fn cached_decode_requires_prompt_state_and_capacity() {
        let config = tiny_config();
        let mut model = NileMiniModel::from_config(config.clone()).expect("valid model");
        model.weights = Some(synthetic_weights(&config));
        let mut empty_cache = model.allocate_kv_cache(2).expect("allocated cache");
        assert!(model.forward_cached_token(1, &mut empty_cache).is_err());

        let mut full_cache = model.allocate_kv_cache(1).expect("allocated cache");
        model
            .forward_prefill_with_cache(&[1], &mut full_cache)
            .expect("prompt prefill");
        assert!(model.forward_cached_token(2, &mut full_cache).is_err());
        assert_eq!(full_cache.sequence_length(), 1);
    }

    #[test]
    #[ignore = "requires the local 30.52 MiB artifact and a release CPU run"]
    fn exported_smoke_model_runs_one_token_prefill() {
        let artifact =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/nilemini-8m-situ");
        let model_path = artifact.join("model.safetensors");
        if !model_path.is_file() {
            return;
        }

        let config =
            ModelConfig::from_json_path(artifact.join("config.json")).expect("valid config");
        let vocab_size = config.vocab_size;
        let mut model = NileMiniModel::from_config(config).expect("valid model");
        model.load_weights(model_path).expect("valid model weights");

        let logits = model
            .forward_prefill(&[1])
            .expect("one-token full-model prefill");
        assert_eq!(logits.len(), vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn constructs_every_configured_transformer_block() {
        let model = NileMiniModel::from_config(tiny_config()).expect("valid model");
        assert_eq!(model.block_count(), 2);
        assert!(!model.has_weights());
    }
}
