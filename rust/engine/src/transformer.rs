//! Pre-norm NileMini transformer-block execution.

use crate::attention::{AttentionWeights, GroupedQueryAttention};
use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::kv_cache::KvCache;
use crate::rmsnorm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::situ_glu::SituGlu;
use crate::tensor::validate_matrix;
use crate::weights::ModelWeights;

/// Borrowed FP32 tensors required by one transformer block.
///
/// Projection matrices follow the exported `[out_features, in_features]`
/// layout. Norm weights are one-dimensional hidden-width vectors.
#[derive(Clone, Copy, Debug)]
pub struct TransformerBlockWeights<'a> {
    attention_norm: &'a [f32],
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    output: &'a [f32],
    ffn_norm: &'a [f32],
    gate: &'a [f32],
    up: &'a [f32],
    down: &'a [f32],
}

impl TransformerBlockWeights<'_> {
    /// Return the nine canonical exported tensor names for one layer.
    #[must_use]
    pub fn tensor_names(layer_index: usize) -> [String; 9] {
        let prefix = format!("layers.{layer_index}");
        [
            format!("{prefix}.attention_norm.weight"),
            format!("{prefix}.q_proj.weight"),
            format!("{prefix}.k_proj.weight"),
            format!("{prefix}.v_proj.weight"),
            format!("{prefix}.o_proj.weight"),
            format!("{prefix}.ffn_norm.weight"),
            format!("{prefix}.gate_proj.weight"),
            format!("{prefix}.up_proj.weight"),
            format!("{prefix}.down_proj.weight"),
        ]
    }
}

impl<'a> TransformerBlockWeights<'a> {
    /// Borrow one layer directly from validated model weights.
    ///
    /// # Errors
    ///
    /// Returns an invalid-model-weights error if any canonical layer tensor is
    /// absent. Shapes and dtypes were already validated during model loading.
    pub fn from_model_weights(weights: &'a ModelWeights, layer_index: usize) -> Result<Self> {
        let names = Self::tensor_names(layer_index);
        let tensor = |name: &str| {
            weights
                .tensor(name)
                .map(|tensor| tensor.values())
                .ok_or_else(|| {
                    EngineError::invalid_weights(format!(
                        "required transformer tensor {name} is missing"
                    ))
                })
        };

        Ok(Self::new(
            tensor(&names[0])?,
            tensor(&names[1])?,
            tensor(&names[2])?,
            tensor(&names[3])?,
            tensor(&names[4])?,
            tensor(&names[5])?,
            tensor(&names[6])?,
            tensor(&names[7])?,
            tensor(&names[8])?,
        ))
    }

    /// Bundle all norm, attention, and feed-forward tensors for one block.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        attention_norm: &'a [f32],
        query: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        output: &'a [f32],
        ffn_norm: &'a [f32],
        gate: &'a [f32],
        up: &'a [f32],
        down: &'a [f32],
    ) -> Self {
        Self {
            attention_norm,
            query,
            key,
            value,
            output,
            ffn_norm,
            gate,
            up,
            down,
        }
    }

    /// Return the attention RMSNorm scale.
    #[must_use]
    pub const fn attention_norm(&self) -> &'a [f32] {
        self.attention_norm
    }

    /// Return the query-projection matrix.
    #[must_use]
    pub const fn query(&self) -> &'a [f32] {
        self.query
    }

    /// Return the key-projection matrix.
    #[must_use]
    pub const fn key(&self) -> &'a [f32] {
        self.key
    }

    /// Return the value-projection matrix.
    #[must_use]
    pub const fn value(&self) -> &'a [f32] {
        self.value
    }

    /// Return the attention output-projection matrix.
    #[must_use]
    pub const fn output(&self) -> &'a [f32] {
        self.output
    }

    /// Return the feed-forward RMSNorm scale.
    #[must_use]
    pub const fn ffn_norm(&self) -> &'a [f32] {
        self.ffn_norm
    }

    /// Return the SiTU gate-projection matrix.
    #[must_use]
    pub const fn gate(&self) -> &'a [f32] {
        self.gate
    }

    /// Return the SiTU up-projection matrix.
    #[must_use]
    pub const fn up(&self) -> &'a [f32] {
        self.up
    }

    /// Return the feed-forward down-projection matrix.
    #[must_use]
    pub const fn down(&self) -> &'a [f32] {
        self.down
    }
}

/// Executable metadata for one NileMini pre-norm transformer block.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformerBlock {
    hidden_size: usize,
    attention_norm: RmsNorm,
    attention: GroupedQueryAttention,
    rotary: RotaryEmbedding,
    ffn_norm: RmsNorm,
    ffn: SituGlu,
}

impl TransformerBlock {
    /// Build block operations from the validated model configuration.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when any model invariant fails.
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

    /// Execute the pre-norm attention sublayer and its residual connection.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for inconsistent hidden states or any
    /// invalid norm/attention tensor.
    pub fn attention_residual(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        weights: TransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "transformer hidden states",
            hidden_states,
            sequence_length,
            self.hidden_size,
        )?;
        let normalized = self.attention_norm.forward_rows(
            hidden_states,
            sequence_length,
            weights.attention_norm(),
        )?;
        let attention_weights = AttentionWeights::new(
            weights.query(),
            weights.key(),
            weights.value(),
            weights.output(),
        );
        let attention_output = self.attention.forward_prefill(
            &normalized,
            sequence_length,
            attention_weights,
            &self.rotary,
        )?;
        let mut output = hidden_states.to_vec();
        add_residual_in_place(&mut output, &attention_output)?;
        Ok(output)
    }
    /// Execute the pre-norm SiTU-GLU sublayer and residual connection.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for inconsistent hidden states or any
    /// invalid norm/feed-forward tensor.
    pub fn ffn_residual(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        weights: TransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "transformer hidden states",
            hidden_states,
            sequence_length,
            self.hidden_size,
        )?;
        let normalized =
            self.ffn_norm
                .forward_rows(hidden_states, sequence_length, weights.ffn_norm())?;
        let ffn_output =
            self.ffn
                .forward(&normalized, weights.gate(), weights.up(), weights.down())?;
        let mut output = hidden_states.to_vec();
        add_residual_in_place(&mut output, &ffn_output)?;
        Ok(output)
    }
    /// Run a prompt through this block while populating one empty cache layer.
    ///
    /// Execution is identical to [`Self::forward_prefill`], with the rotated
    /// attention keys and values committed only after attention succeeds.
    ///
    /// # Errors
    ///
    /// Returns hidden-state, norm, attention, feed-forward, or cache errors.
    pub fn forward_prefill_with_cache(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        layer_index: usize,
        cache: &mut KvCache,
        weights: TransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "transformer hidden states",
            hidden_states,
            sequence_length,
            self.hidden_size,
        )?;
        let normalized = self.attention_norm.forward_rows(
            hidden_states,
            sequence_length,
            weights.attention_norm(),
        )?;
        let attention_weights = AttentionWeights::new(
            weights.query(),
            weights.key(),
            weights.value(),
            weights.output(),
        );
        let attention_output = self.attention.forward_prefill_with_cache(
            &normalized,
            sequence_length,
            layer_index,
            cache,
            attention_weights,
            &self.rotary,
        )?;
        let mut after_attention = hidden_states.to_vec();
        add_residual_in_place(&mut after_attention, &attention_output)?;
        self.ffn_residual(&after_attention, sequence_length, weights)
    }

    /// Run one decode token through this block using and extending one cache
    /// layer.
    ///
    /// # Errors
    ///
    /// Returns hidden-state, norm, cached-attention, feed-forward, or cache
    /// errors. The input and output both have one hidden-state row.
    pub fn forward_cached_token(
        &self,
        hidden_state: &[f32],
        layer_index: usize,
        cache: &mut KvCache,
        weights: TransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "transformer cached hidden state",
            hidden_state,
            1,
            self.hidden_size,
        )?;
        let normalized =
            self.attention_norm
                .forward_rows(hidden_state, 1, weights.attention_norm())?;
        let attention_weights = AttentionWeights::new(
            weights.query(),
            weights.key(),
            weights.value(),
            weights.output(),
        );
        let attention_output = self.attention.forward_cached_token(
            &normalized,
            layer_index,
            cache,
            attention_weights,
            &self.rotary,
        )?;
        let mut after_attention = hidden_state.to_vec();
        add_residual_in_place(&mut after_attention, &attention_output)?;
        self.ffn_residual(&after_attention, 1, weights)
    }

    /// Run one complete pre-norm transformer block for a fresh sequence.
    ///
    /// Execution order exactly matches the JAX reference:
    /// `x += attention(attention_norm(x))`, then
    /// `x += ffn(ffn_norm(x))`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for inconsistent hidden states or any
    /// invalid block tensor.
    pub fn forward_prefill(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        weights: TransformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        let after_attention = self.attention_residual(hidden_states, sequence_length, weights)?;
        self.ffn_residual(&after_attention, sequence_length, weights)
    }
}

/// Add one residual update into a hidden-state buffer in place.
///
/// # Errors
///
/// Returns [`EngineError::InvalidInput`] when the two flattened tensors have
/// different lengths.
pub fn add_residual_in_place(hidden_states: &mut [f32], update: &[f32]) -> Result<()> {
    if hidden_states.len() != update.len() {
        return Err(EngineError::invalid_input(
            "residual connection",
            format!(
                "hidden-state length {} differs from update length {}",
                hidden_states.len(),
                update.len()
            ),
        ));
    }

    for (hidden_state, &update_value) in hidden_states.iter_mut().zip(update) {
        *hidden_state += update_value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::add_residual_in_place;

    fn tiny_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
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
            expected_parameter_count: 1,
        }
    }

    fn zero_block_weights<'a>(
        norm: &'a [f32; 4],
        square: &'a [f32; 16],
        key_value: &'a [f32; 8],
    ) -> super::TransformerBlockWeights<'a> {
        super::TransformerBlockWeights::new(
            norm, square, key_value, key_value, square, norm, square, square, square,
        )
    }

    #[test]
    fn zero_attention_update_preserves_the_residual_stream() {
        let block =
            super::TransformerBlock::from_config(&tiny_config()).expect("valid transformer block");
        let norm = [1.0; 4];
        let square = [0.0; 16];
        let key_value = [0.0; 8];
        let input = [1.0, -2.0, 3.0, -4.0];

        let output = block
            .attention_residual(&input, 1, zero_block_weights(&norm, &square, &key_value))
            .expect("valid attention residual");
        assert_eq!(output, input);
    }

    #[test]
    fn zero_feed_forward_update_preserves_the_residual_stream() {
        let block =
            super::TransformerBlock::from_config(&tiny_config()).expect("valid transformer block");
        let norm = [1.0; 4];
        let square = [0.0; 16];
        let key_value = [0.0; 8];
        let input = [1.0, -2.0, 3.0, -4.0];

        let output = block
            .ffn_residual(&input, 1, zero_block_weights(&norm, &square, &key_value))
            .expect("valid feed-forward residual");
        assert_eq!(output, input);
    }

    #[test]
    fn complete_block_applies_attention_before_feed_forward() {
        let block =
            super::TransformerBlock::from_config(&tiny_config()).expect("valid transformer block");
        let norm = [1.0; 4];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let key_value = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let zero = [0.0; 16];
        let weights = super::TransformerBlockWeights::new(
            &norm, &identity, &key_value, &key_value, &identity, &norm, &zero, &zero, &zero,
        );
        let input = [1.0, 2.0, 3.0, 4.0];

        let after_attention = block
            .attention_residual(&input, 1, weights)
            .expect("valid attention residual");
        let complete = block
            .forward_prefill(&input, 1, weights)
            .expect("valid complete block");

        assert_eq!(complete, after_attention);
        assert_ne!(complete, input);
    }

    #[test]
    fn complete_zero_weight_block_is_an_exact_identity() {
        let block =
            super::TransformerBlock::from_config(&tiny_config()).expect("valid transformer block");
        let norm = [1.0; 4];
        let square = [0.0; 16];
        let key_value = [0.0; 8];
        let input = [1.0, -2.0, 3.0, -4.0, -0.5, 0.25, 1.5, -3.0];

        let output = block
            .forward_prefill(&input, 2, zero_block_weights(&norm, &square, &key_value))
            .expect("valid complete block");
        assert_eq!(output, input);
    }

    #[test]
    fn cached_block_tokens_match_complete_block_prefill() {
        use crate::kv_cache::{KvCache, KvCacheConfig};

        let config = tiny_config();
        let block = super::TransformerBlock::from_config(&config).expect("valid block");
        let norm = [1.0; 4];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let key_value = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let zero = [0.0; 16];
        let weights = super::TransformerBlockWeights::new(
            &norm, &identity, &key_value, &key_value, &identity, &norm, &zero, &zero, &zero,
        );
        let input = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.25];
        let expected = block
            .forward_prefill(&input, 2, weights)
            .expect("fresh block prefill");

        let cache_config = KvCacheConfig::from_model(&config, 8).expect("valid cache config");
        let mut cache = KvCache::allocate(cache_config).expect("allocated cache");
        let mut actual = block
            .forward_cached_token(&input[..4], 0, &mut cache, weights)
            .expect("first cached token");
        actual.extend(
            block
                .forward_cached_token(&input[4..], 0, &mut cache, weights)
                .expect("second cached token"),
        );

        for (actual_value, expected_value) in actual.into_iter().zip(expected) {
            assert!((actual_value - expected_value).abs() <= 1.0e-6);
        }
        assert_eq!(cache.layer_sequence_length(0).expect("cache length"), 2);
    }

    #[test]
    fn generates_canonical_exported_layer_tensor_names() {
        let names = super::TransformerBlockWeights::tensor_names(7);
        assert_eq!(names[0], "layers.7.attention_norm.weight");
        assert_eq!(names[4], "layers.7.o_proj.weight");
        assert_eq!(names[8], "layers.7.down_proj.weight");
    }

    #[test]
    fn block_weight_bundle_preserves_borrowed_slices() {
        let scalar = [1.0];
        let weights = super::TransformerBlockWeights::new(
            &scalar, &scalar, &scalar, &scalar, &scalar, &scalar, &scalar, &scalar, &scalar,
        );
        assert_eq!(weights.attention_norm(), &scalar);
        assert_eq!(weights.query(), &scalar);
        assert_eq!(weights.down(), &scalar);
    }

    #[test]
    fn adds_residual_values_elementwise() {
        let mut hidden_states = [1.0, 2.0, 3.0, 4.0];
        add_residual_in_place(&mut hidden_states, &[0.5, -1.0, 2.0, 0.0])
            .expect("matching residual tensors");
        assert_eq!(hidden_states, [1.5, 1.0, 5.0, 4.0]);
    }

    #[test]
    fn rejects_mismatched_residual_lengths() {
        assert!(add_residual_in_place(&mut [0.0; 2], &[0.0; 3]).is_err());
    }
}
