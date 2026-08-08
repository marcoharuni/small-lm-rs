//! Key/value cache storage, lifecycle, and shape contracts.

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::tensor::validate_matrix;

/// Dimensions required to allocate a generation key/value cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheConfig {
    /// Number of transformer layers represented in the cache.
    pub num_layers: usize,
    /// Maximum number of cached token positions.
    pub max_sequence_length: usize,
    /// Number of key/value heads per layer.
    pub num_key_value_heads: usize,
    /// Width of each key/value head.
    pub head_dimension: usize,
}

impl KvCacheConfig {
    /// Derive cache dimensions from a model configuration and requested limit.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when the model is invalid or the
    /// requested limit is zero or exceeds the trained context length.
    pub fn from_model(config: &ModelConfig, max_sequence_length: usize) -> Result<Self> {
        config.validate()?;
        if max_sequence_length == 0 || max_sequence_length > config.context_length {
            return Err(EngineError::invalid_configuration(format!(
                "KV-cache max_sequence_length must be in 1..={}",
                config.context_length
            )));
        }
        Ok(Self {
            num_layers: config.num_layers,
            max_sequence_length,
            num_key_value_heads: config.num_key_value_heads,
            head_dimension: config.head_dimension,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.num_layers == 0
            || self.max_sequence_length == 0
            || self.num_key_value_heads == 0
            || self.head_dimension == 0
        {
            return Err(EngineError::invalid_configuration(
                "all KV-cache dimensions must be greater than zero",
            ));
        }
        self.key_value_width()?;
        Ok(())
    }

    fn key_value_width(&self) -> Result<usize> {
        self.num_key_value_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| {
                EngineError::invalid_configuration("KV-cache head width overflows usize")
            })
    }
}

#[derive(Debug)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    sequence_length: usize,
}

impl LayerKvCache {
    fn new(capacity: usize) -> Self {
        Self {
            keys: vec![0.0; capacity],
            values: vec![0.0; capacity],
            sequence_length: 0,
        }
    }
}

/// Engine-owned, dense per-layer key/value tensors for autoregressive decoding.
///
/// Each layer stores row-major `[max_sequence_length, kv_heads, head_dim]`
/// keys and values. Only the prefix indicated by that layer's sequence length
/// is initialized and visible through the public accessors.
#[derive(Debug)]
pub struct KvCache {
    config: KvCacheConfig,
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    /// Allocate zero-initialized storage for every configured layer.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for zero dimensions or any
    /// overflowing cache shape.
    pub fn allocate(config: KvCacheConfig) -> Result<Self> {
        config.validate()?;
        let width = config.key_value_width()?;
        let layer_capacity = config
            .max_sequence_length
            .checked_mul(width)
            .ok_or_else(|| {
                EngineError::invalid_configuration("KV-cache layer size overflows usize")
            })?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LayerKvCache::new(layer_capacity));
        }
        Ok(Self { config, layers })
    }

    /// Return the cache allocation settings.
    #[must_use]
    pub const fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Return the number of token positions completed across every layer.
    ///
    /// During an in-progress multi-layer forward pass, this is the minimum
    /// layer length. Completed model operations leave all layers synchronized.
    #[must_use]
    pub fn sequence_length(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.sequence_length)
            .min()
            .unwrap_or_default()
    }

    /// Return whether every layer currently contains zero token positions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(|layer| layer.sequence_length == 0)
    }

    /// Return whether every layer currently has the same sequence length.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        let expected = self.sequence_length();
        self.layers
            .iter()
            .all(|layer| layer.sequence_length == expected)
    }

    /// Return the number of cached positions in one layer.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when `layer_index` is out of range.
    pub fn layer_sequence_length(&self, layer_index: usize) -> Result<usize> {
        self.layers
            .get(layer_index)
            .map(|layer| layer.sequence_length)
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "KV cache",
                    format!(
                        "layer index {layer_index} is outside 0..{}",
                        self.layers.len()
                    ),
                )
            })
    }

    /// Append token-major key and value rows to one layer.
    ///
    /// `keys` and `values` must both have shape
    /// `[token_count, num_key_value_heads * head_dimension]`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an invalid layer, inconsistent
    /// tensor shape, non-finite values, arithmetic overflow, or exhausted
    /// cache capacity. Validation completes before storage is mutated.
    pub fn append_layer(
        &mut self,
        layer_index: usize,
        keys: &[f32],
        values: &[f32],
        token_count: usize,
    ) -> Result<()> {
        let width = self.config.key_value_width()?;
        validate_matrix("KV-cache keys", keys, token_count, width)?;
        validate_matrix("KV-cache values", values, token_count, width)?;
        if keys.iter().chain(values).any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_input(
                "KV cache",
                "all appended key/value values must be finite",
            ));
        }

        let current_length = self.layer_sequence_length(layer_index)?;
        let new_length = current_length.checked_add(token_count).ok_or_else(|| {
            EngineError::invalid_input("KV cache", "sequence length overflows usize")
        })?;
        if new_length > self.config.max_sequence_length {
            return Err(EngineError::invalid_input(
                "KV cache",
                format!(
                    "layer {layer_index} would grow to {new_length} tokens, exceeding capacity {}",
                    self.config.max_sequence_length
                ),
            ));
        }

        let start = current_length.checked_mul(width).ok_or_else(|| {
            EngineError::invalid_input("KV cache", "append offset overflows usize")
        })?;
        let end = new_length
            .checked_mul(width)
            .ok_or_else(|| EngineError::invalid_input("KV cache", "append end overflows usize"))?;
        let layer_count = self.layers.len();
        let layer = self.layers.get_mut(layer_index).ok_or_else(|| {
            EngineError::invalid_input(
                "KV cache",
                format!("layer index {layer_index} is outside 0..{layer_count}"),
            )
        })?;
        layer.keys[start..end].copy_from_slice(keys);
        layer.values[start..end].copy_from_slice(values);
        layer.sequence_length = new_length;
        Ok(())
    }

    /// Return the initialized key prefix for one layer.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when the layer index is out of range.
    pub fn layer_keys(&self, layer_index: usize) -> Result<&[f32]> {
        self.layer_values_prefix(layer_index, true)
    }

    /// Return the initialized value prefix for one layer.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when the layer index is out of range.
    pub fn layer_values(&self, layer_index: usize) -> Result<&[f32]> {
        self.layer_values_prefix(layer_index, false)
    }

    /// Return one key head at one cached token position.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an invalid layer, token position, or
    /// key/value-head index.
    pub fn key_head(
        &self,
        layer_index: usize,
        token_position: usize,
        key_value_head: usize,
    ) -> Result<&[f32]> {
        self.head_slice(layer_index, token_position, key_value_head, true)
    }

    /// Return one value head at one cached token position.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an invalid layer, token position, or
    /// key/value-head index.
    pub fn value_head(
        &self,
        layer_index: usize,
        token_position: usize,
        key_value_head: usize,
    ) -> Result<&[f32]> {
        self.head_slice(layer_index, token_position, key_value_head, false)
    }

    /// Return the number of token positions still available in every layer.
    ///
    /// During a partially completed operation this uses the longest layer, so
    /// the result never overstates safe remaining capacity.
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        let used = self
            .layers
            .iter()
            .map(|layer| layer.sequence_length)
            .max()
            .unwrap_or_default();
        self.config.max_sequence_length.saturating_sub(used)
    }

    /// Clear all logical sequence state while retaining allocated storage.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.sequence_length = 0;
        }
    }

    pub(crate) fn truncate_all(&mut self, sequence_length: usize) -> Result<()> {
        if let Some((layer_index, current_length)) = self
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| (index, layer.sequence_length))
            .find(|(_, current_length)| *current_length < sequence_length)
        {
            return Err(EngineError::invalid_input(
                "KV cache",
                format!(
                    "cannot restore all layers to {sequence_length}: layer {layer_index} has length {current_length}"
                ),
            ));
        }
        for layer in &mut self.layers {
            layer.sequence_length = sequence_length;
        }
        Ok(())
    }

    pub(crate) fn truncate_layer(
        &mut self,
        layer_index: usize,
        sequence_length: usize,
    ) -> Result<()> {
        let current_length = self.layer_sequence_length(layer_index)?;
        if sequence_length > current_length {
            return Err(EngineError::invalid_input(
                "KV cache",
                format!(
                    "cannot extend layer {layer_index} from {current_length} to {sequence_length} while truncating"
                ),
            ));
        }
        let layer_count = self.layers.len();
        let layer = self.layers.get_mut(layer_index).ok_or_else(|| {
            EngineError::invalid_input(
                "KV cache",
                format!("layer index {layer_index} is outside 0..{layer_count}"),
            )
        })?;
        layer.sequence_length = sequence_length;
        Ok(())
    }

    fn layer_values_prefix(&self, layer_index: usize, keys: bool) -> Result<&[f32]> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            EngineError::invalid_input(
                "KV cache",
                format!(
                    "layer index {layer_index} is outside 0..{}",
                    self.layers.len()
                ),
            )
        })?;
        let width = self.config.key_value_width()?;
        let initialized = layer.sequence_length.checked_mul(width).ok_or_else(|| {
            EngineError::invalid_input("KV cache", "initialized prefix overflows usize")
        })?;
        let storage = if keys { &layer.keys } else { &layer.values };
        Ok(&storage[..initialized])
    }

    fn head_slice(
        &self,
        layer_index: usize,
        token_position: usize,
        key_value_head: usize,
        keys: bool,
    ) -> Result<&[f32]> {
        let sequence_length = self.layer_sequence_length(layer_index)?;
        if token_position >= sequence_length {
            return Err(EngineError::invalid_input(
                "KV cache",
                format!("token position {token_position} is outside 0..{sequence_length}"),
            ));
        }
        if key_value_head >= self.config.num_key_value_heads {
            return Err(EngineError::invalid_input(
                "KV cache",
                format!(
                    "key/value head {key_value_head} is outside 0..{}",
                    self.config.num_key_value_heads
                ),
            ));
        }

        let width = self.config.key_value_width()?;
        let token_offset = token_position.checked_mul(width).ok_or_else(|| {
            EngineError::invalid_input("KV cache", "token offset overflows usize")
        })?;
        let head_offset = key_value_head
            .checked_mul(self.config.head_dimension)
            .ok_or_else(|| EngineError::invalid_input("KV cache", "head offset overflows usize"))?;
        let start = token_offset
            .checked_add(head_offset)
            .ok_or_else(|| EngineError::invalid_input("KV cache", "head start overflows usize"))?;
        let end = start
            .checked_add(self.config.head_dimension)
            .ok_or_else(|| EngineError::invalid_input("KV cache", "head end overflows usize"))?;
        let storage = if keys {
            self.layer_keys(layer_index)?
        } else {
            self.layer_values(layer_index)?
        };
        storage.get(start..end).ok_or_else(|| {
            EngineError::invalid_input("KV cache", "cached head storage is truncated")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{KvCache, KvCacheConfig};
    use crate::config::ModelConfig;

    fn tiny_model_config() -> ModelConfig {
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
            expected_parameter_count: 1,
        }
    }

    #[test]
    fn allocates_dense_storage_for_every_layer() {
        let config = KvCacheConfig::from_model(&tiny_model_config(), 4).expect("valid cache");
        let cache = KvCache::allocate(config).expect("allocated cache");
        assert_eq!(cache.config(), &config);
        assert_eq!(cache.sequence_length(), 0);
        assert!(cache.is_empty());
        assert!(cache.is_synchronized());
        assert!(cache.layer_keys(0).expect("layer zero").is_empty());
        assert!(cache.layer_values(1).expect("layer one").is_empty());
    }

    #[test]
    fn appends_and_indexes_token_major_key_value_heads() {
        let config = KvCacheConfig::from_model(&tiny_model_config(), 4).expect("valid cache");
        let mut cache = KvCache::allocate(config).expect("allocated cache");
        cache
            .append_layer(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2)
            .expect("valid append");

        assert_eq!(cache.layer_sequence_length(0).expect("layer length"), 2);
        assert_eq!(cache.layer_sequence_length(1).expect("layer length"), 0);
        assert!(!cache.is_synchronized());
        assert_eq!(cache.key_head(0, 1, 0).expect("second key"), &[3.0, 4.0]);
        assert_eq!(cache.value_head(0, 0, 0).expect("first value"), &[5.0, 6.0]);
    }

    #[test]
    fn capacity_failure_does_not_mutate_the_cache() {
        let config = KvCacheConfig::from_model(&tiny_model_config(), 1).expect("valid cache");
        let mut cache = KvCache::allocate(config).expect("allocated cache");
        cache
            .append_layer(0, &[1.0, 2.0], &[3.0, 4.0], 1)
            .expect("first token");
        assert!(cache.append_layer(0, &[5.0, 6.0], &[7.0, 8.0], 1).is_err());
        assert_eq!(cache.layer_sequence_length(0).expect("layer length"), 1);
        assert_eq!(cache.key_head(0, 0, 0).expect("preserved key"), &[1.0, 2.0]);
    }

    #[test]
    fn remaining_capacity_tracks_the_longest_layer() {
        let config = KvCacheConfig::from_model(&tiny_model_config(), 4).expect("valid cache");
        let mut cache = KvCache::allocate(config).expect("allocated cache");
        assert_eq!(cache.remaining_capacity(), 4);
        cache
            .append_layer(0, &[1.0, 2.0], &[3.0, 4.0], 1)
            .expect("valid append");
        assert_eq!(cache.remaining_capacity(), 3);
    }

    #[test]
    fn whole_cache_truncation_is_transactional() {
        let config = KvCacheConfig::from_model(&tiny_model_config(), 4).expect("valid cache");
        let mut cache = KvCache::allocate(config).expect("allocated cache");
        for layer in 0..config.num_layers {
            cache
                .append_layer(layer, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2)
                .expect("valid append");
        }
        cache.truncate_all(1).expect("valid rollback");
        assert!(cache.is_synchronized());
        assert_eq!(cache.sequence_length(), 1);
        assert!(cache.truncate_all(2).is_err());
        assert_eq!(cache.sequence_length(), 1);
    }

    #[test]
    fn clear_retains_storage_and_resets_all_lengths() {
        let config = KvCacheConfig::from_model(&tiny_model_config(), 4).expect("valid cache");
        let mut cache = KvCache::allocate(config).expect("allocated cache");
        for layer in 0..config.num_layers {
            cache
                .append_layer(layer, &[1.0, 2.0], &[3.0, 4.0], 1)
                .expect("valid append");
        }
        assert_eq!(cache.sequence_length(), 1);
        assert!(cache.is_synchronized());
        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.is_synchronized());
    }
}
