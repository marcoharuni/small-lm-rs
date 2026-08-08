//! Grouped-query causal self-attention preparation and execution.

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::kv_cache::KvCache;
use crate::linear::linear;
use crate::rope::RotaryEmbedding;
use crate::softmax::softmax_rows;
use crate::tensor::validate_matrix;

/// Borrowed projection weights for one grouped-query attention layer.
///
/// Every matrix follows the exported `[out_features, in_features]` layout.
#[derive(Clone, Copy, Debug)]
pub struct AttentionWeights<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    output: &'a [f32],
}

impl<'a> AttentionWeights<'a> {
    /// Bundle query, key, value, and output projection matrices.
    #[must_use]
    pub const fn new(
        query: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        output: &'a [f32],
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
        }
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

    /// Return the output-projection matrix.
    #[must_use]
    pub const fn output(&self) -> &'a [f32] {
        self.output
    }
}

/// Shape metadata for the model's grouped-query attention layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedQueryAttention {
    hidden_size: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dimension: usize,
}

/// Projected query, key, and value tensors in token-major head layout.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedQkv {
    sequence_length: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dimension: usize,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl ProjectedQkv {
    /// Return the number of projected token positions.
    #[must_use]
    pub const fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Return flattened query values shaped `[tokens, query_heads, head_dim]`.
    #[must_use]
    pub fn query_values(&self) -> &[f32] {
        &self.query
    }

    /// Return flattened key values shaped `[tokens, kv_heads, head_dim]`.
    #[must_use]
    pub fn key_values(&self) -> &[f32] {
        &self.key
    }

    /// Return flattened value values shaped `[tokens, kv_heads, head_dim]`.
    #[must_use]
    pub fn value_values(&self) -> &[f32] {
        &self.value
    }

    /// Return the number of query heads represented by each physical KV head.
    #[must_use]
    pub const fn query_group_size(&self) -> usize {
        self.num_query_heads / self.num_key_value_heads
    }

    /// Map a logical query head to its physical key/value head.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when the query-head index is out of range.
    pub fn key_value_head_for_query(&self, query_head: usize) -> Result<usize> {
        if query_head >= self.num_query_heads {
            return Err(EngineError::invalid_input(
                "GQA head mapping",
                format!(
                    "query head {query_head} is outside 0..{}",
                    self.num_query_heads
                ),
            ));
        }
        Ok(query_head / self.query_group_size())
    }

    /// Return one query head at one token position.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an out-of-range token or head index.
    pub fn query_head(&self, token_position: usize, query_head: usize) -> Result<&[f32]> {
        head_slice(
            "query head",
            &self.query,
            self.sequence_length,
            self.num_query_heads,
            self.head_dimension,
            token_position,
            query_head,
        )
    }

    /// Return one key head at one token position.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an out-of-range token or head index.
    pub fn key_head(&self, token_position: usize, key_value_head: usize) -> Result<&[f32]> {
        head_slice(
            "key head",
            &self.key,
            self.sequence_length,
            self.num_key_value_heads,
            self.head_dimension,
            token_position,
            key_value_head,
        )
    }

    /// Return one value head at one token position.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an out-of-range token or head index.
    pub fn value_head(&self, token_position: usize, key_value_head: usize) -> Result<&[f32]> {
        head_slice(
            "value head",
            &self.value,
            self.sequence_length,
            self.num_key_value_heads,
            self.head_dimension,
            token_position,
            key_value_head,
        )
    }

    /// Compute scaled causal query-key scores.
    ///
    /// Scores are flattened in `[query_heads, query_tokens, source_tokens]`
    /// order. Future source positions are filled with `f32::MIN`, matching the
    /// JAX reference mask before softmax.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when projected tensor metadata is
    /// inconsistent or a score-buffer dimension overflows.
    pub fn scaled_causal_scores(&self) -> Result<Vec<f32>> {
        let score_rows = self
            .num_query_heads
            .checked_mul(self.sequence_length)
            .ok_or_else(|| {
                EngineError::invalid_input("attention scores", "score row count overflows usize")
            })?;
        let score_count = score_rows
            .checked_mul(self.sequence_length)
            .ok_or_else(|| {
                EngineError::invalid_input("attention scores", "score count overflows usize")
            })?;
        let mut scores = vec![f32::MIN; score_count];
        let scale = (self.head_dimension as f32).sqrt().recip();

        for query_head in 0..self.num_query_heads {
            let key_value_head = self.key_value_head_for_query(query_head)?;

            for query_position in 0..self.sequence_length {
                let query = self.query_head(query_position, query_head)?;
                let row_index = query_head
                    .checked_mul(self.sequence_length)
                    .and_then(|offset| offset.checked_add(query_position))
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "attention scores",
                            "score row index overflows usize",
                        )
                    })?;
                let row_start = row_index.checked_mul(self.sequence_length).ok_or_else(|| {
                    EngineError::invalid_input(
                        "attention scores",
                        "score row offset overflows usize",
                    )
                })?;

                for source_position in 0..=query_position {
                    let key = self.key_head(source_position, key_value_head)?;
                    let dot_product = query
                        .iter()
                        .zip(key)
                        .fold(0.0_f32, |sum, (&query_value, &key_value)| {
                            query_value.mul_add(key_value, sum)
                        });
                    scores[row_start + source_position] = dot_product * scale;
                }
            }
        }

        Ok(scores)
    }

    /// Compute row-wise causal attention probabilities.
    ///
    /// The returned layout is `[query_heads, query_tokens, source_tokens]`.
    /// Masked future positions become exactly zero after stable softmax.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when score dimensions overflow or a
    /// probability row cannot be normalized.
    pub fn causal_probabilities(&self) -> Result<Vec<f32>> {
        let mut probabilities = self.scaled_causal_scores()?;
        let rows = self
            .num_query_heads
            .checked_mul(self.sequence_length)
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "attention probabilities",
                    "probability row count overflows usize",
                )
            })?;
        softmax_rows(&mut probabilities, rows, self.sequence_length)?;
        Ok(probabilities)
    }

    /// Aggregate physical value heads using logical-query probabilities.
    ///
    /// `probabilities` must use `[query_heads, query_tokens, source_tokens]`
    /// layout. The returned tensor is token-major
    /// `[query_tokens, query_heads, head_dimension]`, which is also
    /// `[query_tokens, hidden_size]` when flattened.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an inconsistent probability tensor,
    /// non-finite/negative probability, or an overflowing output dimension.
    pub fn attend_values(&self, probabilities: &[f32]) -> Result<Vec<f32>> {
        let probability_rows = self
            .num_query_heads
            .checked_mul(self.sequence_length)
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "attention values",
                    "probability row count overflows usize",
                )
            })?;
        validate_matrix(
            "attention probabilities",
            probabilities,
            probability_rows,
            self.sequence_length,
        )?;
        if probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
        {
            return Err(EngineError::invalid_input(
                "attention probabilities",
                "all probabilities must be finite and non-negative",
            ));
        }

        let hidden_size = self
            .num_query_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| {
                EngineError::invalid_input("attention values", "hidden size overflows usize")
            })?;
        let output_count = self
            .sequence_length
            .checked_mul(hidden_size)
            .ok_or_else(|| {
                EngineError::invalid_input("attention values", "output size overflows usize")
            })?;
        let mut attended = vec![0.0_f32; output_count];

        for query_position in 0..self.sequence_length {
            for query_head in 0..self.num_query_heads {
                let key_value_head = self.key_value_head_for_query(query_head)?;
                let probability_row = query_head
                    .checked_mul(self.sequence_length)
                    .and_then(|offset| offset.checked_add(query_position))
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "attention values",
                            "probability row index overflows usize",
                        )
                    })?;
                let probability_start = probability_row
                    .checked_mul(self.sequence_length)
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "attention values",
                            "probability row offset overflows usize",
                        )
                    })?;
                let probability_end = probability_start
                    .checked_add(self.sequence_length)
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "attention values",
                            "probability row end overflows usize",
                        )
                    })?;
                let probability_values = probabilities
                    .get(probability_start..probability_end)
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "attention values",
                            "probability tensor is truncated",
                        )
                    })?;

                let output_head_index = query_position
                    .checked_mul(self.num_query_heads)
                    .and_then(|offset| offset.checked_add(query_head))
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "attention values",
                            "output head index overflows usize",
                        )
                    })?;
                let output_start = output_head_index
                    .checked_mul(self.head_dimension)
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "attention values",
                            "output head offset overflows usize",
                        )
                    })?;
                let output_end =
                    output_start
                        .checked_add(self.head_dimension)
                        .ok_or_else(|| {
                            EngineError::invalid_input(
                                "attention values",
                                "output head end overflows usize",
                            )
                        })?;
                let output_head = attended.get_mut(output_start..output_end).ok_or_else(|| {
                    EngineError::invalid_input("attention values", "output tensor is truncated")
                })?;

                for (source_position, &probability) in probability_values.iter().enumerate() {
                    let value_head = self.value_head(source_position, key_value_head)?;
                    for (output_value, &value) in output_head.iter_mut().zip(value_head) {
                        *output_value = value.mul_add(probability, *output_value);
                    }
                }
            }
        }

        Ok(attended)
    }

    /// Apply zero-based absolute-position RoPE to every query and key head.
    ///
    /// Value heads are intentionally left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when the rotary head width differs from
    /// the projected head width or a projected tensor has an invalid shape.
    pub fn apply_rotary(&mut self, rotary: &RotaryEmbedding) -> Result<()> {
        self.apply_rotary_with_offset(rotary, 0)
    }

    /// Apply RoPE using positions beginning at `position_offset`.
    ///
    /// This is the decode-path form of rotary embedding: a one-token projection
    /// uses the number of already cached tokens as its absolute position.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for incompatible head dimensions,
    /// malformed projected tensors, or an overflowing absolute position.
    pub fn apply_rotary_with_offset(
        &mut self,
        rotary: &RotaryEmbedding,
        position_offset: usize,
    ) -> Result<()> {
        if rotary.head_dimension() != self.head_dimension {
            return Err(EngineError::invalid_input(
                "attention RoPE",
                format!(
                    "rotary head dimension {} does not match projected dimension {}",
                    rotary.head_dimension(),
                    self.head_dimension
                ),
            ));
        }

        let query_stride = self
            .num_query_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "query stride overflows usize")
            })?;
        let key_stride = self
            .num_key_value_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "key stride overflows usize")
            })?;

        for local_position in 0..self.sequence_length {
            let absolute_position =
                position_offset.checked_add(local_position).ok_or_else(|| {
                    EngineError::invalid_input(
                        "attention RoPE",
                        "absolute position overflows usize",
                    )
                })?;
            let query_start = local_position.checked_mul(query_stride).ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "query offset overflows usize")
            })?;
            let query_end = query_start.checked_add(query_stride).ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "query end overflows usize")
            })?;
            let query_heads = self.query.get_mut(query_start..query_end).ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "query tensor is truncated")
            })?;
            rotary.apply_heads_in_place(query_heads, self.num_query_heads, absolute_position)?;

            let key_start = local_position.checked_mul(key_stride).ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "key offset overflows usize")
            })?;
            let key_end = key_start.checked_add(key_stride).ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "key end overflows usize")
            })?;
            let key_heads = self.key.get_mut(key_start..key_end).ok_or_else(|| {
                EngineError::invalid_input("attention RoPE", "key tensor is truncated")
            })?;
            rotary.apply_heads_in_place(key_heads, self.num_key_value_heads, absolute_position)?;
        }
        Ok(())
    }
}

fn head_slice<'a>(
    component: &'static str,
    values: &'a [f32],
    sequence_length: usize,
    num_heads: usize,
    head_dimension: usize,
    token_position: usize,
    head_index: usize,
) -> Result<&'a [f32]> {
    if token_position >= sequence_length {
        return Err(EngineError::invalid_input(
            component,
            format!("token position {token_position} is outside 0..{sequence_length}"),
        ));
    }
    if head_index >= num_heads {
        return Err(EngineError::invalid_input(
            component,
            format!("head index {head_index} is outside 0..{num_heads}"),
        ));
    }

    let token_stride = num_heads
        .checked_mul(head_dimension)
        .ok_or_else(|| EngineError::invalid_input(component, "head stride overflows usize"))?;
    let token_offset = token_position
        .checked_mul(token_stride)
        .ok_or_else(|| EngineError::invalid_input(component, "token offset overflows usize"))?;
    let head_offset = head_index
        .checked_mul(head_dimension)
        .ok_or_else(|| EngineError::invalid_input(component, "head offset overflows usize"))?;
    let start = token_offset
        .checked_add(head_offset)
        .ok_or_else(|| EngineError::invalid_input(component, "slice offset overflows usize"))?;
    let end = start
        .checked_add(head_dimension)
        .ok_or_else(|| EngineError::invalid_input(component, "slice end overflows usize"))?;

    values
        .get(start..end)
        .ok_or_else(|| EngineError::invalid_input(component, "projected tensor is truncated"))
}

impl GroupedQueryAttention {
    /// Build attention metadata from a validated model configuration.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when model invariants fail.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            hidden_size: config.hidden_size,
            num_query_heads: config.num_query_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dimension: config.head_dimension,
        })
    }

    /// Return the number of query heads.
    #[must_use]
    pub const fn num_query_heads(&self) -> usize {
        self.num_query_heads
    }

    /// Return the number of key/value heads.
    #[must_use]
    pub const fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    /// Return the width of one attention head.
    #[must_use]
    pub const fn head_dimension(&self) -> usize {
        self.head_dimension
    }

    /// Project hidden states into query, key, and value tensors.
    ///
    /// Hidden states are row-major `[sequence_length, hidden_size]`. Exported
    /// weights use `[out_features, in_features]`. Projection matmuls follow the
    /// model's BF16-input/weight and FP32-accumulation contract.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an empty sequence, inconsistent hidden
    /// states, or incorrectly sized projection weights.
    pub fn project_qkv(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        query_weight: &[f32],
        key_weight: &[f32],
        value_weight: &[f32],
    ) -> Result<ProjectedQkv> {
        validate_matrix(
            "attention hidden states",
            hidden_states,
            sequence_length,
            self.hidden_size,
        )?;

        let key_value_width = self
            .num_key_value_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| {
                EngineError::invalid_input("attention", "key/value width overflows usize")
            })?;

        let query = linear(
            hidden_states,
            sequence_length,
            self.hidden_size,
            query_weight,
            self.hidden_size,
        )?;
        let key = linear(
            hidden_states,
            sequence_length,
            self.hidden_size,
            key_weight,
            key_value_width,
        )?;
        let value = linear(
            hidden_states,
            sequence_length,
            self.hidden_size,
            value_weight,
            key_value_width,
        )?;

        Ok(ProjectedQkv {
            sequence_length,
            num_query_heads: self.num_query_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dimension: self.head_dimension,
            query,
            key,
            value,
        })
    }

    /// Apply the bias-free attention output projection.
    ///
    /// `attended` is token-major `[sequence_length, hidden_size]`, and
    /// `output_weight` is `[hidden_size, hidden_size]`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an inconsistent attended tensor or
    /// output-projection matrix.
    pub fn project_output(
        &self,
        attended: &[f32],
        sequence_length: usize,
        output_weight: &[f32],
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "attention output",
            attended,
            sequence_length,
            self.hidden_size,
        )?;
        linear(
            attended,
            sequence_length,
            self.hidden_size,
            output_weight,
            self.hidden_size,
        )
    }

    fn validate_cache_shape(&self, layer_index: usize, cache: &KvCache) -> Result<()> {
        let config = cache.config();
        if config.num_key_value_heads != self.num_key_value_heads
            || config.head_dimension != self.head_dimension
        {
            return Err(EngineError::invalid_input(
                "cached attention",
                format!(
                    "cache shape uses {} heads of width {}, expected {} heads of width {}",
                    config.num_key_value_heads,
                    config.head_dimension,
                    self.num_key_value_heads,
                    self.head_dimension
                ),
            ));
        }
        cache.layer_sequence_length(layer_index)?;
        Ok(())
    }

    fn attend_cached_query(
        &self,
        query: &[f32],
        layer_index: usize,
        cache: &KvCache,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "cached attention query",
            query,
            self.num_query_heads,
            self.head_dimension,
        )?;
        let source_length = cache.layer_sequence_length(layer_index)?;
        if source_length == 0 {
            return Err(EngineError::invalid_input(
                "cached attention",
                "at least one cached key/value position is required",
            ));
        }

        let score_count = self
            .num_query_heads
            .checked_mul(source_length)
            .ok_or_else(|| {
                EngineError::invalid_input("cached attention", "score count overflows usize")
            })?;
        let mut probabilities = vec![0.0_f32; score_count];
        let scale = (self.head_dimension as f32).sqrt().recip();

        for (query_head, score_row) in probabilities.chunks_exact_mut(source_length).enumerate() {
            let query_start = query_head.checked_mul(self.head_dimension).ok_or_else(|| {
                EngineError::invalid_input("cached attention", "query offset overflows usize")
            })?;
            let query_end = query_start
                .checked_add(self.head_dimension)
                .ok_or_else(|| {
                    EngineError::invalid_input("cached attention", "query end overflows usize")
                })?;
            let query_values = query.get(query_start..query_end).ok_or_else(|| {
                EngineError::invalid_input("cached attention", "query tensor is truncated")
            })?;
            let key_value_head = query_head / (self.num_query_heads / self.num_key_value_heads);

            for (source_position, score) in score_row.iter_mut().enumerate() {
                let key = cache.key_head(layer_index, source_position, key_value_head)?;
                let dot_product = query_values
                    .iter()
                    .zip(key)
                    .fold(0.0_f32, |sum, (&query_value, &key_value)| {
                        query_value.mul_add(key_value, sum)
                    });
                *score = dot_product * scale;
            }
        }
        softmax_rows(&mut probabilities, self.num_query_heads, source_length)?;

        let mut attended = vec![0.0_f32; self.hidden_size];
        for (query_head, output_head) in attended.chunks_exact_mut(self.head_dimension).enumerate()
        {
            let probability_start = query_head.checked_mul(source_length).ok_or_else(|| {
                EngineError::invalid_input("cached attention", "probability offset overflows usize")
            })?;
            let probability_end =
                probability_start
                    .checked_add(source_length)
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "cached attention",
                            "probability end overflows usize",
                        )
                    })?;
            let probability_row = &probabilities[probability_start..probability_end];
            let key_value_head = query_head / (self.num_query_heads / self.num_key_value_heads);

            for (source_position, &probability) in probability_row.iter().enumerate() {
                let value = cache.value_head(layer_index, source_position, key_value_head)?;
                for (output_value, &value_element) in output_head.iter_mut().zip(value) {
                    *output_value = value_element.mul_add(probability, *output_value);
                }
            }
        }
        Ok(attended)
    }

    /// Execute one-token attention against a layer's existing cache and append
    /// the current rotated key and value before attending.
    ///
    /// The hidden-state input has shape `[1, hidden_size]`. RoPE uses the
    /// layer's pre-append sequence length as the current absolute position.
    ///
    /// # Errors
    ///
    /// Returns validation, projection, cache-capacity, rotary, softmax, or
    /// output-projection errors. A failed attention calculation rolls the
    /// layer back to its original logical length.
    pub fn forward_cached_token(
        &self,
        hidden_state: &[f32],
        layer_index: usize,
        cache: &mut KvCache,
        weights: AttentionWeights<'_>,
        rotary: &RotaryEmbedding,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "cached attention hidden state",
            hidden_state,
            1,
            self.hidden_size,
        )?;
        self.validate_cache_shape(layer_index, cache)?;
        validate_matrix(
            "cached attention output weight",
            weights.output(),
            self.hidden_size,
            self.hidden_size,
        )?;

        let absolute_position = cache.layer_sequence_length(layer_index)?;
        let mut projected = self.project_qkv(
            hidden_state,
            1,
            weights.query(),
            weights.key(),
            weights.value(),
        )?;
        projected.apply_rotary_with_offset(rotary, absolute_position)?;
        cache.append_layer(
            layer_index,
            projected.key_values(),
            projected.value_values(),
            1,
        )?;

        let attended = match self.attend_cached_query(projected.query_values(), layer_index, cache)
        {
            Ok(attended) => attended,
            Err(error) => {
                cache.truncate_layer(layer_index, absolute_position)?;
                return Err(error);
            }
        };
        match self.project_output(&attended, 1, weights.output()) {
            Ok(output) => Ok(output),
            Err(error) => {
                cache.truncate_layer(layer_index, absolute_position)?;
                Err(error)
            }
        }
    }

    /// Run fresh-sequence attention and commit its rotated keys and values to
    /// an empty cache layer after successful output computation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when the target layer is already
    /// populated, cache dimensions differ from the attention configuration, or
    /// any projection, rotary, causal-attention, or cache operation fails.
    pub fn forward_prefill_with_cache(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        layer_index: usize,
        cache: &mut KvCache,
        weights: AttentionWeights<'_>,
        rotary: &RotaryEmbedding,
    ) -> Result<Vec<f32>> {
        self.validate_cache_shape(layer_index, cache)?;
        if cache.layer_sequence_length(layer_index)? != 0 {
            return Err(EngineError::invalid_input(
                "cached attention prefill",
                format!("cache layer {layer_index} must be empty before prompt prefill"),
            ));
        }

        let mut projected = self.project_qkv(
            hidden_states,
            sequence_length,
            weights.query(),
            weights.key(),
            weights.value(),
        )?;
        projected.apply_rotary(rotary)?;
        let probabilities = projected.causal_probabilities()?;
        let attended = projected.attend_values(&probabilities)?;
        let output = self.project_output(&attended, sequence_length, weights.output())?;
        cache.append_layer(
            layer_index,
            projected.key_values(),
            projected.value_values(),
            sequence_length,
        )?;
        Ok(output)
    }

    /// Run complete causal grouped-query attention for a fresh sequence.
    ///
    /// This performs Q/K/V projection, zero-based absolute-position RoPE,
    /// scaled causal scoring, stable softmax, grouped value aggregation, and
    /// output projection. KV-cache decoding is implemented separately.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for inconsistent tensors, projection
    /// weights, sequence length, or rotary configuration.
    pub fn forward_prefill(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        weights: AttentionWeights<'_>,
        rotary: &RotaryEmbedding,
    ) -> Result<Vec<f32>> {
        let mut projected = self.project_qkv(
            hidden_states,
            sequence_length,
            weights.query(),
            weights.key(),
            weights.value(),
        )?;
        projected.apply_rotary(rotary)?;
        let probabilities = projected.causal_probabilities()?;
        let attended = projected.attend_values(&probabilities)?;
        self.project_output(&attended, sequence_length, weights.output())
    }

    /// Run causal grouped-query attention over row-major hidden states.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidInput`] when the slice shape is invalid and
    /// [`EngineError::NotImplemented`] until the complete attention kernel is
    /// available.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        sequence_length: usize,
        _cache: Option<&mut KvCache>,
    ) -> Result<Vec<f32>> {
        validate_matrix(
            "attention hidden states",
            hidden_states,
            sequence_length,
            self.hidden_size,
        )?;
        Err(EngineError::not_implemented(
            "grouped-query attention CPU kernel",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{AttentionWeights, GroupedQueryAttention};
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
    fn projects_queries_keys_and_values_in_export_layout() {
        let attention =
            GroupedQueryAttention::from_config(&tiny_config()).expect("valid attention");
        let identity = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let key_weight = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ];
        let value_weight = [
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];

        let projected = attention
            .project_qkv(
                &[1.0, 2.0, 3.0, 4.0],
                1,
                &identity,
                &key_weight,
                &value_weight,
            )
            .expect("valid projections");

        assert_eq!(projected.sequence_length(), 1);
        assert_eq!(projected.query_values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(projected.key_values(), &[1.0, 2.0]);
        assert_eq!(projected.value_values(), &[3.0, 4.0]);
        assert_eq!(projected.query_group_size(), 2);
        assert_eq!(projected.key_value_head_for_query(0).expect("mapping"), 0);
        assert_eq!(projected.key_value_head_for_query(1).expect("mapping"), 0);
        assert_eq!(
            projected.query_head(0, 1).expect("second query head"),
            &[3.0, 4.0]
        );
        assert_eq!(projected.key_head(0, 0).expect("key head"), &[1.0, 2.0]);
        assert_eq!(projected.value_head(0, 0).expect("value head"), &[3.0, 4.0]);
        assert!(projected.query_head(1, 0).is_err());
        assert!(projected.key_value_head_for_query(2).is_err());
    }

    #[test]
    fn computes_scaled_scores_with_a_strict_causal_mask() {
        let inverse_sqrt_two = 2.0_f32.sqrt().recip();
        let projected = super::ProjectedQkv {
            sequence_length: 2,
            num_query_heads: 2,
            num_key_value_heads: 1,
            head_dimension: 2,
            query: vec![
                1.0, 0.0, 0.0, 1.0, // token 0
                1.0, 1.0, 1.0, -1.0, // token 1
            ],
            key: vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
            ],
            value: vec![0.0; 4],
        };

        let scores = projected
            .scaled_causal_scores()
            .expect("valid causal scores");

        assert_eq!(scores.len(), 8);
        assert_close(scores[0], inverse_sqrt_two);
        assert_eq!(scores[1], f32::MIN);
        assert_close(scores[2], inverse_sqrt_two);
        assert_close(scores[3], inverse_sqrt_two);

        assert_close(scores[4], 0.0);
        assert_eq!(scores[5], f32::MIN);
        assert_close(scores[6], inverse_sqrt_two);
        assert_close(scores[7], -inverse_sqrt_two);
    }

    #[test]
    fn normalizes_causal_scores_per_query_row() {
        let projected = super::ProjectedQkv {
            sequence_length: 2,
            num_query_heads: 1,
            num_key_value_heads: 1,
            head_dimension: 2,
            query: vec![
                1.0, 0.0, // token 0
                1.0, 1.0, // token 1
            ],
            key: vec![
                1.0, 0.0, // token 0
                0.0, 1.0, // token 1
            ],
            value: vec![0.0; 4],
        };

        let probabilities = projected
            .causal_probabilities()
            .expect("valid probabilities");

        assert_eq!(probabilities.len(), 4);
        assert_close(probabilities[0], 1.0);
        assert_close(probabilities[1], 0.0);
        assert_close(probabilities[2], 0.5);
        assert_close(probabilities[3], 0.5);
    }

    #[test]
    fn aggregates_shared_value_heads_in_token_major_layout() {
        let projected = super::ProjectedQkv {
            sequence_length: 2,
            num_query_heads: 2,
            num_key_value_heads: 1,
            head_dimension: 2,
            query: vec![0.0; 8],
            key: vec![0.0; 4],
            value: vec![
                10.0, 20.0, // token 0
                30.0, 40.0, // token 1
            ],
        };
        let probabilities = [
            1.0, 0.0, 0.25, 0.75, // query head 0
            1.0, 0.0, 0.5, 0.5, // query head 1
        ];

        let attended = projected
            .attend_values(&probabilities)
            .expect("valid aggregation");

        assert_eq!(
            attended,
            vec![
                10.0, 20.0, 10.0, 20.0, // token 0, both logical query heads
                25.0, 35.0, 20.0, 30.0, // token 1
            ]
        );
    }

    #[test]
    fn rotary_offset_matches_the_same_absolute_prefill_position() {
        use crate::rope::RotaryEmbedding;

        let rotary = RotaryEmbedding::new(2, 10_000.0).expect("valid RoPE");
        let mut full = super::ProjectedQkv {
            sequence_length: 2,
            num_query_heads: 1,
            num_key_value_heads: 1,
            head_dimension: 2,
            query: vec![1.0, 0.0, 1.0, 0.0],
            key: vec![0.0, 1.0, 0.0, 1.0],
            value: vec![0.0; 4],
        };
        let mut token = super::ProjectedQkv {
            sequence_length: 1,
            num_query_heads: 1,
            num_key_value_heads: 1,
            head_dimension: 2,
            query: vec![1.0, 0.0],
            key: vec![0.0, 1.0],
            value: vec![0.0; 2],
        };

        full.apply_rotary(&rotary).expect("full rotation");
        token
            .apply_rotary_with_offset(&rotary, 1)
            .expect("offset rotation");
        assert_eq!(token.query_values(), &full.query_values()[2..4]);
        assert_eq!(token.key_values(), &full.key_values()[2..4]);
    }

    #[test]
    fn applies_absolute_position_rope_to_queries_and_keys_only() {
        use crate::rope::RotaryEmbedding;

        let mut projected = super::ProjectedQkv {
            sequence_length: 2,
            num_query_heads: 2,
            num_key_value_heads: 1,
            head_dimension: 2,
            query: vec![
                1.0, 0.0, 0.0, 1.0, // position 0
                1.0, 0.0, 0.0, 1.0, // position 1
            ],
            key: vec![
                1.0, 0.0, // position 0
                0.0, 1.0, // position 1
            ],
            value: vec![
                5.0, 6.0, // position 0
                7.0, 8.0, // position 1
            ],
        };
        let original_values = projected.value.clone();
        let rotary = RotaryEmbedding::new(2, 10_000.0).expect("valid RoPE");
        projected.apply_rotary(&rotary).expect("valid rotation");

        assert_eq!(
            projected.query_head(0, 0).expect("position zero"),
            &[1.0, 0.0]
        );
        let (sin, cos) = 1.0_f32.sin_cos();
        assert_close(projected.query_head(1, 0).expect("rotated query")[0], cos);
        assert_close(projected.query_head(1, 0).expect("rotated query")[1], sin);
        assert_close(projected.key_head(1, 0).expect("rotated key")[0], -sin);
        assert_close(projected.key_head(1, 0).expect("rotated key")[1], cos);
        assert_eq!(projected.value_values(), original_values.as_slice());
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn projects_attended_heads_back_to_hidden_width() {
        let attention =
            GroupedQueryAttention::from_config(&tiny_config()).expect("valid attention");
        let identity = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let attended = [1.0, 2.0, 3.0, 4.0];

        let output = attention
            .project_output(&attended, 1, &identity)
            .expect("valid output projection");

        assert_eq!(output, attended);
    }

    #[test]
    fn runs_complete_single_token_causal_attention() {
        use crate::rope::RotaryEmbedding;

        let attention =
            GroupedQueryAttention::from_config(&tiny_config()).expect("valid attention");
        let rotary = RotaryEmbedding::new(2, 10_000.0).expect("valid RoPE");
        let identity = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let key_value_weight = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ];
        let weights =
            AttentionWeights::new(&identity, &key_value_weight, &key_value_weight, &identity);

        let output = attention
            .forward_prefill(&[1.0, 2.0, 3.0, 4.0], 1, weights, &rotary)
            .expect("valid causal prefill");

        assert_eq!(output, vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn cached_prompt_prefill_preserves_output_and_populates_rotated_keys() {
        use crate::kv_cache::{KvCache, KvCacheConfig};
        use crate::rope::RotaryEmbedding;

        let config = tiny_config();
        let attention = GroupedQueryAttention::from_config(&config).expect("valid attention");
        let rotary = RotaryEmbedding::new(2, 10_000.0).expect("valid RoPE");
        let identity = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let key_value = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ];
        let weights = AttentionWeights::new(&identity, &key_value, &key_value, &identity);
        let hidden_states = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.25];
        let expected = attention
            .forward_prefill(&hidden_states, 2, weights, &rotary)
            .expect("fresh prefill");

        let cache_config = KvCacheConfig::from_model(&config, 8).expect("valid cache config");
        let mut cache = KvCache::allocate(cache_config).expect("allocated cache");
        let actual = attention
            .forward_prefill_with_cache(&hidden_states, 2, 0, &mut cache, weights, &rotary)
            .expect("cached prompt prefill");

        assert_eq!(actual, expected);
        assert_eq!(cache.layer_sequence_length(0).expect("cache length"), 2);
        assert_eq!(cache.layer_keys(0).expect("cached keys").len(), 4);
        assert!(attention
            .forward_prefill_with_cache(&hidden_states, 2, 0, &mut cache, weights, &rotary,)
            .is_err());
    }

    #[test]
    fn sequential_cached_attention_matches_fresh_sequence_attention() {
        use crate::kv_cache::{KvCache, KvCacheConfig};
        use crate::rope::RotaryEmbedding;

        let config = tiny_config();
        let attention = GroupedQueryAttention::from_config(&config).expect("valid attention");
        let rotary = RotaryEmbedding::new(2, 10_000.0).expect("valid RoPE");
        let identity = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let key_value = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ];
        let weights = AttentionWeights::new(&identity, &key_value, &key_value, &identity);
        let hidden_states = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.25];
        let full = attention
            .forward_prefill(&hidden_states, 2, weights, &rotary)
            .expect("fresh prefill");

        let cache_config = KvCacheConfig::from_model(&config, 8).expect("valid cache config");
        let mut cache = KvCache::allocate(cache_config).expect("allocated cache");
        let mut cached = attention
            .forward_cached_token(&hidden_states[..4], 0, &mut cache, weights, &rotary)
            .expect("first cached token");
        cached.extend(
            attention
                .forward_cached_token(&hidden_states[4..], 0, &mut cache, weights, &rotary)
                .expect("second cached token"),
        );

        assert_eq!(cache.layer_sequence_length(0).expect("cache length"), 2);
        for (cached_value, full_value) in cached.into_iter().zip(full) {
            assert_close(cached_value, full_value);
        }
    }

    #[test]
    fn rejects_invalid_projection_shapes() {
        let attention =
            GroupedQueryAttention::from_config(&tiny_config()).expect("valid attention");
        assert!(attention
            .project_qkv(&[0.0; 3], 1, &[0.0; 16], &[0.0; 8], &[0.0; 8])
            .is_err());
    }
}
