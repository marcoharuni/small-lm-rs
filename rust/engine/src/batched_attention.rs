//! Batched projection with request-local cached attention.

use crate::attention::{AttentionWeights, GroupedQueryAttention};
use crate::error::{EngineError, Result};
use crate::kv_cache::KvCache;
use crate::rope::RotaryEmbedding;
use crate::softmax::softmax_in_place;

struct RotatedRow {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

/// Run one cached decode attention step for multiple independent requests.
///
/// Q/K/V and output projections are executed as multi-row matrix operations.
/// RoPE positions, KV-cache writes, and attention reductions remain isolated
/// per request because active sequences may have different cached lengths.
///
/// # Errors
///
/// Returns an invalid-input error for an empty batch, mismatched tensor/cache
/// shapes, exhausted cache capacity, or any projection, RoPE, softmax, or cache
/// failure. Cache mutations are rolled back if a later operation fails.
pub fn forward_cached_batch(
    attention: &GroupedQueryAttention,
    hidden_states: &[f32],
    layer_index: usize,
    caches: &mut [&mut KvCache],
    weights: AttentionWeights<'_>,
    rotary: &RotaryEmbedding,
) -> Result<Vec<f32>> {
    if caches.is_empty() {
        return Err(EngineError::invalid_input(
            "batched cached attention",
            "at least one request is required",
        ));
    }

    let query_heads = attention.num_query_heads();
    let key_value_heads = attention.num_key_value_heads();
    let head_dimension = attention.head_dimension();
    let hidden_size = query_heads.checked_mul(head_dimension).ok_or_else(|| {
        EngineError::invalid_input("batched cached attention", "hidden size overflows usize")
    })?;
    let expected_hidden = caches.len().checked_mul(hidden_size).ok_or_else(|| {
        EngineError::invalid_input(
            "batched cached attention",
            "hidden-state size overflows usize",
        )
    })?;
    if hidden_states.len() != expected_hidden {
        return Err(EngineError::invalid_input(
            "batched cached attention",
            format!(
                "received {} hidden-state values for {} requests, expected {expected_hidden}",
                hidden_states.len(),
                caches.len()
            ),
        ));
    }

    let mut original_lengths = Vec::with_capacity(caches.len());
    for cache in caches.iter() {
        let config = cache.config();
        if config.num_key_value_heads != key_value_heads || config.head_dimension != head_dimension {
            return Err(EngineError::invalid_input(
                "batched cached attention",
                "KV-cache head shape does not match attention configuration",
            ));
        }
        let length = cache.layer_sequence_length(layer_index)?;
        if length == 0 {
            return Err(EngineError::invalid_input(
                "batched cached attention",
                "prompt prefill must populate every request cache before decode",
            ));
        }
        if cache.remaining_capacity() == 0 {
            return Err(EngineError::invalid_input(
                "batched cached attention",
                "a request KV cache has no remaining token capacity",
            ));
        }
        original_lengths.push(length);
    }

    let projected = attention.project_qkv(
        hidden_states,
        caches.len(),
        weights.query(),
        weights.key(),
        weights.value(),
    )?;
    let query_width = hidden_size;
    let key_value_width = key_value_heads
        .checked_mul(head_dimension)
        .ok_or_else(|| {
            EngineError::invalid_input(
                "batched cached attention",
                "key/value width overflows usize",
            )
        })?;
    let mut rotated = Vec::with_capacity(caches.len());

    for (row, &position) in original_lengths.iter().enumerate() {
        let query_start = row.checked_mul(query_width).ok_or_else(|| {
            EngineError::invalid_input(
                "batched cached attention",
                "query row offset overflows usize",
            )
        })?;
        let query_end = query_start.checked_add(query_width).ok_or_else(|| {
            EngineError::invalid_input(
                "batched cached attention",
                "query row end overflows usize",
            )
        })?;
        let kv_start = row.checked_mul(key_value_width).ok_or_else(|| {
            EngineError::invalid_input(
                "batched cached attention",
                "key/value row offset overflows usize",
            )
        })?;
        let kv_end = kv_start.checked_add(key_value_width).ok_or_else(|| {
            EngineError::invalid_input(
                "batched cached attention",
                "key/value row end overflows usize",
            )
        })?;

        let mut query = projected
            .query_values()
            .get(query_start..query_end)
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "batched cached attention",
                    "projected query tensor is truncated",
                )
            })?
            .to_vec();
        let mut key = projected
            .key_values()
            .get(kv_start..kv_end)
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "batched cached attention",
                    "projected key tensor is truncated",
                )
            })?
            .to_vec();
        let value = projected
            .value_values()
            .get(kv_start..kv_end)
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "batched cached attention",
                    "projected value tensor is truncated",
                )
            })?
            .to_vec();

        rotary.apply_heads_in_place(&mut query, query_heads, position)?;
        rotary.apply_heads_in_place(&mut key, key_value_heads, position)?;
        rotated.push(RotatedRow { query, key, value });
    }

    for (cache, row) in caches.iter_mut().zip(&rotated) {
        if let Err(error) = cache.append_layer(layer_index, &row.key, &row.value, 1) {
            rollback_layer(caches, layer_index, &original_lengths)?;
            return Err(error);
        }
    }

    let attended = match attend_requests(
        query_heads,
        key_value_heads,
        head_dimension,
        layer_index,
        caches,
        &rotated,
    ) {
        Ok(attended) => attended,
        Err(error) => {
            rollback_layer(caches, layer_index, &original_lengths)?;
            return Err(error);
        }
    };

    match attention.project_output(&attended, caches.len(), weights.output()) {
        Ok(output) => Ok(output),
        Err(error) => {
            rollback_layer(caches, layer_index, &original_lengths)?;
            Err(error)
        }
    }
}

fn attend_requests(
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    layer_index: usize,
    caches: &[&mut KvCache],
    rows: &[RotatedRow],
) -> Result<Vec<f32>> {
    let hidden_size = query_heads.checked_mul(head_dimension).ok_or_else(|| {
        EngineError::invalid_input("batched cached attention", "hidden size overflows usize")
    })?;
    let output_count = rows.len().checked_mul(hidden_size).ok_or_else(|| {
        EngineError::invalid_input(
            "batched cached attention",
            "attention output size overflows usize",
        )
    })?;
    let mut attended = vec![0.0_f32; output_count];
    let group_size = query_heads / key_value_heads;
    let scale = (head_dimension as f32).sqrt().recip();

    for (request_index, (cache, row)) in caches.iter().zip(rows).enumerate() {
        let source_length = cache.layer_sequence_length(layer_index)?;
        let output_start = request_index.checked_mul(hidden_size).ok_or_else(|| {
            EngineError::invalid_input(
                "batched cached attention",
                "output row offset overflows usize",
            )
        })?;
        let output_end = output_start.checked_add(hidden_size).ok_or_else(|| {
            EngineError::invalid_input(
                "batched cached attention",
                "output row end overflows usize",
            )
        })?;
        let output_row = &mut attended[output_start..output_end];

        for query_head in 0..query_heads {
            let key_value_head = query_head / group_size;
            let query_start = query_head.checked_mul(head_dimension).ok_or_else(|| {
                EngineError::invalid_input(
                    "batched cached attention",
                    "query head offset overflows usize",
                )
            })?;
            let query_end = query_start.checked_add(head_dimension).ok_or_else(|| {
                EngineError::invalid_input(
                    "batched cached attention",
                    "query head end overflows usize",
                )
            })?;
            let query = &row.query[query_start..query_end];
            let mut probabilities = Vec::with_capacity(source_length);

            for source_position in 0..source_length {
                let key = cache.key_head(layer_index, source_position, key_value_head)?;
                let score = query
                    .iter()
                    .zip(key)
                    .fold(0.0_f32, |sum, (&query_value, &key_value)| {
                        query_value.mul_add(key_value, sum)
                    });
                probabilities.push(score * scale);
            }
            softmax_in_place(&mut probabilities)?;

            let head_start = query_head.checked_mul(head_dimension).ok_or_else(|| {
                EngineError::invalid_input(
                    "batched cached attention",
                    "output head offset overflows usize",
                )
            })?;
            let head_end = head_start.checked_add(head_dimension).ok_or_else(|| {
                EngineError::invalid_input(
                    "batched cached attention",
                    "output head end overflows usize",
                )
            })?;
            let output_head = &mut output_row[head_start..head_end];
            for (source_position, &probability) in probabilities.iter().enumerate() {
                let value = cache.value_head(layer_index, source_position, key_value_head)?;
                for (output_value, &value_element) in output_head.iter_mut().zip(value) {
                    *output_value = value_element.mul_add(probability, *output_value);
                }
            }
        }
    }

    Ok(attended)
}

fn rollback_layer(
    caches: &mut [&mut KvCache],
    layer_index: usize,
    original_lengths: &[usize],
) -> Result<()> {
    for (cache, &length) in caches.iter_mut().zip(original_lengths) {
        cache.truncate_layer(layer_index, length)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::forward_cached_batch;
    use crate::attention::{AttentionWeights, GroupedQueryAttention};
    use crate::config::ModelConfig;
    use crate::kv_cache::{KvCache, KvCacheConfig};
    use crate::rope::RotaryEmbedding;

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

    fn cache(config: &ModelConfig, key: &[f32; 2], value: &[f32; 2]) -> KvCache {
        let cache_config = KvCacheConfig::from_model(config, 4).expect("valid cache config");
        let mut cache = KvCache::allocate(cache_config).expect("allocated cache");
        cache
            .append_layer(0, key, value, 1)
            .expect("seed cache position");
        cache
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert!(
                (left - right).abs() <= 1.0e-6,
                "left={left}, right={right}"
            );
        }
    }

    #[test]
    fn batched_projection_matches_independent_cached_attention() {
        let config = tiny_config();
        let attention = GroupedQueryAttention::from_config(&config).expect("valid attention");
        let rotary = RotaryEmbedding::new(config.head_dimension, config.rope_theta).expect("RoPE");
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
        let first_hidden = [1.0, 2.0, 3.0, 4.0];
        let second_hidden = [-1.0, 0.5, 2.0, -3.0];

        let mut first_reference = cache(&config, &[0.2, -0.4], &[1.0, 2.0]);
        let mut second_reference = cache(&config, &[-0.3, 0.7], &[-1.0, 0.5]);
        let first_expected = attention
            .forward_cached_token(&first_hidden, 0, &mut first_reference, weights, &rotary)
            .expect("first independent attention");
        let second_expected = attention
            .forward_cached_token(&second_hidden, 0, &mut second_reference, weights, &rotary)
            .expect("second independent attention");

        let mut first_batch = cache(&config, &[0.2, -0.4], &[1.0, 2.0]);
        let mut second_batch = cache(&config, &[-0.3, 0.7], &[-1.0, 0.5]);
        let mut caches = [&mut first_batch, &mut second_batch];
        let hidden_states = [
            first_hidden.as_slice(),
            second_hidden.as_slice(),
        ]
        .concat();
        let actual = forward_cached_batch(
            &attention,
            &hidden_states,
            0,
            &mut caches,
            weights,
            &rotary,
        )
        .expect("batched cached attention");

        assert_close(&actual[..4], &first_expected);
        assert_close(&actual[4..], &second_expected);
        assert_eq!(first_batch.sequence_length(), 2);
        assert_eq!(second_batch.sequence_length(), 2);
    }
}
