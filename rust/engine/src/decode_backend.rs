//! Shared decode contract for single-request and batched model executors.

use crate::batched_model::BatchedDecodeModel;
use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::kv_cache::KvCache;
use crate::model::SmallLMModel;

/// Model operations required by generation sessions and the scheduler.
pub trait DecodeBackend {
    /// Return the architecture configuration used by this backend.
    fn config(&self) -> &ModelConfig;

    /// Allocate one request-local KV cache.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for an unsupported capacity.
    fn allocate_kv_cache(&self, max_sequence_length: usize) -> Result<KvCache>;

    /// Prefill one prompt and populate its request-local cache.
    ///
    /// # Errors
    ///
    /// Propagates model, cache, weight, and projection errors.
    fn forward_prefill_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>>;

    /// Decode one token against one request-local cache.
    ///
    /// # Errors
    ///
    /// Propagates model, cache, weight, and projection errors.
    fn forward_cached_token(&self, token_id: u32, cache: &mut KvCache) -> Result<Vec<f32>>;

    /// Decode one token for each request cache in the batch.
    ///
    /// Returned logits are row-major `[batch, vocab_size]` in request order.
    /// Implementations must restore all cache lengths if any request fails.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for mismatched batch dimensions or
    /// propagates model, cache, weight, and projection errors.
    fn forward_cached_batch(
        &self,
        token_ids: &[u32],
        caches: &mut [&mut KvCache],
    ) -> Result<Vec<f32>>;
}

impl DecodeBackend for SmallLMModel {
    fn config(&self) -> &ModelConfig {
        SmallLMModel::config(self)
    }

    fn allocate_kv_cache(&self, max_sequence_length: usize) -> Result<KvCache> {
        SmallLMModel::allocate_kv_cache(self, max_sequence_length)
    }

    fn forward_prefill_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        SmallLMModel::forward_prefill_with_cache(self, token_ids, cache)
    }

    fn forward_cached_token(&self, token_id: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        SmallLMModel::forward_cached_token(self, token_id, cache)
    }

    fn forward_cached_batch(
        &self,
        token_ids: &[u32],
        caches: &mut [&mut KvCache],
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() || token_ids.len() != caches.len() {
            return Err(EngineError::invalid_input(
                "decode backend batch",
                "token and cache batches must have the same non-zero length",
            ));
        }

        let original_lengths = caches
            .iter()
            .map(|cache| cache.sequence_length())
            .collect::<Vec<_>>();
        let mut logits = Vec::with_capacity(
            token_ids
                .len()
                .checked_mul(self.config().vocab_size)
                .ok_or_else(|| {
                    EngineError::invalid_input(
                        "decode backend batch",
                        "logit count overflows usize",
                    )
                })?,
        );

        for (&token_id, cache) in token_ids.iter().zip(caches.iter_mut()) {
            match SmallLMModel::forward_cached_token(self, token_id, cache) {
                Ok(row) => logits.extend(row),
                Err(error) => {
                    rollback_caches(caches, &original_lengths)?;
                    return Err(error);
                }
            }
        }
        Ok(logits)
    }
}

impl DecodeBackend for BatchedDecodeModel {
    fn config(&self) -> &ModelConfig {
        BatchedDecodeModel::config(self)
    }

    fn allocate_kv_cache(&self, max_sequence_length: usize) -> Result<KvCache> {
        BatchedDecodeModel::allocate_kv_cache(self, max_sequence_length)
    }

    fn forward_prefill_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        BatchedDecodeModel::forward_prefill_with_cache(self, token_ids, cache)
    }

    fn forward_cached_token(&self, token_id: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        BatchedDecodeModel::forward_cached_token(self, token_id, cache)
    }

    fn forward_cached_batch(
        &self,
        token_ids: &[u32],
        caches: &mut [&mut KvCache],
    ) -> Result<Vec<f32>> {
        BatchedDecodeModel::forward_cached_batch(self, token_ids, caches)
    }
}

fn rollback_caches(caches: &mut [&mut KvCache], lengths: &[usize]) -> Result<()> {
    for (cache, &length) in caches.iter_mut().zip(lengths) {
        cache.truncate_all(length)?;
    }
    Ok(())
}
