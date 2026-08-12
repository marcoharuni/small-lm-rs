//! Reusable prompt-prefix snapshots for generation prefill.

use crate::error::{EngineError, Result};
use crate::kv_cache::KvCache;

/// Default number of reusable prompt prefixes retained by a scheduler.
pub const DEFAULT_PREFIX_CACHE_ENTRIES: usize = 16;

#[derive(Clone, Debug)]
struct PrefixEntry {
    token_ids: Vec<u32>,
    cache: KvCache,
    final_logits: Vec<f32>,
}

/// Bounded cache of prompt KV snapshots keyed by token prefixes.
#[derive(Debug)]
pub struct PrefixCache {
    max_entries: usize,
    entries: Vec<PrefixEntry>,
    hits: u64,
    misses: u64,
}

/// Snapshot returned for the longest reusable prefix of a prompt.
#[derive(Clone, Debug)]
pub struct PrefixCacheHit {
    /// Number of prompt tokens already represented by the snapshot.
    pub matched_tokens: usize,
    /// Paged KV snapshot after the matched prefix.
    pub cache: KvCache,
    /// Logits after the final matched prefix token.
    pub final_logits: Vec<f32>,
}

impl PrefixCache {
    /// Create an empty bounded prefix cache.
    ///
    /// # Errors
    /// Returns an invalid-configuration error when `max_entries` is zero.
    pub fn new(max_entries: usize) -> Result<Self> {
        if max_entries == 0 {
            return Err(EngineError::invalid_configuration(
                "prefix cache max_entries must be greater than zero",
            ));
        }
        Ok(Self {
            max_entries,
            entries: Vec::with_capacity(max_entries),
            hits: 0,
            misses: 0,
        })
    }

    /// Return the number of retained prompt prefixes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the cache contains no retained prefixes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return cumulative successful longest-prefix lookups.
    #[must_use]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Return cumulative lookups with no reusable prefix.
    #[must_use]
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Return a cloned snapshot for the longest cached prefix of `prompt`.
    pub fn lookup(&mut self, prompt: &[u32]) -> Option<PrefixCacheHit> {
        let best = self
            .entries
            .iter()
            .filter(|entry| {
                entry.token_ids.len() <= prompt.len()
                    && prompt.starts_with(entry.token_ids.as_slice())
            })
            .max_by_key(|entry| entry.token_ids.len());

        match best {
            Some(entry) => {
                self.hits = self.hits.saturating_add(1);
                Some(PrefixCacheHit {
                    matched_tokens: entry.token_ids.len(),
                    cache: entry.cache.clone(),
                    final_logits: entry.final_logits.clone(),
                })
            }
            None => {
                self.misses = self.misses.saturating_add(1);
                None
            }
        }
    }

    /// Insert or replace one complete prompt snapshot.
    pub fn insert(&mut self, token_ids: &[u32], cache: &KvCache, final_logits: &[f32]) {
        if token_ids.is_empty() {
            return;
        }
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.token_ids == token_ids)
        {
            entry.cache = cache.clone();
            entry.final_logits.clear();
            entry.final_logits.extend_from_slice(final_logits);
            return;
        }
        if self.entries.len() == self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(PrefixEntry {
            token_ids: token_ids.to_vec(),
            cache: cache.clone(),
            final_logits: final_logits.to_vec(),
        });
    }

    /// Remove all retained prefixes while preserving hit/miss counters.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(DEFAULT_PREFIX_CACHE_ENTRIES).expect("default prefix cache capacity is non-zero")
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;
    use crate::kv_cache::{KvCache, KvCacheConfig};

    fn cache_with_length(length: usize) -> KvCache {
        let config = KvCacheConfig {
            num_layers: 1,
            max_sequence_length: 8,
            num_key_value_heads: 1,
            head_dimension: 1,
        };
        let mut cache = KvCache::allocate(config).expect("cache");
        if length > 0 {
            let values = vec![1.0; length];
            cache
                .append_layer(0, &values, &values, length)
                .expect("append");
        }
        cache
    }

    #[test]
    fn chooses_longest_matching_prefix() {
        let mut cache = PrefixCache::new(4).expect("prefix cache");
        cache.insert(&[1, 2], &cache_with_length(2), &[2.0]);
        cache.insert(&[1, 2, 3], &cache_with_length(3), &[3.0]);

        let hit = cache.lookup(&[1, 2, 3, 4]).expect("prefix hit");
        assert_eq!(hit.matched_tokens, 3);
        assert_eq!(hit.cache.sequence_length(), 3);
        assert_eq!(hit.final_logits, vec![3.0]);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn bounded_cache_evicts_oldest_entry() {
        let mut cache = PrefixCache::new(1).expect("prefix cache");
        cache.insert(&[1], &cache_with_length(1), &[1.0]);
        cache.insert(&[2], &cache_with_length(1), &[2.0]);
        assert!(cache.lookup(&[1, 3]).is_none());
        assert!(cache.lookup(&[2, 3]).is_some());
    }
}
