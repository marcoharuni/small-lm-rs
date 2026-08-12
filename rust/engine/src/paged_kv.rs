//! Fixed-size paged storage for autoregressive key/value tensors.

use crate::error::{EngineError, Result};

/// Default number of token rows stored in one KV page.
pub const DEFAULT_KV_PAGE_TOKENS: usize = 16;

#[derive(Clone, Debug)]
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvPage {
    fn new(page_tokens: usize, width: usize) -> Result<Self> {
        let elements = page_tokens.checked_mul(width).ok_or_else(|| {
            EngineError::invalid_configuration("KV page element count overflows usize")
        })?;
        Ok(Self {
            keys: vec![0.0; elements],
            values: vec![0.0; elements],
        })
    }
}

/// Lazily allocated fixed-size pages for one transformer's KV layer.
#[derive(Clone, Debug)]
pub(crate) struct PagedLayerKvCache {
    width: usize,
    max_sequence_length: usize,
    page_tokens: usize,
    pages: Vec<KvPage>,
    sequence_length: usize,
}

impl PagedLayerKvCache {
    pub(crate) fn new(
        width: usize,
        max_sequence_length: usize,
        page_tokens: usize,
    ) -> Result<Self> {
        if width == 0 || max_sequence_length == 0 || page_tokens == 0 {
            return Err(EngineError::invalid_configuration(
                "paged KV dimensions must be greater than zero",
            ));
        }
        Ok(Self {
            width,
            max_sequence_length,
            page_tokens,
            pages: Vec::new(),
            sequence_length: 0,
        })
    }

    #[must_use]
    pub(crate) const fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    #[must_use]
    pub(crate) fn allocated_pages(&self) -> usize {
        self.pages.len()
    }

    #[must_use]
    pub(crate) fn allocated_token_capacity(&self) -> usize {
        self.pages
            .len()
            .saturating_mul(self.page_tokens)
            .min(self.max_sequence_length)
    }

    pub(crate) fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        token_count: usize,
    ) -> Result<()> {
        let expected = token_count.checked_mul(self.width).ok_or_else(|| {
            EngineError::invalid_input("paged KV cache", "append shape overflows usize")
        })?;
        if keys.len() != expected || values.len() != expected {
            return Err(EngineError::invalid_input(
                "paged KV cache",
                format!(
                    "expected {expected} key/value elements for {token_count} tokens, received {} keys and {} values",
                    keys.len(),
                    values.len()
                ),
            ));
        }
        if keys.iter().chain(values).any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_input(
                "paged KV cache",
                "all appended key/value values must be finite",
            ));
        }

        let new_length = self
            .sequence_length
            .checked_add(token_count)
            .ok_or_else(|| {
                EngineError::invalid_input("paged KV cache", "sequence length overflows usize")
            })?;
        if new_length > self.max_sequence_length {
            return Err(EngineError::invalid_input(
                "paged KV cache",
                format!(
                    "append would grow cache to {new_length} tokens, exceeding capacity {}",
                    self.max_sequence_length
                ),
            ));
        }

        let required_pages = new_length.div_ceil(self.page_tokens);
        while self.pages.len() < required_pages {
            self.pages.push(KvPage::new(self.page_tokens, self.width)?);
        }

        for token_offset in 0..token_count {
            let logical_position = self.sequence_length + token_offset;
            let page_index = logical_position / self.page_tokens;
            let position_in_page = logical_position % self.page_tokens;
            let destination_start = position_in_page.checked_mul(self.width).ok_or_else(|| {
                EngineError::invalid_input("paged KV cache", "page offset overflows usize")
            })?;
            let destination_end = destination_start + self.width;
            let source_start = token_offset.checked_mul(self.width).ok_or_else(|| {
                EngineError::invalid_input("paged KV cache", "source offset overflows usize")
            })?;
            let source_end = source_start + self.width;
            self.pages[page_index].keys[destination_start..destination_end]
                .copy_from_slice(&keys[source_start..source_end]);
            self.pages[page_index].values[destination_start..destination_end]
                .copy_from_slice(&values[source_start..source_end]);
        }
        self.sequence_length = new_length;
        Ok(())
    }

    pub(crate) fn key_row(&self, token_position: usize) -> Result<&[f32]> {
        self.row(token_position, true)
    }

    pub(crate) fn value_row(&self, token_position: usize) -> Result<&[f32]> {
        self.row(token_position, false)
    }

    pub(crate) fn materialize_keys(&self) -> Result<Vec<f32>> {
        self.materialize(true)
    }

    pub(crate) fn materialize_values(&self) -> Result<Vec<f32>> {
        self.materialize(false)
    }

    pub(crate) fn truncate(&mut self, sequence_length: usize) -> Result<()> {
        if sequence_length > self.sequence_length {
            return Err(EngineError::invalid_input(
                "paged KV cache",
                format!(
                    "cannot extend cache from {} to {sequence_length} while truncating",
                    self.sequence_length
                ),
            ));
        }
        self.sequence_length = sequence_length;
        let retained_pages = sequence_length.div_ceil(self.page_tokens);
        self.pages.truncate(retained_pages);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.sequence_length = 0;
        self.pages.clear();
    }

    fn materialize(&self, keys: bool) -> Result<Vec<f32>> {
        let element_count = self
            .sequence_length
            .checked_mul(self.width)
            .ok_or_else(|| {
                EngineError::invalid_input("paged KV cache", "materialized size overflows usize")
            })?;
        let mut values = Vec::with_capacity(element_count);
        for token_position in 0..self.sequence_length {
            values.extend_from_slice(self.row(token_position, keys)?);
        }
        Ok(values)
    }

    fn row(&self, token_position: usize, keys: bool) -> Result<&[f32]> {
        if token_position >= self.sequence_length {
            return Err(EngineError::invalid_input(
                "paged KV cache",
                format!(
                    "token position {token_position} is outside 0..{}",
                    self.sequence_length
                ),
            ));
        }
        let page_index = token_position / self.page_tokens;
        let position_in_page = token_position % self.page_tokens;
        let start = position_in_page.checked_mul(self.width).ok_or_else(|| {
            EngineError::invalid_input("paged KV cache", "row offset overflows usize")
        })?;
        let end = start + self.width;
        let page = self.pages.get(page_index).ok_or_else(|| {
            EngineError::invalid_input("paged KV cache", "logical page is not allocated")
        })?;
        let storage = if keys { &page.keys } else { &page.values };
        storage.get(start..end).ok_or_else(|| {
            EngineError::invalid_input("paged KV cache", "paged row storage is truncated")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::PagedLayerKvCache;

    #[test]
    fn allocates_pages_lazily_across_boundaries() {
        let mut cache = PagedLayerKvCache::new(2, 10, 4).expect("valid cache");
        assert_eq!(cache.allocated_pages(), 0);
        assert_eq!(cache.allocated_token_capacity(), 0);

        cache
            .append(&[1.0, 2.0], &[3.0, 4.0], 1)
            .expect("first token");
        assert_eq!(cache.allocated_pages(), 1);
        assert_eq!(cache.allocated_token_capacity(), 4);

        cache
            .append(
                &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                4,
            )
            .expect("cross page boundary");
        assert_eq!(cache.sequence_length(), 5);
        assert_eq!(cache.allocated_pages(), 2);
        assert_eq!(cache.allocated_token_capacity(), 8);
        assert_eq!(cache.key_row(4).expect("row four"), &[11.0, 12.0]);
        assert_eq!(cache.value_row(4).expect("row four"), &[19.0, 20.0]);
        assert_eq!(
            cache.materialize_keys().expect("materialized keys"),
            vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        assert_eq!(
            cache.materialize_values().expect("materialized values"),
            vec![3.0, 4.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        );
    }

    #[test]
    fn truncate_releases_trailing_pages() {
        let mut cache = PagedLayerKvCache::new(1, 16, 4).expect("valid cache");
        cache
            .append(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0, 8.0, 9.0, 10.0], 5)
            .expect("append");
        assert_eq!(cache.allocated_pages(), 2);

        cache.truncate(3).expect("truncate");
        assert_eq!(cache.sequence_length(), 3);
        assert_eq!(cache.allocated_pages(), 1);
        assert_eq!(cache.key_row(2).expect("retained row"), &[3.0]);
    }

    #[test]
    fn clear_releases_all_pages() {
        let mut cache = PagedLayerKvCache::new(1, 8, 4).expect("valid cache");
        cache.append(&[1.0], &[2.0], 1).expect("append");
        cache.clear();
        assert_eq!(cache.sequence_length(), 0);
        assert_eq!(cache.allocated_pages(), 0);
    }
}
