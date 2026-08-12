//! Continuous scheduling for independent generation sessions.

use crate::decode_backend::DecodeBackend;
use crate::error::{EngineError, Result};
use crate::prefix_cache::PrefixCache;
use crate::sampler::SamplingConfig;
use crate::session::{GenerationFinishReason, GenerationSession};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(u64);

impl SequenceId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    pub max_active_sequences: usize,
}

impl SchedulerConfig {
    pub fn validate(self) -> Result<()> {
        if self.max_active_sequences == 0 {
            return Err(EngineError::invalid_configuration(
                "scheduler max_active_sequences must be greater than zero",
            ));
        }
        Ok(())
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { max_active_sequences: 8 }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenEvent {
    pub sequence_id: SequenceId,
    pub token_id: u32,
    pub finish_reason: Option<GenerationFinishReason>,
}

#[derive(Debug)]
struct ActiveSequence {
    id: SequenceId,
    session: GenerationSession,
    prefill_token_pending: bool,
}

#[derive(Debug)]
pub struct GenerationScheduler {
    config: SchedulerConfig,
    active: Vec<ActiveSequence>,
    next_sequence_id: u64,
    prefix_cache: PrefixCache,
}

impl GenerationScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            active: Vec::with_capacity(config.max_active_sequences),
            next_sequence_id: 1,
            prefix_cache: PrefixCache::default(),
        })
    }

    #[must_use]
    pub const fn config(&self) -> SchedulerConfig {
        self.config
    }

    #[must_use]
    pub fn active_sequence_count(&self) -> usize {
        self.active.len()
    }

    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.active.len() < self.config.max_active_sequences
    }

    #[must_use]
    pub const fn prefix_cache(&self) -> &PrefixCache {
        &self.prefix_cache
    }

    pub fn clear_prefix_cache(&mut self) {
        self.prefix_cache.clear();
    }

    pub fn admit<M: DecodeBackend + ?Sized>(
        &mut self,
        model: &M,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        sampling: SamplingConfig,
    ) -> Result<SequenceId> {
        if !self.has_capacity() {
            return Err(EngineError::invalid_input(
                "generation scheduler",
                "active sequence capacity is exhausted",
            ));
        }

        let session = GenerationSession::prefill_with_prefix_cache(
            model,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
            sampling,
            &mut self.prefix_cache,
        )?;
        let sequence_id = SequenceId(self.next_sequence_id);
        self.next_sequence_id = self.next_sequence_id.checked_add(1).ok_or_else(|| {
            EngineError::invalid_input(
                "generation scheduler",
                "sequence identifier space is exhausted",
            )
        })?;
        self.active.push(ActiveSequence {
            id: sequence_id,
            session,
            prefill_token_pending: true,
        });
        Ok(sequence_id)
    }

    pub fn step<M: DecodeBackend + ?Sized>(&mut self, model: &M) -> Result<Vec<TokenEvent>> {
        let mut event_slots = vec![None; self.active.len()];
        let mut decode_ready = vec![false; self.active.len()];

        for (index, sequence) in self.active.iter_mut().enumerate() {
            if sequence.prefill_token_pending {
                let token_id = *sequence
                    .session
                    .generated_token_ids()
                    .last()
                    .ok_or_else(|| {
                        EngineError::invalid_input(
                            "generation scheduler",
                            "prefilled session has no generated token",
                        )
                    })?;
                sequence.prefill_token_pending = false;
                event_slots[index] = Some(TokenEvent {
                    sequence_id: sequence.id,
                    token_id,
                    finish_reason: sequence.session.finish_reason(),
                });
                continue;
            }

            if sequence.session.is_finished() {
                return Err(EngineError::invalid_input(
                    "generation scheduler",
                    "finished session remained active after its terminal event",
                ));
            }
            decode_ready[index] = true;
        }

        let token_ids = self
            .active
            .iter()
            .zip(&decode_ready)
            .filter_map(|(sequence, &ready)| ready.then_some(&sequence.session))
            .map(GenerationSession::pending_decode_token)
            .collect::<Result<Vec<_>>>()?;

        if !token_ids.is_empty() {
            let logits = {
                let mut caches = self
                    .active
                    .iter_mut()
                    .zip(&decode_ready)
                    .filter_map(|(sequence, &ready)| ready.then(|| sequence.session.cache_mut()))
                    .collect::<Vec<_>>();
                model.forward_cached_batch(&token_ids, &mut caches)?
            };

            let vocab_size = model.config().vocab_size;
            let expected_logits = token_ids.len().checked_mul(vocab_size).ok_or_else(|| {
                EngineError::invalid_input(
                    "generation scheduler",
                    "batched logit count overflows usize",
                )
            })?;
            if logits.len() != expected_logits {
                return Err(EngineError::invalid_input(
                    "generation scheduler",
                    format!(
                        "backend returned {} logits for {} requests, expected {expected_logits}",
                        logits.len(), token_ids.len()
                    ),
                ));
            }

            let mut row_index = 0_usize;
            for (index, (sequence, &ready)) in self.active.iter_mut().zip(&decode_ready).enumerate() {
                if !ready {
                    continue;
                }
                let row_start = row_index.checked_mul(vocab_size).ok_or_else(|| {
                    EngineError::invalid_input("generation scheduler", "logit row offset overflows usize")
                })?;
                let row_end = row_start.checked_add(vocab_size).ok_or_else(|| {
                    EngineError::invalid_input("generation scheduler", "logit row end overflows usize")
                })?;
                let row = &logits[row_start..row_end];
                let token_id = sequence.session.accept_logits(row)?;
                event_slots[index] = Some(TokenEvent {
                    sequence_id: sequence.id,
                    token_id,
                    finish_reason: sequence.session.finish_reason(),
                });
                row_index += 1;
            }
        }

        self.active.retain(|sequence| !sequence.session.is_finished());
        Ok(event_slots.into_iter().flatten().collect())
    }
}

impl Default for GenerationScheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default()).expect("default scheduler configuration is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationScheduler, SchedulerConfig, SequenceId};

    #[test]
    fn scheduler_rejects_zero_capacity() {
        let error = GenerationScheduler::new(SchedulerConfig { max_active_sequences: 0 })
            .expect_err("zero capacity must fail");
        assert!(error.to_string().contains("max_active_sequences"));
    }

    #[test]
    fn fresh_scheduler_reports_capacity_and_empty_prefix_cache() {
        let scheduler = GenerationScheduler::new(SchedulerConfig { max_active_sequences: 4 })
            .expect("valid scheduler");
        assert_eq!(scheduler.active_sequence_count(), 0);
        assert!(scheduler.has_capacity());
        assert_eq!(scheduler.config().max_active_sequences, 4);
        assert!(scheduler.prefix_cache().is_empty());
    }

    #[test]
    fn sequence_id_exposes_stable_numeric_value() {
        assert_eq!(SequenceId(17).get(), 17);
    }
}
