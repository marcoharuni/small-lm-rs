//! Continuous scheduling for independent generation sessions.

use crate::error::{EngineError, Result};
use crate::model::SmallLMModel;
use crate::sampler::SamplingConfig;
use crate::session::{GenerationFinishReason, GenerationSession};

/// Stable identifier assigned to one admitted generation sequence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Return the numeric sequence identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Scheduler capacity controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    /// Maximum number of simultaneously active sequences.
    pub max_active_sequences: usize,
}

impl SchedulerConfig {
    /// Validate scheduler capacity.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when the active-sequence limit
    /// is zero.
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
        Self {
            max_active_sequences: 8,
        }
    }
}

/// One token emitted while advancing an active sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenEvent {
    /// Sequence that produced the token.
    pub sequence_id: SequenceId,
    /// Newly sampled token identifier.
    pub token_id: u32,
    /// Terminal reason when this token completed the sequence.
    pub finish_reason: Option<GenerationFinishReason>,
}

#[derive(Debug)]
struct ActiveSequence {
    id: SequenceId,
    session: GenerationSession,
    prefill_token_pending: bool,
}

/// Request scheduler that interleaves token-stepped generation sessions.
///
/// Each call to [`Self::step`] emits or advances every active sequence at most
/// once. This establishes continuous admission, independent request state, and
/// completion removal. The current implementation executes per-sequence model
/// calls serially; a batched decode kernel can replace that execution detail
/// without changing scheduler semantics.
#[derive(Debug)]
pub struct GenerationScheduler {
    config: SchedulerConfig,
    active: Vec<ActiveSequence>,
    next_sequence_id: u64,
}

impl GenerationScheduler {
    /// Create an empty scheduler.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for zero scheduler capacity.
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            active: Vec::with_capacity(config.max_active_sequences),
            next_sequence_id: 1,
        })
    }

    /// Return scheduler configuration.
    #[must_use]
    pub const fn config(&self) -> SchedulerConfig {
        self.config
    }

    /// Return the number of active sequences.
    #[must_use]
    pub fn active_sequence_count(&self) -> usize {
        self.active.len()
    }

    /// Return whether another sequence can be admitted immediately.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.active.len() < self.config.max_active_sequences
    }

    /// Admit a prompt as a new request-local generation session.
    ///
    /// Prompt prefill happens during admission. The token sampled from the
    /// final prompt logits is retained and emitted on the next scheduler step.
    /// A future prefill scheduler can move that work into its own batching
    /// policy without changing sequence identifiers or decode semantics.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when scheduler capacity is exhausted, or
    /// propagates model, context, cache, and sampling failures from prefill.
    pub fn admit(
        &mut self,
        model: &SmallLMModel,
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

        let session = GenerationSession::prefill(
            model,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
            sampling,
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

    /// Emit or advance each active sequence by at most one token.
    ///
    /// A newly admitted sequence first emits the token sampled during prompt
    /// prefill. Later steps perform one cached decode operation. Finished
    /// sequences are removed after their terminal token has been reported.
    ///
    /// # Errors
    ///
    /// Propagates model, cache, or sampling failures from any active session.
    pub fn step(&mut self, model: &SmallLMModel) -> Result<Vec<TokenEvent>> {
        let mut events = Vec::with_capacity(self.active.len());

        for sequence in &mut self.active {
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
                events.push(TokenEvent {
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

            let token_id = sequence.session.advance(model)?;
            events.push(TokenEvent {
                sequence_id: sequence.id,
                token_id,
                finish_reason: sequence.session.finish_reason(),
            });
        }

        self.active
            .retain(|sequence| !sequence.session.is_finished());
        Ok(events)
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
        let error = GenerationScheduler::new(SchedulerConfig {
            max_active_sequences: 0,
        })
        .expect_err("zero capacity must fail");
        assert!(error.to_string().contains("max_active_sequences"));
    }

    #[test]
    fn fresh_scheduler_reports_capacity() {
        let scheduler = GenerationScheduler::new(SchedulerConfig {
            max_active_sequences: 4,
        })
        .expect("valid scheduler");
        assert_eq!(scheduler.active_sequence_count(), 0);
        assert!(scheduler.has_capacity());
        assert_eq!(scheduler.config().max_active_sequences, 4);
    }

    #[test]
    fn sequence_id_exposes_stable_numeric_value() {
        assert_eq!(SequenceId(17).get(), 17);
    }
}
