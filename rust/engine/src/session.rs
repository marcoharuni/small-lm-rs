//! Stateful generation sessions for scheduler-driven decoding.

use crate::decode_backend::DecodeBackend;
use crate::error::{EngineError, Result};
use crate::kv_cache::KvCache;
use crate::sampler::{Sampler, SamplingConfig};

/// Terminal reason for one generation session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationFinishReason {
    /// The configured end-of-sequence token was sampled.
    Eos,
    /// The requested new-token budget was exhausted.
    Length,
}

impl GenerationFinishReason {
    /// Return the wire-compatible finish-reason string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::Length => "length",
        }
    }
}

/// Request-local state that can advance one decode token at a time.
///
/// A session owns its KV cache and sampler. The model remains shared and
/// immutable, which lets a scheduler interleave independent sessions without
/// mixing request state.
#[derive(Debug)]
pub struct GenerationSession {
    cache: KvCache,
    sampler: Sampler,
    eos_token_id: Option<u32>,
    max_new_tokens: usize,
    generated_token_ids: Vec<u32>,
    next_input_token: Option<u32>,
    finish_reason: Option<GenerationFinishReason>,
}

impl GenerationSession {
    /// Prefill one prompt and create a decode-ready session.
    ///
    /// Prefill also samples the first generated token from the final prompt
    /// logits. Further tokens are produced by calling [`Self::advance`].
    ///
    /// # Errors
    ///
    /// Returns input, context, model, cache, or sampling errors.
    pub fn prefill<M: DecodeBackend + ?Sized>(
        model: &M,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        sampling: SamplingConfig,
    ) -> Result<Self> {
        let requested_length =
            validate_generation_inputs(model, prompt_token_ids, max_new_tokens, eos_token_id)?;
        let mut sampler = Sampler::new(sampling)?;
        let mut cache = model.allocate_kv_cache(requested_length)?;
        let prefill_logits = model.forward_prefill_with_cache(prompt_token_ids, &mut cache)?;
        let first_token = sampler.sample(final_logits(
            &prefill_logits,
            prompt_token_ids.len(),
            model.config().vocab_size,
        )?)?;

        let finish_reason = if eos_token_id == Some(first_token) {
            Some(GenerationFinishReason::Eos)
        } else if max_new_tokens == 1 {
            Some(GenerationFinishReason::Length)
        } else {
            None
        };
        let next_input_token = finish_reason.is_none().then_some(first_token);

        Ok(Self {
            cache,
            sampler,
            eos_token_id,
            max_new_tokens,
            generated_token_ids: vec![first_token],
            next_input_token,
            finish_reason,
        })
    }

    /// Return whether no additional decode step is required.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Return the current terminal reason, when generation has finished.
    #[must_use]
    pub const fn finish_reason(&self) -> Option<GenerationFinishReason> {
        self.finish_reason
    }

    /// Return all tokens generated so far.
    #[must_use]
    pub fn generated_token_ids(&self) -> &[u32] {
        &self.generated_token_ids
    }

    /// Return the token that will be consumed by the next cached decode step.
    #[must_use]
    pub const fn next_input_token(&self) -> Option<u32> {
        self.next_input_token
    }

    /// Return the number of prompt/generated input positions currently cached.
    #[must_use]
    pub fn cached_sequence_length(&self) -> usize {
        self.cache.sequence_length()
    }

    pub(crate) fn pending_decode_token(&self) -> Result<u32> {
        if self.is_finished() {
            return Err(EngineError::invalid_input(
                "generation session",
                "cannot advance a finished session",
            ));
        }
        self.next_input_token.ok_or_else(|| {
            EngineError::invalid_input(
                "generation session",
                "running session has no pending decode token",
            )
        })
    }

    pub(crate) const fn cache_mut(&mut self) -> &mut KvCache {
        &mut self.cache
    }

    pub(crate) fn accept_logits(&mut self, logits: &[f32]) -> Result<u32> {
        if self.is_finished() {
            return Err(EngineError::invalid_input(
                "generation session",
                "cannot accept logits for a finished session",
            ));
        }

        let next_token = self.sampler.sample(logits)?;
        self.generated_token_ids.push(next_token);

        if self.eos_token_id == Some(next_token) {
            self.finish_reason = Some(GenerationFinishReason::Eos);
            self.next_input_token = None;
        } else if self.generated_token_ids.len() == self.max_new_tokens {
            self.finish_reason = Some(GenerationFinishReason::Length);
            self.next_input_token = None;
        } else {
            self.next_input_token = Some(next_token);
        }
        Ok(next_token)
    }

    /// Advance this session by exactly one cached decode step.
    ///
    /// The returned token is appended to [`Self::generated_token_ids`].
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when called after completion, or
    /// propagates model, cache, and sampling errors.
    pub fn advance<M: DecodeBackend + ?Sized>(&mut self, model: &M) -> Result<u32> {
        let input_token = self.pending_decode_token()?;
        let logits = model.forward_cached_token(input_token, &mut self.cache)?;
        self.accept_logits(&logits)
    }

    /// Consume a completed session into generated tokens and finish reason.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error if generation is still running.
    pub fn into_parts(self) -> Result<(Vec<u32>, GenerationFinishReason)> {
        let finish_reason = self.finish_reason.ok_or_else(|| {
            EngineError::invalid_input(
                "generation session",
                "cannot finish a session while generation is still running",
            )
        })?;
        Ok((self.generated_token_ids, finish_reason))
    }
}

fn validate_generation_inputs<M: DecodeBackend + ?Sized>(
    model: &M,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<usize> {
    if prompt_token_ids.is_empty() {
        return Err(EngineError::invalid_input(
            "generation",
            "the tokenized prompt must not be empty",
        ));
    }
    if max_new_tokens == 0 {
        return Err(EngineError::invalid_input(
            "generation",
            "max_new_tokens must be greater than zero",
        ));
    }
    if let Some(eos_token_id) = eos_token_id {
        if eos_token_id as usize >= model.config().vocab_size {
            return Err(EngineError::invalid_input(
                "generation",
                "EOS token is outside the model vocabulary",
            ));
        }
    }

    let requested_length = prompt_token_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| {
            EngineError::invalid_input("generation", "requested length overflows usize")
        })?;
    if requested_length > model.config().context_length {
        return Err(EngineError::invalid_input(
            "generation",
            format!(
                "requested length {requested_length} exceeds context length {}",
                model.config().context_length
            ),
        ));
    }
    Ok(requested_length)
}

fn final_logits(logits: &[f32], sequence_length: usize, vocab_size: usize) -> Result<&[f32]> {
    let expected = sequence_length.checked_mul(vocab_size).ok_or_else(|| {
        EngineError::invalid_input("generation", "prefill logit count overflows usize")
    })?;
    if logits.len() != expected {
        return Err(EngineError::invalid_input(
            "generation",
            format!(
                "model returned {} logits, expected {expected}",
                logits.len()
            ),
        ));
    }
    Ok(&logits[expected - vocab_size..])
}

#[cfg(test)]
mod tests {
    use super::{validate_generation_inputs, GenerationFinishReason};
    use crate::config::ModelConfig;
    use crate::model::SmallLMModel;

    fn tiny_model() -> SmallLMModel {
        let config = ModelConfig {
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
        };
        SmallLMModel::from_config(config).expect("valid tiny model")
    }

    #[test]
    fn generation_inputs_reserve_prompt_and_decode_capacity() {
        let model = tiny_model();
        assert_eq!(
            validate_generation_inputs(&model, &[1, 2, 3], 4, Some(7)).expect("valid generation"),
            7
        );
    }

    #[test]
    fn generation_inputs_reject_invalid_session_contracts() {
        let model = tiny_model();
        assert!(validate_generation_inputs(&model, &[], 1, None).is_err());
        assert!(validate_generation_inputs(&model, &[1], 0, None).is_err());
        assert!(validate_generation_inputs(&model, &[1], 1, Some(8)).is_err());
        assert!(validate_generation_inputs(&model, &[1, 2, 3, 4, 5], 4, None).is_err());
    }

    #[test]
    fn finish_reasons_match_existing_wire_values() {
        assert_eq!(GenerationFinishReason::Eos.as_str(), "eos");
        assert_eq!(GenerationFinishReason::Length.as_str(), "length");
    }
}
