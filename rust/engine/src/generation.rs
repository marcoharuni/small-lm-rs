//! KV-cached autoregressive text-generation orchestration.

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};
use crate::model::NileMiniModel;
use crate::sampler::{Sampler, SamplingConfig};
use crate::tokenizer::NileTokenizer;

/// A single text-generation request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationRequest {
    /// Prompt text to tokenize and continue.
    pub prompt: String,
    /// Maximum number of newly generated tokens.
    pub max_new_tokens: usize,
    /// Sampling controls.
    #[serde(default)]
    pub sampling: SamplingConfig,
}

impl GenerationRequest {
    /// Validate request-level controls.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for empty prompts or a zero token budget,
    /// and propagates invalid sampling settings.
    pub fn validate(&self) -> Result<()> {
        if self.prompt.is_empty() {
            return Err(EngineError::invalid_input(
                "generation",
                "prompt must not be empty",
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(EngineError::invalid_input(
                "generation",
                "max_new_tokens must be greater than zero",
            ));
        }
        self.sampling.validate()
    }
}

/// Token-level output from cached autoregressive generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TokenGenerationOutput {
    /// Newly produced token identifiers.
    pub generated_token_ids: Vec<u32>,
    /// Reason generation stopped: `eos` or `length`.
    pub finish_reason: String,
}

/// Output from a completed text-generation request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationOutput {
    /// Complete decoded text containing prompt and continuation.
    pub text: String,
    /// Newly produced token identifiers.
    pub generated_token_ids: Vec<u32>,
    /// Reason generation stopped: `eos` or `length`.
    pub finish_reason: String,
}

/// Generate token identifiers using prompt prefill and KV-cached decoding.
///
/// # Errors
///
/// Returns request, context, EOS, model, cache, or sampling errors.
pub fn generate_token_ids(
    model: &NileMiniModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
    sampling: SamplingConfig,
) -> Result<TokenGenerationOutput> {
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

    let mut sampler = Sampler::new(sampling)?;
    let mut cache = model.allocate_kv_cache(requested_length)?;
    let prefill_logits = model.forward_prefill_with_cache(prompt_token_ids, &mut cache)?;
    let mut next_token = sampler.sample(final_logits(
        &prefill_logits,
        prompt_token_ids.len(),
        model.config().vocab_size,
    )?)?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(next_token);
    if eos_token_id == Some(next_token) {
        return Ok(TokenGenerationOutput {
            generated_token_ids: generated,
            finish_reason: "eos".to_owned(),
        });
    }

    for _ in 1..max_new_tokens {
        let logits = model.forward_cached_token(next_token, &mut cache)?;
        next_token = sampler.sample(&logits)?;
        generated.push(next_token);
        if eos_token_id == Some(next_token) {
            return Ok(TokenGenerationOutput {
                generated_token_ids: generated,
                finish_reason: "eos".to_owned(),
            });
        }
    }

    Ok(TokenGenerationOutput {
        generated_token_ids: generated,
        finish_reason: "length".to_owned(),
    })
}

/// Tokenize a prompt, generate with KV caching, and decode complete text.
///
/// This stable request API stops at the requested length. Use
/// [`generate_with_eos`] when an artifact-specific EOS token is available.
///
/// # Errors
///
/// Returns request-validation, tokenizer, model, cache, or sampling errors.
pub fn generate(
    model: &NileMiniModel,
    tokenizer: &NileTokenizer,
    request: &GenerationRequest,
) -> Result<GenerationOutput> {
    generate_with_eos(model, tokenizer, request, None)
}

/// Generate text with an optional artifact-specific EOS token identifier.
///
/// # Errors
///
/// Returns request-validation, tokenizer, model, cache, EOS, or sampling
/// errors.
pub fn generate_with_eos(
    model: &NileMiniModel,
    tokenizer: &NileTokenizer,
    request: &GenerationRequest,
    eos_token_id: Option<u32>,
) -> Result<GenerationOutput> {
    request.validate()?;
    if tokenizer.vocab_size() != model.config().vocab_size {
        return Err(EngineError::invalid_input(
            "generation",
            format!(
                "tokenizer vocabulary {} does not match model vocabulary {}",
                tokenizer.vocab_size(),
                model.config().vocab_size
            ),
        ));
    }
    let prompt_token_ids = tokenizer.encode(&request.prompt, true)?;
    let token_output = generate_token_ids(
        model,
        &prompt_token_ids,
        request.max_new_tokens,
        eos_token_id,
        request.sampling.clone(),
    )?;
    let mut complete_token_ids = prompt_token_ids;
    complete_token_ids.extend_from_slice(&token_output.generated_token_ids);
    let text = tokenizer.decode(&complete_token_ids, true)?;
    Ok(GenerationOutput {
        text,
        generated_token_ids: token_output.generated_token_ids,
        finish_reason: token_output.finish_reason,
    })
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
    use super::GenerationRequest;
    use crate::sampler::SamplingConfig;

    #[test]
    fn zero_generation_budget_is_rejected() {
        let request = GenerationRequest {
            prompt: "Hello".to_owned(),
            max_new_tokens: 0,
            sampling: SamplingConfig::default(),
        };

        let error = request.validate().expect_err("zero budget must fail");
        assert!(error.to_string().contains("max_new_tokens"));
    }
}
