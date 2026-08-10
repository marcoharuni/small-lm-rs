//! KV-cached autoregressive text-generation orchestration.

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};
use crate::model::SmallLMModel;
use crate::sampler::SamplingConfig;
use crate::session::GenerationSession;
use crate::tokenizer::SmallLMTokenizer;

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
/// The request is executed through [`GenerationSession`], the same
/// token-stepped state machine used by scheduler-driven generation.
///
/// # Errors
///
/// Returns request, context, EOS, model, cache, or sampling errors.
pub fn generate_token_ids(
    model: &SmallLMModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
    sampling: SamplingConfig,
) -> Result<TokenGenerationOutput> {
    let mut session = GenerationSession::prefill(
        model,
        prompt_token_ids,
        max_new_tokens,
        eos_token_id,
        sampling,
    )?;
    while !session.is_finished() {
        session.advance(model)?;
    }
    let (generated_token_ids, finish_reason) = session.into_parts()?;
    Ok(TokenGenerationOutput {
        generated_token_ids,
        finish_reason: finish_reason.as_str().to_owned(),
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
    model: &SmallLMModel,
    tokenizer: &SmallLMTokenizer,
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
    model: &SmallLMModel,
    tokenizer: &SmallLMTokenizer,
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
