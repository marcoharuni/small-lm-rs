//! Deterministic uncached greedy decoding for correctness validation.

use crate::error::{EngineError, Result};
use crate::model::NileMiniModel;

/// Generate tokens by repeatedly recomputing a complete prefill and selecting
/// the largest final-position logit.
///
/// This correctness-oriented path deliberately does not use a KV cache. The
/// cached production decode path belongs to the next milestone.
///
/// # Errors
///
/// Returns an invalid-input error for an empty prompt, zero token budget,
/// invalid EOS token, context overflow, malformed logits, or non-finite logits;
/// otherwise propagates model-execution errors.
pub fn greedy_generate_uncached(
    model: &NileMiniModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<Vec<u32>> {
    if prompt_token_ids.is_empty() {
        return Err(EngineError::invalid_input(
            "uncached greedy generation",
            "prompt must contain at least one token",
        ));
    }
    if max_new_tokens == 0 {
        return Err(EngineError::invalid_input(
            "uncached greedy generation",
            "max_new_tokens must be greater than zero",
        ));
    }
    if let Some(eos_token_id) = eos_token_id {
        if eos_token_id as usize >= model.config().vocab_size {
            return Err(EngineError::invalid_input(
                "uncached greedy generation",
                "EOS token is outside the model vocabulary",
            ));
        }
    }
    let requested_length = prompt_token_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| {
            EngineError::invalid_input(
                "uncached greedy generation",
                "requested sequence length overflows usize",
            )
        })?;
    if requested_length > model.config().context_length {
        return Err(EngineError::invalid_input(
            "uncached greedy generation",
            format!(
                "requested length {requested_length} exceeds context length {}",
                model.config().context_length
            ),
        ));
    }

    let mut complete = prompt_token_ids.to_vec();
    let mut generated = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let logits = model.forward_prefill(&complete)?;
        let expected_logits = complete
            .len()
            .checked_mul(model.config().vocab_size)
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "uncached greedy generation",
                    "logit count overflows usize",
                )
            })?;
        if logits.len() != expected_logits {
            return Err(EngineError::invalid_input(
                "uncached greedy generation",
                format!(
                    "model returned {} logits, expected {expected_logits}",
                    logits.len()
                ),
            ));
        }
        let start = expected_logits - model.config().vocab_size;
        let next_token = argmax_finite(&logits[start..])?;
        generated.push(next_token);
        complete.push(next_token);
        if eos_token_id == Some(next_token) {
            break;
        }
    }
    Ok(generated)
}

/// Generate tokens with prompt prefill followed by one-token KV-cached decode.
///
/// The selected tokens are identical to the uncached greedy path when cached
/// and fresh-sequence logits agree.
///
/// # Errors
///
/// Returns invalid-input, cache, or model-execution errors.
pub fn greedy_generate_cached(
    model: &NileMiniModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<Vec<u32>> {
    validate_request(model, prompt_token_ids, max_new_tokens, eos_token_id)?;
    let requested_length = prompt_token_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| {
            EngineError::invalid_input(
                "cached greedy generation",
                "requested sequence length overflows usize",
            )
        })?;
    let mut cache = model.allocate_kv_cache(requested_length)?;
    let prefill_logits = model.forward_prefill_with_cache(prompt_token_ids, &mut cache)?;
    let mut next_token = argmax_finite(final_logits(
        &prefill_logits,
        prompt_token_ids.len(),
        model.config().vocab_size,
    )?)?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(next_token);
    if eos_token_id == Some(next_token) {
        return Ok(generated);
    }

    for _ in 1..max_new_tokens {
        let logits = model.forward_cached_token(next_token, &mut cache)?;
        next_token = argmax_finite(&logits)?;
        generated.push(next_token);
        if eos_token_id == Some(next_token) {
            break;
        }
    }
    Ok(generated)
}

fn validate_request(
    model: &NileMiniModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<()> {
    if prompt_token_ids.is_empty() {
        return Err(EngineError::invalid_input(
            "cached greedy generation",
            "prompt must contain at least one token",
        ));
    }
    if max_new_tokens == 0 {
        return Err(EngineError::invalid_input(
            "cached greedy generation",
            "max_new_tokens must be greater than zero",
        ));
    }
    if let Some(eos_token_id) = eos_token_id {
        if eos_token_id as usize >= model.config().vocab_size {
            return Err(EngineError::invalid_input(
                "cached greedy generation",
                "EOS token is outside the model vocabulary",
            ));
        }
    }
    let requested_length = prompt_token_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| {
            EngineError::invalid_input(
                "cached greedy generation",
                "requested sequence length overflows usize",
            )
        })?;
    if requested_length > model.config().context_length {
        return Err(EngineError::invalid_input(
            "cached greedy generation",
            format!(
                "requested length {requested_length} exceeds context length {}",
                model.config().context_length
            ),
        ));
    }
    Ok(())
}

fn final_logits(logits: &[f32], sequence_length: usize, vocab_size: usize) -> Result<&[f32]> {
    let expected = sequence_length.checked_mul(vocab_size).ok_or_else(|| {
        EngineError::invalid_input("cached greedy generation", "logit count overflows usize")
    })?;
    if logits.len() != expected {
        return Err(EngineError::invalid_input(
            "cached greedy generation",
            format!(
                "model returned {} logits, expected {expected}",
                logits.len()
            ),
        ));
    }
    Ok(&logits[expected - vocab_size..])
}

fn argmax_finite(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::invalid_input(
            "uncached greedy generation",
            "final-position logits must be non-empty and finite",
        ));
    }
    let mut best_index = 0_usize;
    let mut best_value = logits[0];
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    u32::try_from(best_index).map_err(|_| {
        EngineError::invalid_input(
            "uncached greedy generation",
            "selected token index exceeds u32",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::argmax_finite;

    #[test]
    fn selects_the_largest_finite_logit() {
        assert_eq!(argmax_finite(&[-1.0, 3.0, 2.0]).expect("valid logits"), 1);
    }

    #[test]
    fn rejects_empty_or_non_finite_logits() {
        assert!(argmax_finite(&[]).is_err());
        assert!(argmax_finite(&[0.0, f32::NAN]).is_err());
    }
}
