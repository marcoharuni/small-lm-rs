//! Async HTTP boundary for the dedicated CPU batching runtime.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use smalllm_engine::chat::{
    format_chat_prompt, ChatMessage as EngineChatMessage, ChatRole,
};
use smalllm_engine::sampler::SamplingConfig;
use smalllm_engine::service::ServiceGenerationOutput;
use smalllm_engine::EngineError;

use crate::batch_worker::BatchRuntime;
use crate::errors::ApiError;
use crate::schema::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    CompletionChoice, CompletionRequest, CompletionResponse, Usage,
};

const DEFAULT_MAX_TOKENS: usize = 16;
const MAX_NEW_TOKENS: usize = 128;
static NEXT_RESPONSE_ID: AtomicU64 = AtomicU64::new(1);

/// Async request handle backed by one long-lived CPU batching thread.
#[derive(Clone, Debug)]
pub struct InferenceWorker {
    runtime: Option<BatchRuntime>,
    model_name: String,
}

impl InferenceWorker {
    /// Construct an unloaded worker used by lightweight router tests.
    #[must_use]
    pub fn new() -> Self {
        Self {
            runtime: None,
            model_name: "small-lm-8m".to_owned(),
        }
    }

    /// Load artifacts and start a ready continuous-batching worker.
    ///
    /// # Errors
    ///
    /// Returns configuration, tokenizer, model-weight, or scheduler errors.
    pub fn from_artifact_dir(
        root: impl AsRef<Path>,
        max_active_sequences: usize,
    ) -> Result<Self, EngineError> {
        let runtime = BatchRuntime::from_artifact_dir(root, max_active_sequences)?;
        let model_name = runtime.model_name().to_owned();
        Ok(Self {
            runtime: Some(runtime),
            model_name,
        })
    }

    /// Return whether model artifacts and the batch worker are loaded.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.runtime.is_some()
    }

    /// Return the served model identifier.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Execute a chat-completions request through the shared batch scheduler.
    ///
    /// # Errors
    ///
    /// Returns request, tokenizer, scheduler, model, cache, or sampling errors.
    pub async fn complete_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ApiError> {
        self.validate_model(&request.model)?;
        let max_tokens = validate_max_tokens(request.max_tokens)?;
        let sampling = sampling_config(
            request.temperature,
            request.top_p,
            request.top_k,
            request.seed,
        )?;
        let messages = convert_messages(request.messages)?;
        let runtime = self
            .runtime
            .clone()
            .ok_or_else(|| ApiError::not_implemented("OpenAI-compatible chat completions"))?;
        let prompt = format_chat_prompt(&messages).map_err(ApiError::from)?;
        let prompt_token_ids = runtime.encode(&prompt, false).map_err(ApiError::from)?;
        let prompt_token_count = prompt_token_ids.len();
        let generated = runtime
            .generate(prompt_token_ids, max_tokens, sampling)
            .await?;
        let completion = runtime
            .decode(&generated.generated_token_ids)
            .map_err(ApiError::from)?;
        let output = ServiceGenerationOutput {
            prompt_token_count,
            completion,
            generated_token_ids: generated.generated_token_ids,
            finish_reason: generated.finish_reason,
        };

        Ok(build_chat_response(&self.model_name, output))
    }

    /// Execute a text-completions request through the shared batch scheduler.
    ///
    /// # Errors
    ///
    /// Returns request, tokenizer, scheduler, model, cache, or sampling errors.
    pub async fn complete_text(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ApiError> {
        self.validate_model(&request.model)?;
        let max_tokens = validate_max_tokens(request.max_tokens)?;
        let sampling = sampling_config(
            request.temperature,
            request.top_p,
            request.top_k,
            request.seed,
        )?;
        let runtime = self
            .runtime
            .clone()
            .ok_or_else(|| ApiError::not_implemented("OpenAI-compatible text completions"))?;
        let prompt_token_ids = runtime
            .encode(&request.prompt, true)
            .map_err(ApiError::from)?;
        let prompt_token_count = prompt_token_ids.len();
        let generated = runtime
            .generate(prompt_token_ids, max_tokens, sampling)
            .await?;
        let completion = runtime
            .decode(&generated.generated_token_ids)
            .map_err(ApiError::from)?;
        let output = ServiceGenerationOutput {
            prompt_token_count,
            completion,
            generated_token_ids: generated.generated_token_ids,
            finish_reason: generated.finish_reason,
        };

        Ok(build_completion_response(&self.model_name, output))
    }

    fn validate_model(&self, requested: &str) -> Result<(), ApiError> {
        if requested != self.model_name {
            return Err(ApiError::bad_request(format!(
                "model {requested:?} is unavailable; expected {:?}",
                self.model_name
            )));
        }
        Ok(())
    }
}

impl Default for InferenceWorker {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_max_tokens(value: Option<usize>) -> Result<usize, ApiError> {
    let value = value.unwrap_or(DEFAULT_MAX_TOKENS);
    if value == 0 || value > MAX_NEW_TOKENS {
        return Err(ApiError::bad_request(format!(
            "max_tokens must be in the range 1..={MAX_NEW_TOKENS}"
        )));
    }
    Ok(value)
}

fn sampling_config(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> Result<SamplingConfig, ApiError> {
    let config = SamplingConfig {
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        top_k: top_k.unwrap_or(0),
        seed: seed.unwrap_or(0),
    };
    config.validate().map_err(ApiError::from)?;
    Ok(config)
}

fn convert_messages(messages: Vec<ChatMessage>) -> Result<Vec<EngineChatMessage>, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            let role = match message.role.as_str() {
                "system" => ChatRole::System,
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                other => {
                    return Err(ApiError::bad_request(format!(
                        "message {index} has unsupported role {other:?}"
                    )))
                }
            };
            Ok(EngineChatMessage {
                role,
                content: message.content,
            })
        })
        .collect()
}

fn build_chat_response(
    model_name: &str,
    output: ServiceGenerationOutput,
) -> ChatCompletionResponse {
    let created = unix_timestamp();
    let completion_tokens = output.generated_token_ids.len();
    ChatCompletionResponse {
        id: response_id("chatcmpl", created),
        object: "chat.completion".to_owned(),
        created,
        model: model_name.to_owned(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_owned(),
                content: output.completion,
            },
            finish_reason: normalize_finish_reason(&output.finish_reason),
        }],
        usage: Usage::new(output.prompt_token_count, completion_tokens),
    }
}

fn build_completion_response(
    model_name: &str,
    output: ServiceGenerationOutput,
) -> CompletionResponse {
    let created = unix_timestamp();
    let completion_tokens = output.generated_token_ids.len();
    CompletionResponse {
        id: response_id("cmpl", created),
        object: "text_completion".to_owned(),
        created,
        model: model_name.to_owned(),
        choices: vec![CompletionChoice {
            text: output.completion,
            index: 0,
            finish_reason: normalize_finish_reason(&output.finish_reason),
        }],
        usage: Usage::new(output.prompt_token_count, completion_tokens),
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn response_id(prefix: &str, created: u64) -> String {
    let sequence = NEXT_RESPONSE_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{created:x}-{sequence:x}")
}

fn normalize_finish_reason(reason: &str) -> String {
    if reason == "eos" {
        "stop".to_owned()
    } else {
        reason.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{sampling_config, validate_max_tokens};

    #[test]
    fn request_limits_are_validated_before_inference() {
        assert!(validate_max_tokens(Some(0)).is_err());
        assert!(validate_max_tokens(Some(129)).is_err());
        assert_eq!(validate_max_tokens(None).expect("default"), 16);
        assert!(sampling_config(Some(-1.0), None, None, None).is_err());
        assert!(sampling_config(None, Some(0.0), None, None).is_err());
    }
}
