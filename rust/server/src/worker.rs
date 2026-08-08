//! Async-to-blocking boundary for CPU inference.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use nilemini_engine::chat::{ChatMessage as EngineChatMessage, ChatRole};
use nilemini_engine::generation::GenerationRequest;
use nilemini_engine::sampler::SamplingConfig;
use nilemini_engine::service::{GenerationService, ServiceGenerationOutput};

use crate::errors::ApiError;
use crate::schema::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    CompletionChoice, CompletionRequest, CompletionResponse, Usage,
};

const DEFAULT_MAX_TOKENS: usize = 16;
const MAX_NEW_TOKENS: usize = 128;
static NEXT_RESPONSE_ID: AtomicU64 = AtomicU64::new(1);

/// Serialized CPU inference worker with request-isolated KV caches.
#[derive(Clone, Debug)]
pub struct InferenceWorker {
    service: Option<Arc<Mutex<GenerationService>>>,
    model_name: String,
}

impl InferenceWorker {
    /// Construct an unloaded worker used by lightweight router tests.
    #[must_use]
    pub fn new() -> Self {
        Self {
            service: None,
            model_name: "nilemini-8m-situ".to_owned(),
        }
    }

    /// Construct a ready worker around one reusable loaded service.
    #[must_use]
    pub fn from_service(service: GenerationService) -> Self {
        let model_name = service.model_config().model_name.clone();
        Self {
            service: Some(Arc::new(Mutex::new(service))),
            model_name,
        }
    }

    /// Return whether model artifacts are loaded.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.service.is_some()
    }

    /// Return the served model identifier.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Execute a chat-completions request.
    ///
    /// CPU work runs on Tokio's blocking pool. The reusable service is
    /// serialized, and every generation call creates an independent KV cache.
    ///
    /// # Errors
    ///
    /// Returns request, worker, tokenizer, cache, model, or sampling errors.
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
        let service = self
            .service
            .clone()
            .ok_or_else(|| ApiError::not_implemented("OpenAI-compatible chat completions"))?;

        let output = tokio::task::spawn_blocking(move || {
            let service = service.lock().map_err(|_| ApiError::Worker {
                message: "generation-service mutex was poisoned".to_owned(),
            })?;
            service
                .generate_chat(&messages, max_tokens, sampling)
                .map_err(ApiError::from)
        })
        .await
        .map_err(|source| ApiError::Worker {
            message: source.to_string(),
        })??;

        Ok(build_chat_response(&self.model_name, output))
    }

    /// Execute a text-completions request.
    ///
    /// # Errors
    ///
    /// Returns request, worker, tokenizer, cache, model, or sampling errors.
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
        let generation_request = GenerationRequest {
            prompt: request.prompt,
            max_new_tokens: max_tokens,
            sampling,
        };
        let service = self
            .service
            .clone()
            .ok_or_else(|| ApiError::not_implemented("OpenAI-compatible text completions"))?;

        let output = tokio::task::spawn_blocking(move || {
            let service = service.lock().map_err(|_| ApiError::Worker {
                message: "generation-service mutex was poisoned".to_owned(),
            })?;
            service
                .generate_text(&generation_request)
                .map_err(ApiError::from)
        })
        .await
        .map_err(|source| ApiError::Worker {
            message: source.to_string(),
        })??;

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
