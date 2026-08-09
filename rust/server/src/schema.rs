//! HTTP request and response schemas.

use serde::{Deserialize, Serialize};

/// Server health response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Liveness status.
    pub status: String,
    /// Serving process name.
    pub service: String,
    /// Whether a model is loaded.
    pub ready: bool,
    /// Loaded model identifier.
    pub model: Option<String>,
}

impl HealthResponse {
    /// Build a health response.
    #[must_use]
    pub fn new(ready: bool, model: Option<String>) -> Self {
        Self {
            status: "ok".to_owned(),
            service: "small-lm-rs".to_owned(),
            ready,
            model,
        }
    }
}

/// One chat message.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    /// `system`, `user`, or `assistant`.
    pub role: String,
    /// UTF-8 message text.
    pub content: String,
}

/// Chat-completions request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation in prompt order.
    pub messages: Vec<ChatMessage>,
    /// Request SSE framing.
    #[serde(default)]
    pub stream: bool,
    /// Maximum generated tokens.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus threshold.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k candidate limit.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Sampler seed.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Text-completions request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Prompt text.
    pub prompt: String,
    /// Request SSE framing.
    #[serde(default)]
    pub stream: bool,
    /// Maximum generated tokens.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus threshold.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k candidate limit.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Sampler seed.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Token accounting.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt token count.
    pub prompt_tokens: usize,
    /// Generated token count.
    pub completion_tokens: usize,
    /// Prompt plus generated tokens.
    pub total_tokens: usize,
}

impl Usage {
    /// Build token counters.
    #[must_use]
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

/// One chat choice.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    /// Choice index.
    pub index: usize,
    /// Generated assistant message.
    pub message: ChatMessage,
    /// Stop reason.
    pub finish_reason: String,
}

/// Chat-completions response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Response identifier.
    pub id: String,
    /// API object type.
    pub object: String,
    /// Unix creation time.
    pub created: u64,
    /// Model identifier.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatCompletionChoice>,
    /// Token accounting.
    pub usage: Usage,
}

/// One text-completion choice.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Generated continuation.
    pub text: String,
    /// Choice index.
    pub index: usize,
    /// Stop reason.
    pub finish_reason: String,
}

/// Text-completions response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Response identifier.
    pub id: String,
    /// API object type.
    pub object: String,
    /// Unix creation time.
    pub created: u64,
    /// Model identifier.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<CompletionChoice>,
    /// Token accounting.
    pub usage: Usage,
}

/// Model metadata entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelObject {
    /// Stable model identifier.
    pub id: String,
    /// API object type.
    pub object: String,
    /// Project ownership label.
    pub owned_by: String,
}

/// Model-list response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelListResponse {
    /// API object type.
    pub object: String,
    /// Available model entries.
    pub data: Vec<ModelObject>,
}

/// API error envelope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error details.
    pub error: ErrorDetail,
}

impl ErrorResponse {
    /// Build an API error response.
    #[must_use]
    pub fn new(message: impl Into<String>, error_type: &'static str, code: &'static str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.to_owned(),
                code: code.to_owned(),
            },
        }
    }
}

/// Machine-readable API error.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Human-readable message.
    pub message: String,
    /// Broad error category.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Stable error code.
    pub code: String,
}

#[cfg(test)]
mod tests {
    use super::{ChatCompletionRequest, CompletionRequest};

    #[test]
    fn unknown_chat_fields_are_rejected() {
        let json = r#"{
            "model":"smalllm-8m-situ",
            "messages":[{"role":"user","content":"Hello"}],
            "unsupported":true
        }"#;
        assert!(serde_json::from_str::<ChatCompletionRequest>(json).is_err());
    }

    #[test]
    fn unknown_completion_fields_are_rejected() {
        let json = r#"{
            "model":"smalllm-8m-situ",
            "prompt":"Hello",
            "n":2
        }"#;
        assert!(serde_json::from_str::<CompletionRequest>(json).is_err());
    }
}
