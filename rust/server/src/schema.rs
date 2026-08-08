//! HTTP request and response schemas.

use serde::{Deserialize, Serialize};

/// Response returned by the health endpoint.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Process liveness status.
    pub status: String,
    /// Name of the serving process.
    pub service: String,
    /// Whether model artifacts are loaded for inference.
    pub ready: bool,
    /// Loaded model identifier when ready.
    pub model: Option<String>,
}

impl HealthResponse {
    /// Construct a server health response.
    #[must_use]
    pub fn new(ready: bool, model: Option<String>) -> Self {
        Self {
            status: "ok".to_owned(),
            service: "nilemini-server".to_owned(),
            ready,
            model,
        }
    }
}

/// One role-tagged chat message.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    /// Message author role: `system`, `user`, or `assistant`.
    pub role: String,
    /// Plain UTF-8 message content.
    pub content: String,
}

/// Accepted OpenAI-compatible chat-completions subset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    /// Requested model identifier.
    pub model: String,
    /// Conversation messages in prompt order.
    pub messages: Vec<ChatMessage>,
    /// Whether to return Server-Sent Events.
    #[serde(default)]
    pub stream: bool,
    /// Optional cap on newly generated tokens.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Optional sampling temperature; zero selects greedy decoding.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional nucleus-sampling threshold.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Optional top-k candidate cap; zero disables it.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Optional deterministic sampler seed.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Accepted OpenAI-compatible text-completions subset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    /// Requested model identifier.
    pub model: String,
    /// Plain-text prompt to continue.
    pub prompt: String,
    /// Whether to return Server-Sent Events.
    #[serde(default)]
    pub stream: bool,
    /// Optional cap on newly generated tokens.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Optional sampling temperature; zero selects greedy decoding.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional nucleus-sampling threshold.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Optional top-k candidate cap; zero disables it.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Optional deterministic sampler seed.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Token accounting attached to completed responses.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub completion_tokens: usize,
    /// Sum of prompt and generated tokens.
    pub total_tokens: usize,
}

impl Usage {
    /// Construct usage counters.
    #[must_use]
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

/// One choice in a chat-completions response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    /// Zero-based choice index.
    pub index: usize,
    /// Assistant message produced by the engine.
    pub message: ChatMessage,
    /// Stop condition: `stop` or `length`.
    pub finish_reason: String,
}

/// Non-streaming chat-completions response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique response identifier.
    pub id: String,
    /// API object type: `chat.completion`.
    pub object: String,
    /// Unix timestamp at response creation.
    pub created: u64,
    /// Model identifier used for generation.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatCompletionChoice>,
    /// Token accounting.
    pub usage: Usage,
}

/// One choice in a text-completions response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Generated continuation text.
    pub text: String,
    /// Zero-based choice index.
    pub index: usize,
    /// Stop condition: `stop` or `length`.
    pub finish_reason: String,
}

/// Non-streaming text-completions response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Unique response identifier.
    pub id: String,
    /// API object type: `text_completion`.
    pub object: String,
    /// Unix timestamp at response creation.
    pub created: u64,
    /// Model identifier used for generation.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<CompletionChoice>,
    /// Token accounting.
    pub usage: Usage,
}

/// One model entry returned by `/v1/models`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelObject {
    /// Stable model identifier.
    pub id: String,
    /// API object type: `model`.
    pub object: String,
    /// Owning organization label.
    pub owned_by: String,
}

/// OpenAI-shaped model-list response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelListResponse {
    /// API object type: `list`.
    pub object: String,
    /// Available models.
    pub data: Vec<ModelObject>,
}

/// OpenAI-shaped envelope for an API failure.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Structured error details.
    pub error: ErrorDetail,
}

impl ErrorResponse {
    /// Construct a structured error envelope.
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

/// Machine- and human-readable API error information.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Human-readable explanation.
    pub message: String,
    /// Broad OpenAI-style error category.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Stable machine-readable error code.
    pub code: String,
}

#[cfg(test)]
mod tests {
    use super::{ChatCompletionRequest, CompletionRequest};

    #[test]
    fn unknown_chat_fields_are_rejected() {
        let json = r#"{
            "model":"nilemini-8m-situ",
            "messages":[{"role":"user","content":"Hello"}],
            "unsupported":true
        }"#;
        assert!(serde_json::from_str::<ChatCompletionRequest>(json).is_err());
    }

    #[test]
    fn unknown_completion_fields_are_rejected() {
        let json = r#"{
            "model":"nilemini-8m-situ",
            "prompt":"Hello",
            "n":2
        }"#;
        assert!(serde_json::from_str::<CompletionRequest>(json).is_err());
    }
}
