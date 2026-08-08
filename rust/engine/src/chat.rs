//! Versioned chat-message validation and prompt formatting.

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

const RESERVED_MARKERS: [&str; 6] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
];

/// Supported authors in a NileMini chat history.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// Instruction or behavior-setting message.
    System,
    /// Human message.
    User,
    /// Model message retained as conversation history.
    Assistant,
}

impl ChatRole {
    const fn marker(self) -> &'static str {
        match self {
            Self::System => "<|system|>",
            Self::User => "<|user|>",
            Self::Assistant => "<|assistant|>",
        }
    }
}

/// One validated role-tagged chat message.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    /// Message author.
    pub role: ChatRole,
    /// Plain UTF-8 message content.
    pub content: String,
}

impl ChatMessage {
    /// Construct a system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    /// Construct a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    /// Construct an assistant-history message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Format messages with the frozen NileMini chat template.
///
/// The format is:
/// `<|bos|>(<|role|>\ncontent<|eos|>\n)+<|assistant|>\n`.
///
/// # Errors
///
/// Returns an invalid-input error for an empty conversation, empty content,
/// reserved-token injection, a non-leading system message, or a conversation
/// that does not end with a user message.
pub fn format_chat_prompt(messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        return Err(EngineError::invalid_input(
            "chat template",
            "at least one message is required",
        ));
    }

    let mut prompt = String::from("<|bos|>");
    for (index, message) in messages.iter().enumerate() {
        if message.content.trim().is_empty() {
            return Err(EngineError::invalid_input(
                "chat template",
                format!("message {index} content must not be empty"),
            ));
        }
        if message.role == ChatRole::System && index != 0 {
            return Err(EngineError::invalid_input(
                "chat template",
                "a system message is only permitted at index zero",
            ));
        }
        if RESERVED_MARKERS
            .iter()
            .any(|marker| message.content.contains(marker))
        {
            return Err(EngineError::invalid_input(
                "chat template",
                format!("message {index} contains a reserved token marker"),
            ));
        }

        prompt.push_str(message.role.marker());
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|eos|>\n");
    }

    if messages.last().map(|message| message.role) != Some(ChatRole::User) {
        return Err(EngineError::invalid_input(
            "chat template",
            "the final message must have role user",
        ));
    }

    prompt.push_str("<|assistant|>\n");
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::{format_chat_prompt, ChatMessage};

    #[test]
    fn one_user_message_matches_the_frozen_template() {
        let prompt = format_chat_prompt(&[ChatMessage::user("Hello")]).expect("valid conversation");
        assert_eq!(prompt, "<|bos|><|user|>\nHello<|eos|>\n<|assistant|>\n");
    }

    #[test]
    fn system_and_history_messages_are_preserved_in_order() {
        let prompt = format_chat_prompt(&[
            ChatMessage::system("Be concise."),
            ChatMessage::user("One"),
            ChatMessage::assistant("Two"),
            ChatMessage::user("Three"),
        ])
        .expect("valid conversation");

        assert!(prompt.starts_with("<|bos|><|system|>\nBe concise.<|eos|>\n"));
        assert!(prompt.ends_with("<|user|>\nThree<|eos|>\n<|assistant|>\n"));
    }

    #[test]
    fn unsafe_or_incomplete_conversations_are_rejected() {
        assert!(format_chat_prompt(&[]).is_err());
        assert!(format_chat_prompt(&[ChatMessage::user("  ")]).is_err());
        assert!(format_chat_prompt(&[ChatMessage::user("<|assistant|>")]).is_err());
        assert!(
            format_chat_prompt(&[ChatMessage::user("Hello"), ChatMessage::assistant("Hi"),])
                .is_err()
        );
    }
}
