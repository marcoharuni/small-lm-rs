//! Hugging Face tokenizer loading and text/token conversion.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::generation_config::GenerationConfig;

/// Exact identifiers assigned to SmallLM's reserved tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecialTokenIds {
    /// Padding token `<|pad|>`.
    pub pad: u32,
    /// Beginning-of-sequence token `<|bos|>`.
    pub bos: u32,
    /// End-of-sequence token `<|eos|>`.
    pub eos: u32,
    /// System-role marker `<|system|>`.
    pub system: u32,
    /// User-role marker `<|user|>`.
    pub user: u32,
    /// Assistant-role marker `<|assistant|>`.
    pub assistant: u32,
}

/// A validated wrapper around the tokenizer used by SmallLM artifacts.
pub struct SmallLMTokenizer {
    inner: Tokenizer,
}

impl std::fmt::Debug for SmallLMTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SmallLMTokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

impl SmallLMTokenizer {
    /// Load a Hugging Face `tokenizer.json` artifact.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::TokenizerLoad`] when the tokenizer cannot be read
    /// or decoded.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = Tokenizer::from_file(path).map_err(|source| EngineError::TokenizerLoad {
            path: path.to_path_buf(),
            message: source.to_string(),
        })?;
        Ok(Self { inner })
    }

    /// Encode text into token identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Tokenizer`] if the tokenizer pipeline fails.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|source| EngineError::Tokenizer {
                message: source.to_string(),
            })
    }

    /// Decode token identifiers into text.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Tokenizer`] if the tokenizer pipeline fails.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|source| EngineError::Tokenizer {
                message: source.to_string(),
            })
    }

    /// Resolve a token string to its vocabulary identifier.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when the requested token is absent.
    pub fn token_id(&self, token: &str) -> Result<u32> {
        self.inner.token_to_id(token).ok_or_else(|| {
            EngineError::invalid_input(
                "tokenizer contract",
                format!("required token {token:?} is absent"),
            )
        })
    }

    /// Return the exact identifiers of every reserved SmallLM token.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when any required token is absent.
    pub fn special_token_ids(&self) -> Result<SpecialTokenIds> {
        Ok(SpecialTokenIds {
            pad: self.token_id("<|pad|>")?,
            bos: self.token_id("<|bos|>")?,
            eos: self.token_id("<|eos|>")?,
            system: self.token_id("<|system|>")?,
            user: self.token_id("<|user|>")?,
            assistant: self.token_id("<|assistant|>")?,
        })
    }

    /// Validate vocabulary size and reserved identifiers against exported
    /// model and generation metadata.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for any tokenizer-contract mismatch.
    pub fn validate_contract(
        &self,
        model_config: &ModelConfig,
        generation_config: &GenerationConfig,
    ) -> Result<SpecialTokenIds> {
        if self.vocab_size() != model_config.vocab_size {
            return Err(EngineError::invalid_input(
                "tokenizer contract",
                format!(
                    "tokenizer vocabulary {} does not match model vocabulary {}",
                    self.vocab_size(),
                    model_config.vocab_size
                ),
            ));
        }

        let ids = self.special_token_ids()?;
        let expected = SpecialTokenIds {
            pad: 0,
            bos: 1,
            eos: 2,
            system: 3,
            user: 4,
            assistant: 5,
        };
        if ids != expected {
            return Err(EngineError::invalid_input(
                "tokenizer contract",
                format!("reserved token identifiers are {ids:?}, expected {expected:?}"),
            ));
        }
        if ids.pad != generation_config.pad_token_id || ids.eos != generation_config.eos_token_id {
            return Err(EngineError::invalid_input(
                "tokenizer contract",
                "tokenizer pad/EOS identifiers do not match generation metadata",
            ));
        }
        Ok(ids)
    }

    /// Return the tokenizer vocabulary size, including added tokens.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
