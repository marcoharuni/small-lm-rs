//! Loading and validation of generation-specific artifact settings.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};

/// Generation metadata exported beside the model and tokenizer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    /// Maximum sequence length supported by the exported runtime.
    pub context_length: usize,
    /// Padding token identifier.
    pub pad_token_id: u32,
    /// End-of-sequence token identifier.
    pub eos_token_id: u32,
}

impl GenerationConfig {
    /// Load generation metadata from a UTF-8 JSON artifact.
    ///
    /// # Errors
    ///
    /// Returns artifact I/O, JSON decoding, or validation errors.
    pub fn from_json_path(path: impl AsRef<Path>, model_config: &ModelConfig) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| EngineError::ArtifactIo {
            artifact: "generation configuration",
            path: path.to_path_buf(),
            source,
        })?;
        let config =
            serde_json::from_str::<Self>(&contents).map_err(|source| EngineError::Json {
                artifact: "generation configuration",
                path: path.to_path_buf(),
                source,
            })?;
        config.validate(model_config)?;
        Ok(config)
    }

    /// Validate generation metadata against the model architecture.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for inconsistent context or
    /// token identifiers.
    pub fn validate(&self, model_config: &ModelConfig) -> Result<()> {
        if self.context_length != model_config.context_length {
            return Err(EngineError::invalid_configuration(format!(
                "generation context length {} does not match model context length {}",
                self.context_length, model_config.context_length
            )));
        }
        for (name, token_id) in [
            ("pad_token_id", self.pad_token_id),
            ("eos_token_id", self.eos_token_id),
        ] {
            if token_id as usize >= model_config.vocab_size {
                return Err(EngineError::invalid_configuration(format!(
                    "{name} {token_id} is outside vocabulary size {}",
                    model_config.vocab_size
                )));
            }
        }
        if self.pad_token_id == self.eos_token_id {
            return Err(EngineError::invalid_configuration(
                "pad_token_id and eos_token_id must be different",
            ));
        }
        Ok(())
    }
}
