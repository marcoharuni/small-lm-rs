//! Error types shared by engine components.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// The result type returned by engine operations.
pub type Result<T> = std::result::Result<T, EngineError>;

/// Structured failures produced by the inference engine scaffold.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A configuration value or relationship is invalid.
    #[error("invalid model configuration: {message}")]
    InvalidConfiguration {
        /// A human-readable description of the failed constraint.
        message: String,
    },

    /// An engine input has an invalid shape or value.
    #[error("invalid input for {component}: {message}")]
    InvalidInput {
        /// The component which rejected the input.
        component: &'static str,
        /// A human-readable description of the failed constraint.
        message: String,
    },

    /// An artifact could not be read from disk.
    #[error("failed to read {artifact} at {path}: {source}", path = path.display())]
    ArtifactIo {
        /// The kind of artifact being read.
        artifact: &'static str,
        /// The artifact path.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: io::Error,
    },

    /// A JSON artifact could not be decoded.
    #[error("failed to decode {artifact} JSON at {path}: {source}", path = path.display())]
    Json {
        /// The kind of artifact being decoded.
        artifact: &'static str,
        /// The artifact path.
        path: PathBuf,
        /// The JSON parser error.
        #[source]
        source: serde_json::Error,
    },

    /// A SafeTensors artifact is malformed or unsupported.
    #[error("invalid SafeTensors artifact at {path}: {source}", path = path.display())]
    SafeTensors {
        /// The artifact path.
        path: PathBuf,
        /// The SafeTensors parser error.
        #[source]
        source: safetensors::SafeTensorError,
    },

    /// Model weights do not match the exported architecture contract.
    #[error("invalid model weights: {message}")]
    InvalidWeights {
        /// Description of the tensor-contract violation.
        message: String,
    },

    /// A Hugging Face tokenizer artifact could not be loaded.
    #[error("failed to load tokenizer at {path}: {message}", path = path.display())]
    TokenizerLoad {
        /// The tokenizer path.
        path: PathBuf,
        /// The error reported by the tokenizer library.
        message: String,
    },

    /// A tokenizer operation failed.
    #[error("tokenizer operation failed: {message}")]
    Tokenizer {
        /// The error reported by the tokenizer library.
        message: String,
    },

    /// The requested component has not been implemented in the scaffold.
    #[error("{feature} is not implemented in the initial scaffold")]
    NotImplemented {
        /// The missing feature.
        feature: &'static str,
    },
}

impl EngineError {
    /// Construct an explicit not-implemented error for a named feature.
    #[must_use]
    pub const fn not_implemented(feature: &'static str) -> Self {
        Self::NotImplemented { feature }
    }

    /// Construct an invalid-configuration error.
    #[must_use]
    pub fn invalid_configuration(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Construct an invalid-model-weights error.
    #[must_use]
    pub fn invalid_weights(message: impl Into<String>) -> Self {
        Self::InvalidWeights {
            message: message.into(),
        }
    }

    /// Construct an invalid-input error for a named component.
    #[must_use]
    pub fn invalid_input(component: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidInput {
            component,
            message: message.into(),
        }
    }
}
