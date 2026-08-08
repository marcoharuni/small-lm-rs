//! Loading of JAX reference inputs and output logits.

use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// Token sequences exported by the JAX reference implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceInputs {
    token_ids: Vec<Vec<u32>>,
}

impl ReferenceInputs {
    /// Load and validate a `reference_inputs.json` artifact.
    ///
    /// # Errors
    ///
    /// Returns an artifact I/O error, JSON decoding error, or invalid-input
    /// error when the batch is empty, sequences are empty, or sequence lengths
    /// differ.
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| EngineError::ArtifactIo {
            artifact: "parity reference inputs",
            path: path.to_path_buf(),
            source,
        })?;
        let inputs =
            serde_json::from_str::<Self>(&contents).map_err(|source| EngineError::Json {
                artifact: "parity reference inputs",
                path: path.to_path_buf(),
                source,
            })?;
        inputs.validate()?;
        Ok(inputs)
    }

    /// Validate that the reference contains a rectangular, non-empty batch.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an empty batch, an empty sequence, or
    /// inconsistent sequence lengths.
    pub fn validate(&self) -> Result<()> {
        let Some(first) = self.token_ids.first() else {
            return Err(EngineError::invalid_input(
                "parity reference inputs",
                "token_ids must contain at least one sequence",
            ));
        };
        if first.is_empty() || self.token_ids.iter().any(Vec::is_empty) {
            return Err(EngineError::invalid_input(
                "parity reference inputs",
                "reference sequences must not be empty",
            ));
        }
        if self
            .token_ids
            .iter()
            .any(|sequence| sequence.len() != first.len())
        {
            return Err(EngineError::invalid_input(
                "parity reference inputs",
                "all reference sequences must have the same length",
            ));
        }
        Ok(())
    }

    /// Return the exported token sequences.
    #[must_use]
    pub fn sequences(&self) -> &[Vec<u32>] {
        &self.token_ids
    }

    /// Return the number of reference sequences.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.token_ids.len()
    }

    /// Return the common reference sequence length.
    #[must_use]
    pub fn sequence_length(&self) -> usize {
        self.token_ids.first().map_or(0, Vec::len)
    }
}

/// FP32 vocabulary logits exported by the JAX reference implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceLogits {
    path: PathBuf,
    batch_size: usize,
    sequence_length: usize,
    vocab_size: usize,
    values: Vec<f32>,
}

impl ReferenceLogits {
    /// Load and validate the `logits` tensor from a SafeTensors artifact.
    ///
    /// The expected tensor shape is `[batch, sequence, vocabulary]` with FP32
    /// values.
    ///
    /// # Errors
    ///
    /// Returns artifact I/O, SafeTensors, or invalid-input errors when the
    /// artifact is malformed or violates the reference-logit contract.
    pub fn from_safetensors_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| EngineError::ArtifactIo {
            artifact: "parity reference logits",
            path: path.to_path_buf(),
            source,
        })?;
        let tensors =
            SafeTensors::deserialize(&bytes).map_err(|source| EngineError::SafeTensors {
                path: path.to_path_buf(),
                source,
            })?;

        let names = tensors.names();
        if names.len() != 1 || names[0] != "logits" {
            return Err(EngineError::invalid_input(
                "parity reference logits",
                format!("expected only a logits tensor, found {names:?}"),
            ));
        }

        let view = tensors
            .tensor("logits")
            .map_err(|source| EngineError::SafeTensors {
                path: path.to_path_buf(),
                source,
            })?;
        if view.dtype() != Dtype::F32 {
            return Err(EngineError::invalid_input(
                "parity reference logits",
                format!("logits must have dtype F32, found {:?}", view.dtype()),
            ));
        }
        if view.shape().len() != 3 || view.shape().contains(&0) {
            return Err(EngineError::invalid_input(
                "parity reference logits",
                format!(
                    "logits must have non-zero [batch, sequence, vocabulary] shape, found {:?}",
                    view.shape()
                ),
            ));
        }

        let batch_size = view.shape()[0];
        let sequence_length = view.shape()[1];
        let vocab_size = view.shape()[2];
        let element_count = batch_size
            .checked_mul(sequence_length)
            .and_then(|count| count.checked_mul(vocab_size))
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "parity reference logits",
                    "reference tensor element count overflows usize",
                )
            })?;
        let expected_bytes = element_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::invalid_input(
                    "parity reference logits",
                    "reference tensor byte count overflows usize",
                )
            })?;
        if view.data().len() != expected_bytes {
            return Err(EngineError::invalid_input(
                "parity reference logits",
                format!(
                    "logits contain {} bytes, expected {expected_bytes}",
                    view.data().len()
                ),
            ));
        }

        let values = view
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();

        Ok(Self {
            path: path.to_path_buf(),
            batch_size,
            sequence_length,
            vocab_size,
            values,
        })
    }

    /// Return the source artifact path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the reference batch size.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Return the reference sequence length.
    #[must_use]
    pub const fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Return the reference vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Return flattened row-major reference logits.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{ReferenceInputs, ReferenceLogits};

    #[test]
    fn validates_rectangular_reference_inputs() {
        let inputs: ReferenceInputs =
            serde_json::from_str(r#"{"token_ids":[[1,2,3],[4,5,6]]}"#).expect("valid JSON");
        inputs.validate().expect("valid reference inputs");
        assert_eq!(inputs.batch_size(), 2);
        assert_eq!(inputs.sequence_length(), 3);
    }

    #[test]
    fn rejects_ragged_reference_inputs() {
        let inputs: ReferenceInputs =
            serde_json::from_str(r#"{"token_ids":[[1,2],[3]]}"#).expect("valid JSON");
        assert!(inputs.validate().is_err());
    }

    #[test]
    fn loads_exported_reference_artifacts_when_available() {
        let artifact =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/nilemini-8m-situ");
        let inputs_path = artifact.join("reference_inputs.json");
        let outputs_path = artifact.join("reference_outputs.safetensors");
        if !inputs_path.is_file() || !outputs_path.is_file() {
            return;
        }

        let inputs = ReferenceInputs::from_json_path(inputs_path).expect("valid reference inputs");
        let logits =
            ReferenceLogits::from_safetensors_path(outputs_path).expect("valid reference logits");
        assert_eq!(inputs.batch_size(), 1);
        assert_eq!(inputs.sequence_length(), 10);
        assert_eq!(logits.batch_size(), 1);
        assert_eq!(logits.sequence_length(), 10);
        assert_eq!(logits.vocab_size(), 8_192);
        assert_eq!(logits.values().len(), 81_920);
    }
}
