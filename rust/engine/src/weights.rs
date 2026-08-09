//! Loading and validation of exported FP32 SafeTensors weights.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors};

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};

/// Metadata describing a validated SafeTensors artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightMetadata {
    path: PathBuf,
    tensor_names: Vec<String>,
    parameter_count: u64,
}

impl WeightMetadata {
    /// Inspect a SafeTensors artifact without materializing its values.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, parsed, or its element
    /// count overflows.
    pub fn inspect(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = read_artifact(path)?;
        let tensors = deserialize(path, &bytes)?;

        let mut tensor_names = tensors
            .names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        tensor_names.sort_unstable();

        let mut parameter_count = 0_u64;
        for name in &tensor_names {
            let view = tensors
                .tensor(name)
                .map_err(|source| EngineError::SafeTensors {
                    path: path.to_path_buf(),
                    source,
                })?;

            parameter_count = parameter_count
                .checked_add(element_count(view.shape())?)
                .ok_or_else(|| EngineError::invalid_weights("total parameter count exceeds u64"))?;
        }

        Ok(Self {
            path: path.to_path_buf(),
            tensor_names,
            parameter_count,
        })
    }

    /// Return the artifact path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return sorted tensor names.
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Return the total number of scalar parameters.
    #[must_use]
    pub const fn parameter_count(&self) -> u64 {
        self.parameter_count
    }
}

/// One engine-owned FP32 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl TensorData {
    /// Return the tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return flattened row-major FP32 values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Fully loaded and validated model weights.
#[derive(Debug)]
pub struct ModelWeights {
    metadata: WeightMetadata,
    tensors: BTreeMap<String, TensorData>,
}

impl ModelWeights {
    /// Load and validate every expected model tensor.
    ///
    /// Validation covers tensor names, FP32 dtype, exact shapes, absence of
    /// unexpected tensors, and the configured total parameter count.
    ///
    /// # Errors
    ///
    /// Returns an error for I/O failures, malformed SafeTensors data, or any
    /// mismatch with the supplied model configuration.
    pub fn load(path: impl AsRef<Path>, config: &ModelConfig) -> Result<Self> {
        config.validate()?;

        let path = path.as_ref();
        let bytes = read_artifact(path)?;
        let safetensors = deserialize(path, &bytes)?;
        let expected = expected_layout(config)?;

        validate_names(&safetensors, &expected)?;

        let expected_count = layout_parameter_count(&expected)?;
        if expected_count != config.expected_parameter_count {
            return Err(EngineError::invalid_weights(format!(
                "configuration expects {} parameters but its tensor layout contains {expected_count}",
                config.expected_parameter_count
            )));
        }

        let mut loaded = BTreeMap::new();
        let mut parameter_count = 0_u64;

        for (name, expected_shape) in &expected {
            let view = safetensors
                .tensor(name)
                .map_err(|source| EngineError::SafeTensors {
                    path: path.to_path_buf(),
                    source,
                })?;

            if view.dtype() != Dtype::F32 {
                return Err(EngineError::invalid_weights(format!(
                    "{name} must have dtype F32, found {:?}",
                    view.dtype()
                )));
            }

            if view.shape() != expected_shape.as_slice() {
                return Err(EngineError::invalid_weights(format!(
                    "{name} has shape {:?}, expected {expected_shape:?}",
                    view.shape()
                )));
            }

            let values = decode_f32(name, view.shape(), view.data())?;
            parameter_count = parameter_count
                .checked_add(element_count(view.shape())?)
                .ok_or_else(|| EngineError::invalid_weights("tensor element count exceeds u64"))?;

            loaded.insert(
                name.clone(),
                TensorData {
                    shape: expected_shape.clone(),
                    values,
                },
            );
        }

        if parameter_count != config.expected_parameter_count {
            return Err(EngineError::invalid_weights(format!(
                "loaded {parameter_count} parameters, expected {}",
                config.expected_parameter_count
            )));
        }

        let tensor_names = loaded.keys().cloned().collect();

        Ok(Self {
            metadata: WeightMetadata {
                path: path.to_path_buf(),
                tensor_names,
                parameter_count,
            },
            tensors: loaded,
        })
    }

    #[cfg(test)]
    pub(crate) fn synthetic<F>(config: &ModelConfig, mut values_for: F) -> Result<Self>
    where
        F: FnMut(&str, &[usize]) -> Vec<f32>,
    {
        config.validate()?;
        let layout = expected_layout(config)?;
        let parameter_count = layout_parameter_count(&layout)?;
        if parameter_count != config.expected_parameter_count {
            return Err(EngineError::invalid_weights(format!(
                "synthetic layout contains {parameter_count} parameters, expected {}",
                config.expected_parameter_count
            )));
        }

        let mut tensors = BTreeMap::new();
        for (name, shape) in layout {
            let values = values_for(&name, &shape);
            let expected_values = usize::try_from(element_count(&shape)?).map_err(|_| {
                EngineError::invalid_weights("synthetic tensor is too large for this platform")
            })?;
            if values.len() != expected_values {
                return Err(EngineError::invalid_weights(format!(
                    "synthetic tensor {name} contains {} values, expected {expected_values}",
                    values.len()
                )));
            }
            tensors.insert(name, TensorData { shape, values });
        }

        let tensor_names = tensors.keys().cloned().collect();
        Ok(Self {
            metadata: WeightMetadata {
                path: PathBuf::from("<synthetic>"),
                tensor_names,
                parameter_count,
            },
            tensors,
        })
    }

    /// Return validated artifact metadata.
    #[must_use]
    pub const fn metadata(&self) -> &WeightMetadata {
        &self.metadata
    }

    /// Look up a tensor by its exported name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorData> {
        self.tensors.get(name)
    }

    /// Return values for one required exported tensor.
    ///
    /// # Errors
    ///
    /// Returns an invalid-model-weights error when the named tensor is absent.
    pub fn required_tensor_values(&self, name: &str) -> Result<&[f32]> {
        self.tensor(name).map(TensorData::values).ok_or_else(|| {
            EngineError::invalid_weights(format!("required tensor {name} is missing"))
        })
    }
}

fn read_artifact(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| EngineError::ArtifactIo {
        artifact: "model weights",
        path: path.to_path_buf(),
        source,
    })
}

fn deserialize<'a>(path: &Path, bytes: &'a [u8]) -> Result<SafeTensors<'a>> {
    SafeTensors::deserialize(bytes).map_err(|source| EngineError::SafeTensors {
        path: path.to_path_buf(),
        source,
    })
}

fn element_count(shape: &[usize]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |count, &dimension| {
        let dimension = u64::try_from(dimension)
            .map_err(|_| EngineError::invalid_weights("tensor dimension exceeds u64"))?;

        count
            .checked_mul(dimension)
            .ok_or_else(|| EngineError::invalid_weights("tensor element count exceeds u64"))
    })
}

fn decode_f32(name: &str, shape: &[usize], bytes: &[u8]) -> Result<Vec<f32>> {
    let count = usize::try_from(element_count(shape)?)
        .map_err(|_| EngineError::invalid_weights("tensor is too large for this platform"))?;

    let expected_bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| EngineError::invalid_weights("tensor byte count exceeds usize"))?;

    if bytes.len() != expected_bytes {
        return Err(EngineError::invalid_weights(format!(
            "{name} contains {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }

    let mut values = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    Ok(values)
}

fn validate_names(
    tensors: &SafeTensors<'_>,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    let actual = tensors
        .names()
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();

    let missing = expected_names
        .difference(&actual)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = actual
        .difference(&expected_names)
        .cloned()
        .collect::<Vec<_>>();

    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }

    Err(EngineError::invalid_weights(format!(
        "tensor-name mismatch; missing={missing:?}; unexpected={unexpected:?}"
    )))
}

fn expected_layout(config: &ModelConfig) -> Result<BTreeMap<String, Vec<usize>>> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dimension)
        .ok_or_else(|| EngineError::invalid_weights("KV projection width exceeds usize"))?;

    let mut tensors = BTreeMap::new();

    tensors.insert(
        "token_embedding.weight".to_owned(),
        vec![config.vocab_size, hidden],
    );

    for layer in 0..config.num_layers {
        let prefix = format!("layers.{layer}");

        tensors.insert(format!("{prefix}.attention_norm.weight"), vec![hidden]);
        tensors.insert(format!("{prefix}.q_proj.weight"), vec![hidden, hidden]);
        tensors.insert(format!("{prefix}.k_proj.weight"), vec![kv_width, hidden]);
        tensors.insert(format!("{prefix}.v_proj.weight"), vec![kv_width, hidden]);
        tensors.insert(format!("{prefix}.o_proj.weight"), vec![hidden, hidden]);

        tensors.insert(format!("{prefix}.ffn_norm.weight"), vec![hidden]);
        tensors.insert(
            format!("{prefix}.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        tensors.insert(
            format!("{prefix}.up_proj.weight"),
            vec![intermediate, hidden],
        );
        tensors.insert(
            format!("{prefix}.down_proj.weight"),
            vec![hidden, intermediate],
        );
    }

    tensors.insert("final_norm.weight".to_owned(), vec![hidden]);

    Ok(tensors)
}

fn layout_parameter_count(layout: &BTreeMap<String, Vec<usize>>) -> Result<u64> {
    layout.values().try_fold(0_u64, |total, shape| {
        total
            .checked_add(element_count(shape)?)
            .ok_or_else(|| EngineError::invalid_weights("layout parameter count exceeds u64"))
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::ModelWeights;
    use crate::config::ModelConfig;

    #[test]
    fn bundled_weights_load_and_validate() {
        let artifact =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/small-lm-8m");
        let model_path = artifact.join("model.safetensors");
        let config =
            ModelConfig::from_json_path(artifact.join("config.json")).expect("valid config");
        let weights = ModelWeights::load(model_path, &config).expect("valid exported weights");

        assert_eq!(weights.metadata().tensor_names().len(), 74);
        assert_eq!(weights.metadata().parameter_count(), 7_999_744);
        assert_eq!(
            weights
                .required_tensor_values("final_norm.weight")
                .expect("required final norm")
                .len(),
            256
        );
        assert_eq!(
            weights
                .tensor("token_embedding.weight")
                .expect("embedding tensor")
                .shape(),
            &[8_192, 256]
        );
    }
}
