//! Loading and validation of mixed FP32/INT8 SmallLM artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors};

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};

const SCALE_SUFFIX: &str = ".scale";

/// One validated per-output-channel INT8 projection matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedMatrix {
    shape: Vec<usize>,
    values: Vec<i8>,
    scales: Vec<f32>,
}

impl QuantizedMatrix {
    /// Return `[out_features, in_features]` dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return flattened row-major INT8 values.
    #[must_use]
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Return one symmetric scale per output row.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
}

/// One engine-owned FP32 tensor retained by the quantized artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl FloatTensor {
    /// Return tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return flattened FP32 values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Fully loaded mixed-precision weight artifact.
#[derive(Debug)]
pub struct QuantizedModelWeights {
    path: PathBuf,
    float_tensors: BTreeMap<String, FloatTensor>,
    projections: BTreeMap<String, QuantizedMatrix>,
}

impl QuantizedModelWeights {
    /// Load and validate one SmallLM INT8 artifact.
    ///
    /// Embeddings and normalization tensors remain FP32. Transformer
    /// projection matrices use I8 values plus a same-name `.scale` FP32 vector
    /// with one scale per output channel.
    ///
    /// # Errors
    ///
    /// Returns I/O, SafeTensors, dtype, shape, naming, or scale-validation
    /// errors.
    pub fn load(path: impl AsRef<Path>, config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| EngineError::ArtifactIo {
            artifact: "INT8 model weights",
            path: path.to_path_buf(),
            source,
        })?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(|source| EngineError::SafeTensors {
            path: path.to_path_buf(),
            source,
        })?;

        let layout = expected_layout(config)?;
        validate_names(&tensors, &layout)?;

        let mut float_tensors = BTreeMap::new();
        let mut projections = BTreeMap::new();
        for (name, spec) in layout {
            match spec {
                TensorSpec::Float(shape) => {
                    let view = tensor_view(&tensors, path, &name)?;
                    if view.dtype() != Dtype::F32 {
                        return Err(EngineError::invalid_weights(format!(
                            "{name} must have dtype F32, found {:?}",
                            view.dtype()
                        )));
                    }
                    validate_shape(&name, view.shape(), &shape)?;
                    float_tensors.insert(
                        name.clone(),
                        FloatTensor {
                            shape,
                            values: decode_f32(&name, view.shape(), view.data())?,
                        },
                    );
                }
                TensorSpec::Int8Projection(shape) => {
                    let view = tensor_view(&tensors, path, &name)?;
                    if view.dtype() != Dtype::I8 {
                        return Err(EngineError::invalid_weights(format!(
                            "{name} must have dtype I8, found {:?}",
                            view.dtype()
                        )));
                    }
                    validate_shape(&name, view.shape(), &shape)?;
                    let values = decode_i8(&name, view.shape(), view.data())?;

                    let scale_name = format!("{name}{SCALE_SUFFIX}");
                    let scale_view = tensor_view(&tensors, path, &scale_name)?;
                    if scale_view.dtype() != Dtype::F32 {
                        return Err(EngineError::invalid_weights(format!(
                            "{scale_name} must have dtype F32, found {:?}",
                            scale_view.dtype()
                        )));
                    }
                    let out_features = shape[0];
                    validate_shape(&scale_name, scale_view.shape(), &[out_features])?;
                    let scales = decode_f32(&scale_name, scale_view.shape(), scale_view.data())?;
                    if scales.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) {
                        return Err(EngineError::invalid_weights(format!(
                            "{scale_name} must contain finite positive scales"
                        )));
                    }

                    projections.insert(
                        name,
                        QuantizedMatrix {
                            shape,
                            values,
                            scales,
                        },
                    );
                }
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            float_tensors,
            projections,
        })
    }

    /// Return the source artifact path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return one required retained FP32 tensor.
    ///
    /// # Errors
    ///
    /// Returns an invalid-model-weights error if the tensor is absent.
    pub fn required_float(&self, name: &str) -> Result<&FloatTensor> {
        self.float_tensors.get(name).ok_or_else(|| {
            EngineError::invalid_weights(format!("required FP32 tensor {name} is missing"))
        })
    }

    /// Return one required INT8 transformer projection.
    ///
    /// # Errors
    ///
    /// Returns an invalid-model-weights error if the projection is absent.
    pub fn required_projection(&self, name: &str) -> Result<&QuantizedMatrix> {
        self.projections.get(name).ok_or_else(|| {
            EngineError::invalid_weights(format!("required INT8 projection {name} is missing"))
        })
    }

    /// Return the number of quantized projection matrices.
    #[must_use]
    pub fn projection_count(&self) -> usize {
        self.projections.len()
    }
}

#[derive(Clone, Debug)]
enum TensorSpec {
    Float(Vec<usize>),
    Int8Projection(Vec<usize>),
}

fn expected_layout(config: &ModelConfig) -> Result<BTreeMap<String, TensorSpec>> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dimension)
        .ok_or_else(|| EngineError::invalid_weights("KV projection width exceeds usize"))?;

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embedding.weight".to_owned(),
        TensorSpec::Float(vec![config.vocab_size, hidden]),
    );

    for layer in 0..config.num_layers {
        let prefix = format!("layers.{layer}");
        tensors.insert(
            format!("{prefix}.attention_norm.weight"),
            TensorSpec::Float(vec![hidden]),
        );
        tensors.insert(
            format!("{prefix}.q_proj.weight"),
            TensorSpec::Int8Projection(vec![hidden, hidden]),
        );
        tensors.insert(
            format!("{prefix}.k_proj.weight"),
            TensorSpec::Int8Projection(vec![kv_width, hidden]),
        );
        tensors.insert(
            format!("{prefix}.v_proj.weight"),
            TensorSpec::Int8Projection(vec![kv_width, hidden]),
        );
        tensors.insert(
            format!("{prefix}.o_proj.weight"),
            TensorSpec::Int8Projection(vec![hidden, hidden]),
        );
        tensors.insert(
            format!("{prefix}.ffn_norm.weight"),
            TensorSpec::Float(vec![hidden]),
        );
        tensors.insert(
            format!("{prefix}.gate_proj.weight"),
            TensorSpec::Int8Projection(vec![intermediate, hidden]),
        );
        tensors.insert(
            format!("{prefix}.up_proj.weight"),
            TensorSpec::Int8Projection(vec![intermediate, hidden]),
        );
        tensors.insert(
            format!("{prefix}.down_proj.weight"),
            TensorSpec::Int8Projection(vec![hidden, intermediate]),
        );
    }

    tensors.insert(
        "final_norm.weight".to_owned(),
        TensorSpec::Float(vec![hidden]),
    );
    Ok(tensors)
}

fn validate_names(tensors: &SafeTensors<'_>, layout: &BTreeMap<String, TensorSpec>) -> Result<()> {
    let mut expected = BTreeSet::new();
    for (name, spec) in layout {
        expected.insert(name.clone());
        if matches!(spec, TensorSpec::Int8Projection(_)) {
            expected.insert(format!("{name}{SCALE_SUFFIX}"));
        }
    }
    let actual = tensors
        .names()
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();

    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
    if missing.is_empty() && unexpected.is_empty() {
        Ok(())
    } else {
        Err(EngineError::invalid_weights(format!(
            "INT8 tensor-name mismatch; missing={missing:?}; unexpected={unexpected:?}"
        )))
    }
}

fn tensor_view<'a>(
    tensors: &'a SafeTensors<'a>,
    path: &Path,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    tensors
        .tensor(name)
        .map_err(|source| EngineError::SafeTensors {
            path: path.to_path_buf(),
            source,
        })
}

fn validate_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(EngineError::invalid_weights(format!(
            "{name} has shape {actual:?}, expected {expected:?}"
        )))
    }
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| EngineError::invalid_weights("tensor element count exceeds usize"))
    })
}

fn decode_i8(name: &str, shape: &[usize], bytes: &[u8]) -> Result<Vec<i8>> {
    let expected = element_count(shape)?;
    if bytes.len() != expected {
        return Err(EngineError::invalid_weights(format!(
            "{name} contains {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok(bytes.iter().map(|value| i8::from_le_bytes([*value])).collect())
}

fn decode_f32(name: &str, shape: &[usize], bytes: &[u8]) -> Result<Vec<f32>> {
    let count = element_count(shape)?;
    let expected_bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| EngineError::invalid_weights("tensor byte count exceeds usize"))?;
    if bytes.len() != expected_bytes {
        return Err(EngineError::invalid_weights(format!(
            "{name} contains {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::expected_layout;
    use crate::config::ModelConfig;

    #[test]
    fn canonical_layout_quantizes_only_projection_matrices() {
        let config = ModelConfig {
            model_name: "tiny".to_owned(),
            architecture: "decoder-only-transformer".to_owned(),
            normalization: "RMSNorm".to_owned(),
            position_encoding: "RoPE".to_owned(),
            activation: "SiTU-GLU".to_owned(),
            weight_layout: "out_features,in_features".to_owned(),
            vocab_size: 8,
            context_length: 8,
            num_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_query_heads: 2,
            num_key_value_heads: 1,
            head_dimension: 2,
            rms_norm_epsilon: 1.0e-5,
            rope_theta: 10_000.0,
            situ_beta_gate: 4.0,
            situ_beta_up: 25.0,
            tie_word_embeddings: true,
            use_bias: false,
            dropout: 0.0,
            expected_parameter_count: 1,
        };
        let layout = expected_layout(&config).expect("valid layout");
        assert_eq!(layout.len(), 11);
        assert!(matches!(
            layout.get("layers.0.q_proj.weight"),
            Some(super::TensorSpec::Int8Projection(_))
        ));
        assert!(matches!(
            layout.get("token_embedding.weight"),
            Some(super::TensorSpec::Float(_))
        ));
    }
}
