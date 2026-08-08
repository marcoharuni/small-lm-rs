//! Model configuration parsing and validation.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// Architecture configuration shared with the JAX reference implementation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Human-readable artifact name.
    pub model_name: String,
    /// Architecture identifier.
    pub architecture: String,
    /// Normalization operation.
    pub normalization: String,
    /// Position encoding operation.
    pub position_encoding: String,
    /// Feed-forward activation.
    pub activation: String,
    /// Exported matrix layout.
    pub weight_layout: String,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Maximum number of tokens in one sequence.
    pub context_length: usize,
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Width of each hidden state.
    pub hidden_size: usize,
    /// Width of the SiTU-GLU intermediate representation.
    pub intermediate_size: usize,
    /// Number of attention query heads.
    pub num_query_heads: usize,
    /// Number of grouped key/value heads.
    pub num_key_value_heads: usize,
    /// Width of one attention head.
    pub head_dimension: usize,
    /// Epsilon used by RMSNorm.
    pub rms_norm_epsilon: f32,
    /// Base frequency used by rotary embeddings.
    pub rope_theta: f32,
    /// SiTU beta applied to the gate projection.
    pub situ_beta_gate: f32,
    /// SiTU beta applied to the up projection.
    pub situ_beta_up: f32,
    /// Whether the token embedding and output projection share weights.
    pub tie_word_embeddings: bool,
    /// Whether linear projections include bias terms.
    pub use_bias: bool,
    /// Dropout probability used during training.
    pub dropout: f32,
    /// Expected parameter count used as an artifact integrity check.
    pub expected_parameter_count: u64,
}

impl ModelConfig {
    /// Load a configuration from a UTF-8 JSON file and validate it.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::ArtifactIo`] when the file cannot be read,
    /// [`EngineError::Json`] when it cannot be decoded, or
    /// [`EngineError::InvalidConfiguration`] when a model invariant is broken.
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| EngineError::ArtifactIo {
            artifact: "model configuration",
            path: path.to_path_buf(),
            source,
        })?;
        let config =
            serde_json::from_str::<Self>(&contents).map_err(|source| EngineError::Json {
                artifact: "model configuration",
                path: path.to_path_buf(),
                source,
            })?;
        config.validate()?;
        Ok(config)
    }

    /// Validate architecture invariants relied upon by the Rust engine.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidConfiguration`] for any invalid field or
    /// inconsistent dimension.
    pub fn validate(&self) -> Result<()> {
        if self.model_name.trim().is_empty() {
            return Err(EngineError::invalid_configuration(
                "model_name must not be empty",
            ));
        }
        if self.architecture != "decoder-only-transformer" {
            return Err(EngineError::invalid_configuration(
                "architecture must be decoder-only-transformer",
            ));
        }
        if self.normalization != "RMSNorm" {
            return Err(EngineError::invalid_configuration(
                "normalization must be RMSNorm",
            ));
        }
        if self.position_encoding != "RoPE" {
            return Err(EngineError::invalid_configuration(
                "position_encoding must be RoPE",
            ));
        }
        if self.activation != "SiTU-GLU" {
            return Err(EngineError::invalid_configuration(
                "activation must be SiTU-GLU",
            ));
        }
        if self.weight_layout != "out_features,in_features" {
            return Err(EngineError::invalid_configuration(
                "weight_layout must be out_features,in_features",
            ));
        }
        if !self.tie_word_embeddings {
            return Err(EngineError::invalid_configuration(
                "tie_word_embeddings must be true",
            ));
        }
        if self.use_bias {
            return Err(EngineError::invalid_configuration("use_bias must be false"));
        }

        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("context_length", self.context_length),
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_query_heads", self.num_query_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dimension", self.head_dimension),
        ] {
            if value == 0 {
                return Err(EngineError::invalid_configuration(format!(
                    "{name} must be greater than zero"
                )));
            }
        }

        if self.num_query_heads % self.num_key_value_heads != 0 {
            return Err(EngineError::invalid_configuration(
                "num_query_heads must be divisible by num_key_value_heads",
            ));
        }

        let projected_hidden = self
            .num_query_heads
            .checked_mul(self.head_dimension)
            .ok_or_else(|| {
                EngineError::invalid_configuration("num_query_heads * head_dimension exceeds usize")
            })?;
        if projected_hidden != self.hidden_size {
            return Err(EngineError::invalid_configuration(
                "num_query_heads * head_dimension must equal hidden_size",
            ));
        }

        if !self.rms_norm_epsilon.is_finite() || self.rms_norm_epsilon <= 0.0 {
            return Err(EngineError::invalid_configuration(
                "rms_norm_epsilon must be finite and greater than zero",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(EngineError::invalid_configuration(
                "rope_theta must be finite and greater than zero",
            ));
        }
        if !self.situ_beta_gate.is_finite() || self.situ_beta_gate <= 0.0 {
            return Err(EngineError::invalid_configuration(
                "situ_beta_gate must be finite and greater than zero",
            ));
        }
        if !self.situ_beta_up.is_finite() || self.situ_beta_up <= 0.0 {
            return Err(EngineError::invalid_configuration(
                "situ_beta_up must be finite and greater than zero",
            ));
        }
        if !self.dropout.is_finite() || !(0.0..1.0).contains(&self.dropout) {
            return Err(EngineError::invalid_configuration(
                "dropout must be finite and in the half-open range [0, 1)",
            ));
        }
        if self.expected_parameter_count == 0 {
            return Err(EngineError::invalid_configuration(
                "expected_parameter_count must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    fn valid_config() -> ModelConfig {
        ModelConfig {
            model_name: "nilemini-8m-situ".to_owned(),
            architecture: "decoder-only-transformer".to_owned(),
            normalization: "RMSNorm".to_owned(),
            position_encoding: "RoPE".to_owned(),
            activation: "SiTU-GLU".to_owned(),
            weight_layout: "out_features,in_features".to_owned(),
            vocab_size: 8_192,
            context_length: 512,
            num_layers: 8,
            hidden_size: 256,
            intermediate_size: 1_664,
            num_query_heads: 4,
            num_key_value_heads: 2,
            head_dimension: 64,
            rms_norm_epsilon: 1.0e-5,
            rope_theta: 10_000.0,
            situ_beta_gate: 4.0,
            situ_beta_up: 25.0,
            tie_word_embeddings: true,
            use_bias: false,
            dropout: 0.0,
            expected_parameter_count: 7_999_744,
        }
    }

    #[test]
    fn canonical_dimensions_validate() {
        valid_config().validate().expect("configuration is valid");
    }

    #[test]
    fn exported_configuration_loads() {
        let json = include_str!("../../../artifacts/nilemini-8m-situ/config.json");
        let config: ModelConfig =
            serde_json::from_str(json).expect("exported configuration must decode");

        config
            .validate()
            .expect("exported configuration must validate");
        assert_eq!(config.expected_parameter_count, 7_999_744);
    }

    #[test]
    fn inconsistent_attention_dimensions_fail() {
        let mut config = valid_config();
        config.head_dimension = 32;

        let error = config.validate().expect_err("configuration must fail");
        assert!(error.to_string().contains("must equal hidden_size"));
    }

    #[test]
    fn dropout_probability_must_be_less_than_one() {
        let mut config = valid_config();
        config.dropout = 1.0;

        let error = config.validate().expect_err("dropout must fail");
        assert!(error.to_string().contains("half-open range [0, 1)"));
    }

    #[test]
    fn unknown_json_fields_are_rejected() {
        let mut value = serde_json::to_value(valid_config()).expect("serializable config");
        value
            .as_object_mut()
            .expect("configuration is an object")
            .insert("unexpected".to_owned(), serde_json::Value::Bool(true));

        let error = serde_json::from_value::<ModelConfig>(value)
            .expect_err("unknown fields must not be accepted");
        assert!(error.to_string().contains("unknown field"));
    }
}
