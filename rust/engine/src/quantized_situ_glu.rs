//! SiTU-GLU execution backed by weight-only INT8 projections.

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::quantized_linear::linear_int8;
use crate::quantized_weights::QuantizedMatrix;
use crate::situ_glu::SituGlu;

/// Quantized form of the SmallLM SiTU-GLU feed-forward layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizedSituGlu {
    operation: SituGlu,
}

impl QuantizedSituGlu {
    /// Build quantized SiTU-GLU metadata from model configuration.
    ///
    /// # Errors
    ///
    /// Returns model-configuration validation errors.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            operation: SituGlu::from_config(config)?,
        })
    }

    /// Run gate, up, and down projections through the INT8 reference kernel.
    ///
    /// # Errors
    ///
    /// Returns invalid-input or quantized-projection errors.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        gate_weight: &QuantizedMatrix,
        up_weight: &QuantizedMatrix,
        down_weight: &QuantizedMatrix,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.operation.hidden_size();
        if hidden_states.is_empty() || hidden_states.len() % hidden_size != 0 {
            return Err(EngineError::invalid_input(
                "quantized SiTU-GLU",
                format!("hidden-state length must be a non-zero multiple of {hidden_size}"),
            ));
        }

        let rows = hidden_states.len() / hidden_size;
        let intermediate = self.operation.intermediate_size();
        let gate = linear_int8(hidden_states, rows, hidden_size, gate_weight, intermediate)?;
        let up = linear_int8(hidden_states, rows, hidden_size, up_weight, intermediate)?;

        let product = gate
            .into_iter()
            .zip(up)
            .map(|(gate_value, up_value)| {
                self.operation.gate_activation(gate_value) * self.operation.up_activation(up_value)
            })
            .collect::<Vec<_>>();

        linear_int8(&product, rows, intermediate, down_weight, hidden_size)
    }
}

#[cfg(test)]
mod tests {
    use super::QuantizedSituGlu;
    use crate::config::ModelConfig;
    use crate::quantized_weights::QuantizedMatrix;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            model_name: "tiny".to_owned(),
            architecture: "decoder-only-transformer".to_owned(),
            normalization: "RMSNorm".to_owned(),
            position_encoding: "RoPE".to_owned(),
            activation: "SiTU-GLU".to_owned(),
            weight_layout: "out_features,in_features".to_owned(),
            vocab_size: 8,
            context_length: 8,
            num_layers: 1,
            hidden_size: 2,
            intermediate_size: 2,
            num_query_heads: 1,
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
        }
    }

    fn identity() -> QuantizedMatrix {
        QuantizedMatrix::from_parts(vec![2, 2], vec![127, 0, 0, 127], vec![1.0 / 127.0; 2])
            .expect("valid identity")
    }

    #[test]
    fn quantized_feed_forward_preserves_shape() {
        let operation = QuantizedSituGlu::from_config(&tiny_config()).expect("valid operation");
        let identity = identity();
        let output = operation
            .forward(&[1.0, -1.0], &identity, &identity, &identity)
            .expect("valid quantized feed-forward");

        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
