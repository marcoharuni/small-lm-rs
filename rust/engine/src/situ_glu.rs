//! SiTU-GLU feed-forward operation used by SmallLM.

use crate::config::ModelConfig;
use crate::error::{EngineError, Result};
use crate::linear::linear;

/// Shape and activation metadata for a SiTU-GLU feed-forward layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SituGlu {
    hidden_size: usize,
    intermediate_size: usize,
    beta_gate: f32,
    beta_up: f32,
}

impl SituGlu {
    /// Build SiTU-GLU metadata from a validated model configuration.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error when model invariants fail.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            beta_gate: config.situ_beta_gate,
            beta_up: config.situ_beta_up,
        })
    }

    /// Return the model hidden width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Return the feed-forward intermediate width.
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Return the beta parameter for the gate projection.
    #[must_use]
    pub const fn beta_gate(&self) -> f32 {
        self.beta_gate
    }

    /// Return the beta parameter for the up projection.
    #[must_use]
    pub const fn beta_up(&self) -> f32 {
        self.beta_up
    }

    /// Apply the exact SiTU gate branch:
    /// `beta_gate * tanh(x / beta_gate) * sigmoid(x)`.
    #[must_use]
    pub fn gate_activation(&self, value: f32) -> f32 {
        self.beta_gate * (value / self.beta_gate).tanh() * sigmoid(value)
    }

    /// Apply the exact SiTU up branch:
    /// `beta_up * tanh(x / beta_up)`.
    #[must_use]
    pub fn up_activation(&self, value: f32) -> f32 {
        self.beta_up * (value / self.beta_up).tanh()
    }

    /// Run the complete bias-free SiTU-GLU feed-forward operation.
    ///
    /// Hidden states are row-major `[rows, hidden_size]`. Gate and up weights
    /// are `[intermediate_size, hidden_size]`; down weights are
    /// `[hidden_size, intermediate_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::InvalidInput`] for an empty or incorrectly shaped
    /// hidden-state matrix or any incorrectly sized weight matrix.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        gate_weight: &[f32],
        up_weight: &[f32],
        down_weight: &[f32],
    ) -> Result<Vec<f32>> {
        if hidden_states.is_empty() || hidden_states.len() % self.hidden_size != 0 {
            return Err(EngineError::invalid_input(
                "SiTU-GLU",
                format!(
                    "hidden-state length must be a non-zero multiple of {}",
                    self.hidden_size
                ),
            ));
        }

        let rows = hidden_states.len() / self.hidden_size;
        let gate = linear(
            hidden_states,
            rows,
            self.hidden_size,
            gate_weight,
            self.intermediate_size,
        )?;
        let up = linear(
            hidden_states,
            rows,
            self.hidden_size,
            up_weight,
            self.intermediate_size,
        )?;

        let product = gate
            .into_iter()
            .zip(up)
            .map(|(gate_value, up_value)| {
                self.gate_activation(gate_value) * self.up_activation(up_value)
            })
            .collect::<Vec<_>>();

        linear(
            &product,
            rows,
            self.intermediate_size,
            down_weight,
            self.hidden_size,
        )
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::SituGlu;
    use crate::config::ModelConfig;
    use crate::linear::round_to_bfloat16;

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

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn branch_formulas_match_the_reference_definition() {
        let operation = SituGlu::from_config(&tiny_config()).expect("valid operation");
        assert_close(operation.gate_activation(0.0), 0.0);
        assert_close(operation.up_activation(0.0), 0.0);

        let expected_gate = 4.0 * (0.25_f32).tanh() / (1.0 + (-1.0_f32).exp());
        assert_close(operation.gate_activation(1.0), expected_gate);
        assert_close(operation.up_activation(1.0), 25.0 * (0.04_f32).tanh());
    }

    #[test]
    fn complete_feed_forward_uses_exported_weight_layout() {
        let operation = SituGlu::from_config(&tiny_config()).expect("valid operation");
        let identity = [
            1.0, 0.0, //
            0.0, 1.0,
        ];

        let output = operation
            .forward(&[1.0, -1.0], &identity, &identity, &identity)
            .expect("valid feed-forward operation");

        let first_product = operation.gate_activation(1.0) * operation.up_activation(1.0);
        let second_product = operation.gate_activation(-1.0) * operation.up_activation(-1.0);
        assert_close(output[0], round_to_bfloat16(first_product));
        assert_close(output[1], round_to_bfloat16(second_product));
    }

    #[test]
    fn rejects_invalid_hidden_state_shape() {
        let operation = SituGlu::from_config(&tiny_config()).expect("valid operation");
        assert!(operation
            .forward(&[1.0], &[0.0; 4], &[0.0; 4], &[0.0; 4])
            .is_err());
    }
}
