//! Execution and acceptance of complete JAX-to-Rust parity reports.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{ParityMetrics, ReferenceInputs, ReferenceLogits};
use crate::error::{EngineError, Result};
use crate::model::NileMiniModel;

/// Baseline acceptance thresholds for CPU-versus-JAX/GPU logit parity.
///
/// Scalar-error, cosine-similarity, overall top-1, and final next-token checks
/// are combined so a tensor-layout error cannot be hidden by one aggregate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParityThresholds {
    /// Largest permitted absolute scalar error.
    pub max_abs_error: f64,
    /// Largest permitted mean absolute scalar error.
    pub mean_abs_error: f64,
    /// Largest permitted root-mean-square scalar error.
    pub root_mean_square_error: f64,
    /// Smallest permitted flattened cosine similarity.
    pub min_cosine_similarity: f64,
    /// Smallest permitted per-position top-1 agreement fraction.
    pub min_top1_agreement: f64,
    /// Whether the final-position next-token prediction must match exactly.
    pub require_final_position_top1_match: bool,
}

impl Default for ParityThresholds {
    fn default() -> Self {
        Self {
            max_abs_error: 0.5,
            mean_abs_error: 0.05,
            root_mean_square_error: 0.1,
            min_cosine_similarity: 0.99,
            min_top1_agreement: 0.8,
            require_final_position_top1_match: true,
        }
    }
}

impl ParityThresholds {
    /// Validate threshold finiteness and ranges.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for negative error limits, non-finite
    /// values, or similarity/agreement limits outside `[0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if !self.max_abs_error.is_finite()
            || !self.mean_abs_error.is_finite()
            || !self.root_mean_square_error.is_finite()
            || self.max_abs_error < 0.0
            || self.mean_abs_error < 0.0
            || self.root_mean_square_error < 0.0
        {
            return Err(EngineError::invalid_input(
                "parity thresholds",
                "error limits must be finite and non-negative",
            ));
        }
        if !self.min_cosine_similarity.is_finite()
            || !(0.0..=1.0).contains(&self.min_cosine_similarity)
            || !self.min_top1_agreement.is_finite()
            || !(0.0..=1.0).contains(&self.min_top1_agreement)
        {
            return Err(EngineError::invalid_input(
                "parity thresholds",
                "minimum cosine similarity and top-1 agreement must be in [0, 1]",
            ));
        }
        Ok(())
    }

    fn violations(&self, metrics: &ParityMetrics) -> Vec<String> {
        let mut violations = Vec::new();
        if metrics.max_abs_error > self.max_abs_error {
            violations.push(format!(
                "max_abs_error {} exceeds {}",
                metrics.max_abs_error, self.max_abs_error
            ));
        }
        if metrics.mean_abs_error > self.mean_abs_error {
            violations.push(format!(
                "mean_abs_error {} exceeds {}",
                metrics.mean_abs_error, self.mean_abs_error
            ));
        }
        if metrics.root_mean_square_error > self.root_mean_square_error {
            violations.push(format!(
                "root_mean_square_error {} exceeds {}",
                metrics.root_mean_square_error, self.root_mean_square_error
            ));
        }
        if metrics.cosine_similarity < self.min_cosine_similarity {
            violations.push(format!(
                "cosine_similarity {} is below {}",
                metrics.cosine_similarity, self.min_cosine_similarity
            ));
        }
        if metrics.top1_agreement < self.min_top1_agreement {
            violations.push(format!(
                "top1_agreement {} is below {}",
                metrics.top1_agreement, self.min_top1_agreement
            ));
        }
        if self.require_final_position_top1_match && !metrics.final_position_top1_match {
            violations.push("final-position top-1 token does not match".to_owned());
        }
        violations
    }
}

/// Complete result of comparing Rust model logits with JAX reference logits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    /// Reference batch size.
    pub batch_size: usize,
    /// Reference sequence length.
    pub sequence_length: usize,
    /// Vocabulary width of each logit row.
    pub vocab_size: usize,
    /// Numerical and top-1 comparison metrics.
    pub metrics: ParityMetrics,
    /// Thresholds used to evaluate the metrics.
    pub thresholds: ParityThresholds,
    /// Whether every acceptance threshold passed.
    pub accepted: bool,
    /// Human-readable threshold violations.
    pub violations: Vec<String>,
}

/// Execute the model on exported token IDs and compare every output logit.
///
/// # Errors
///
/// Returns artifact, shape, model-execution, or metric-validation errors.
pub fn run_reference_parity(
    model: &NileMiniModel,
    inputs_path: impl AsRef<Path>,
    outputs_path: impl AsRef<Path>,
    thresholds: ParityThresholds,
) -> Result<ParityReport> {
    thresholds.validate()?;
    let inputs = ReferenceInputs::from_json_path(inputs_path)?;
    let expected = ReferenceLogits::from_safetensors_path(outputs_path)?;

    if inputs.batch_size() != expected.batch_size()
        || inputs.sequence_length() != expected.sequence_length()
    {
        return Err(EngineError::invalid_input(
            "reference parity",
            format!(
                "input shape [{}, {}] does not match output leading shape [{}, {}]",
                inputs.batch_size(),
                inputs.sequence_length(),
                expected.batch_size(),
                expected.sequence_length()
            ),
        ));
    }
    if expected.vocab_size() != model.config().vocab_size {
        return Err(EngineError::invalid_input(
            "reference parity",
            format!(
                "reference vocabulary {} does not match model vocabulary {}",
                expected.vocab_size(),
                model.config().vocab_size
            ),
        ));
    }

    let mut actual = Vec::with_capacity(expected.values().len());
    for sequence in inputs.sequences() {
        actual.extend(model.forward_prefill(sequence)?);
    }

    let metrics = ParityMetrics::compare(
        &actual,
        expected.values(),
        expected.batch_size(),
        expected.sequence_length(),
        expected.vocab_size(),
    )?;
    let violations = thresholds.violations(&metrics);

    Ok(ParityReport {
        batch_size: expected.batch_size(),
        sequence_length: expected.sequence_length(),
        vocab_size: expected.vocab_size(),
        metrics,
        thresholds,
        accepted: violations.is_empty(),
        violations,
    })
}

#[cfg(test)]
mod tests {
    use super::{ParityMetrics, ParityThresholds};

    #[test]
    fn default_thresholds_accept_identical_logits() {
        let metrics =
            ParityMetrics::compare(&[1.0, 2.0], &[1.0, 2.0], 1, 1, 2).expect("valid metrics");
        let thresholds = ParityThresholds::default();
        assert!(thresholds.violations(&metrics).is_empty());
    }
}
