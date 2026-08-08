//! JAX-to-Rust reference-logit parity validation.

mod metrics;
mod reference;
mod runner;

pub use metrics::ParityMetrics;
pub use reference::{ReferenceInputs, ReferenceLogits};
pub use runner::{run_reference_parity, ParityReport, ParityThresholds};
