//! Real-artifact JAX-to-Rust parity integration test.

use std::path::PathBuf;

use nilemini_engine::parity::{run_reference_parity, ParityThresholds};
use nilemini_engine::{ModelConfig, NileMiniModel};

#[test]
#[ignore = "requires the local 30.52 MiB model and a release CPU run"]
fn exported_smoke_logits_match_jax_reference() {
    let artifact =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/nilemini-8m-situ");
    let model_path = artifact.join("model.safetensors");
    if !model_path.is_file() {
        return;
    }

    let config = ModelConfig::from_json_path(artifact.join("config.json")).expect("valid config");
    let mut model = NileMiniModel::from_config(config).expect("valid model");
    model.load_weights(model_path).expect("valid model weights");
    let report = run_reference_parity(
        &model,
        artifact.join("reference_inputs.json"),
        artifact.join("reference_outputs.safetensors"),
        ParityThresholds::default(),
    )
    .expect("parity execution must succeed");

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serializable parity report")
    );
    assert!(
        report.accepted,
        "parity thresholds failed: {:?}",
        report.violations
    );
    assert!(report.metrics.final_position_top1_match);
}
