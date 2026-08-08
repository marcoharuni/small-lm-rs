//! Integration tests for exported generation metadata.

use std::path::PathBuf;

use nilemini_engine::config::ModelConfig;
use nilemini_engine::generation_config::GenerationConfig;

fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/nilemini-8m-situ")
}

#[test]
fn exported_generation_configuration_matches_the_model() {
    let artifact = artifact_dir();
    let model =
        ModelConfig::from_json_path(artifact.join("config.json")).expect("valid model config");
    let generation =
        GenerationConfig::from_json_path(artifact.join("generation_config.json"), &model)
            .expect("valid generation config");

    assert_eq!(generation.context_length, 512);
    assert_eq!(generation.pad_token_id, 0);
    assert_eq!(generation.eos_token_id, 2);
}
