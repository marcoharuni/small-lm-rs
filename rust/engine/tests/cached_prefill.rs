//! Real-artifact parity test for cache-populating prompt prefill.

use std::path::PathBuf;

use smalllm_engine::parity::ReferenceInputs;
use smalllm_engine::{ModelConfig, SmallLMModel};

#[test]
#[ignore = "requires the local 30.52 MiB model and two release CPU prefill runs"]
fn exported_cached_prompt_prefill_matches_uncached_logits() {
    let artifact = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/small-lm-8m");
    let model_path = artifact.join("model.safetensors");
    if !model_path.is_file() {
        return;
    }

    let config = ModelConfig::from_json_path(artifact.join("config.json")).expect("valid config");
    let inputs = ReferenceInputs::from_json_path(artifact.join("reference_inputs.json"))
        .expect("valid reference inputs");
    let token_ids = inputs
        .sequences()
        .first()
        .expect("one exported reference sequence");
    let mut model = SmallLMModel::from_config(config).expect("valid model");
    model.load_weights(model_path).expect("valid weights");

    let expected = model.forward_prefill(token_ids).expect("uncached prefill");
    let mut cache = model
        .allocate_kv_cache(model.config().context_length)
        .expect("allocated cache");
    let actual = model
        .forward_prefill_with_cache(token_ids, &mut cache)
        .expect("cached prompt prefill");

    assert_eq!(actual, expected);
    assert_eq!(cache.sequence_length(), token_ids.len());
    assert!(cache.is_synchronized());
}
