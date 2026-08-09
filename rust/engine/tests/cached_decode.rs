//! Real-artifact cached-token versus uncached-logit parity test.

use std::path::PathBuf;

use smalllm_engine::parity::ReferenceInputs;
use smalllm_engine::{ModelConfig, SmallLMModel};

#[test]
#[ignore = "requires the local 30.52 MiB model and a release CPU run"]
fn exported_cached_token_matches_uncached_final_logits() {
    let artifact = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/small-lm-8m");
    let model_path = artifact.join("model.safetensors");
    if !model_path.is_file() {
        return;
    }

    let config = ModelConfig::from_json_path(artifact.join("config.json")).expect("valid config");
    let vocab_size = config.vocab_size;
    let inputs = ReferenceInputs::from_json_path(artifact.join("reference_inputs.json"))
        .expect("valid inputs");
    let sequence = inputs.sequences().first().expect("one sequence");
    assert!(sequence.len() >= 2);
    let split = sequence.len() - 1;

    let mut model = SmallLMModel::from_config(config).expect("valid model");
    model.load_weights(model_path).expect("valid weights");
    let fresh = model.forward_prefill(sequence).expect("fresh logits");
    let mut cache = model
        .allocate_kv_cache(sequence.len())
        .expect("allocated cache");
    model
        .forward_prefill_with_cache(&sequence[..split], &mut cache)
        .expect("cached prompt prefill");
    let cached = model
        .forward_cached_token(sequence[split], &mut cache)
        .expect("cached token");
    let expected = &fresh[fresh.len() - vocab_size..];

    let mut max_abs_error = 0.0_f32;
    let mut sum_abs_error = 0.0_f64;
    for (&actual, &reference) in cached.iter().zip(expected) {
        let error = (actual - reference).abs();
        max_abs_error = max_abs_error.max(error);
        sum_abs_error += f64::from(error);
    }
    let mean_abs_error = sum_abs_error / cached.len() as f64;
    eprintln!("cached decode parity: max_abs={max_abs_error:.9}, mean_abs={mean_abs_error:.9}");
    assert!(
        max_abs_error <= 0.02,
        "maximum absolute error is {max_abs_error}"
    );
    assert!(
        mean_abs_error <= 0.003,
        "mean absolute error is {mean_abs_error}"
    );
    assert_eq!(cache.sequence_length(), sequence.len());
    assert!(cache.is_synchronized());
}
