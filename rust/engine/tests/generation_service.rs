//! End-to-end checks for the reusable generation service.

use std::path::PathBuf;

use smalllm_engine::chat::ChatMessage;
use smalllm_engine::sampler::SamplingConfig;
use smalllm_engine::service::GenerationService;

fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/small-lm-8m")
}

#[test]
#[ignore = "requires the local 30.52 MiB model and a release CPU generation run"]
fn exported_service_generates_the_reference_next_token() {
    let service =
        GenerationService::from_artifact_dir(artifact_dir()).expect("valid exported service");
    let output = service
        .generate_chat(
            &[ChatMessage::user("Hello")],
            1,
            SamplingConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
            },
        )
        .expect("cached chat generation");

    assert_eq!(output.prompt_token_count, 10);
    assert_eq!(output.generated_token_ids, [2451]);
    assert_eq!(output.finish_reason, "length");
}
