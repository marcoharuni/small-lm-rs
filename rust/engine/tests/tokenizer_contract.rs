//! Artifact-level tokenizer contract tests.

use std::path::PathBuf;

use smalllm_engine::chat::{format_chat_prompt, ChatMessage};
use smalllm_engine::config::ModelConfig;
use smalllm_engine::generation_config::GenerationConfig;
use smalllm_engine::tokenizer::{SmallLMTokenizer, SpecialTokenIds};

fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/small-lm-8m")
}

#[test]
fn exported_tokenizer_has_the_frozen_reserved_identifiers() {
    let artifact = artifact_dir();
    let model =
        ModelConfig::from_json_path(artifact.join("config.json")).expect("valid model config");
    let generation =
        GenerationConfig::from_json_path(artifact.join("generation_config.json"), &model)
            .expect("valid generation config");
    let tokenizer =
        SmallLMTokenizer::from_file(artifact.join("tokenizer.json")).expect("valid tokenizer");

    let ids = tokenizer
        .validate_contract(&model, &generation)
        .expect("valid tokenizer contract");

    assert_eq!(
        ids,
        SpecialTokenIds {
            pad: 0,
            bos: 1,
            eos: 2,
            system: 3,
            user: 4,
            assistant: 5,
        }
    );
}

#[test]
fn exported_reference_prompt_encodes_exactly() {
    let tokenizer = SmallLMTokenizer::from_file(artifact_dir().join("tokenizer.json"))
        .expect("valid tokenizer");
    let prompt = format_chat_prompt(&[ChatMessage::user("Hello")]).expect("valid chat prompt");
    let token_ids = tokenizer
        .encode(&prompt, false)
        .expect("reference prompt encoding");

    assert_eq!(token_ids, [1, 4, 204, 45, 474, 84, 2, 204, 5, 204]);
}
