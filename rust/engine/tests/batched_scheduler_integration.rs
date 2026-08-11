//! End-to-end equivalence checks for the batched scheduler backend.

use std::collections::HashMap;
use std::path::PathBuf;

use smalllm_engine::generation::generate_token_ids;
use smalllm_engine::generation_config::GenerationConfig;
use smalllm_engine::sampler::SamplingConfig;
use smalllm_engine::scheduler::{GenerationScheduler, SchedulerConfig, SequenceId};
use smalllm_engine::tokenizer::SmallLMTokenizer;
use smalllm_engine::{BatchedDecodeModel, ModelConfig, SmallLMModel};

fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/small-lm-8m")
}

fn greedy() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    }
}

#[test]
fn batched_scheduler_matches_independent_generation() {
    let artifact = artifact_dir();
    let config =
        ModelConfig::from_json_path(artifact.join("config.json")).expect("valid model config");
    let generation_config =
        GenerationConfig::from_json_path(artifact.join("generation_config.json"), &config)
            .expect("valid generation config");
    let tokenizer =
        SmallLMTokenizer::from_file(artifact.join("tokenizer.json")).expect("valid tokenizer");

    let mut reference_model = SmallLMModel::from_config(config.clone()).expect("reference model");
    reference_model
        .load_weights(artifact.join("model.safetensors"))
        .expect("reference weights");
    let mut batched_model = BatchedDecodeModel::from_config(config).expect("batched model");
    batched_model
        .load_weights(artifact.join("model.safetensors"))
        .expect("batched weights");

    let prompts = ["Hello", "Rust", "Tanzania"];
    let prompt_tokens = prompts
        .iter()
        .map(|prompt| tokenizer.encode(prompt, true).expect("prompt encoding"))
        .collect::<Vec<_>>();
    let max_new_tokens = 4;

    let expected = prompt_tokens
        .iter()
        .map(|tokens| {
            generate_token_ids(
                &reference_model,
                tokens,
                max_new_tokens,
                Some(generation_config.eos_token_id),
                greedy(),
            )
            .expect("independent generation")
        })
        .collect::<Vec<_>>();

    let mut scheduler = GenerationScheduler::new(SchedulerConfig {
        max_active_sequences: prompts.len(),
    })
    .expect("valid scheduler");
    let sequence_ids = prompt_tokens
        .iter()
        .map(|tokens| {
            scheduler
                .admit(
                    &batched_model,
                    tokens,
                    max_new_tokens,
                    Some(generation_config.eos_token_id),
                    greedy(),
                )
                .expect("batched admission")
        })
        .collect::<Vec<_>>();

    let mut actual = HashMap::<SequenceId, Vec<u32>>::new();
    let mut finish_reasons = HashMap::<SequenceId, String>::new();
    while scheduler.active_sequence_count() > 0 {
        for event in scheduler
            .step(&batched_model)
            .expect("batched scheduler step")
        {
            actual
                .entry(event.sequence_id)
                .or_default()
                .push(event.token_id);
            if let Some(reason) = event.finish_reason {
                finish_reasons.insert(event.sequence_id, reason.as_str().to_owned());
            }
        }
    }

    for ((sequence_id, expected_output), prompt) in
        sequence_ids.iter().zip(expected.iter()).zip(prompts.iter())
    {
        assert_eq!(
            actual.get(sequence_id).map(Vec::as_slice),
            Some(expected_output.generated_token_ids.as_slice()),
            "batched tokens differ for prompt {prompt:?}"
        );
        assert_eq!(
            finish_reasons.get(sequence_id).map(String::as_str),
            Some(expected_output.finish_reason.as_str()),
            "batched finish reason differs for prompt {prompt:?}"
        );
    }
}
