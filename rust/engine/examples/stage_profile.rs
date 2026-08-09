//! Opt-in stage profiler for fresh-sequence CPU prefill.
//!
//! This utility times the major public model stages separately so performance
//! work can target measured bottlenecks without adding instrumentation to the
//! normal inference path.

use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use smalllm_engine::parity::ReferenceInputs;
use smalllm_engine::{ModelConfig, SmallLMModel};

#[derive(Serialize)]
struct StageProfile {
    model: String,
    rayon_threads: usize,
    prompt_tokens: usize,
    iterations: usize,
    embedding_milliseconds: f64,
    transformer_milliseconds: f64,
    final_norm_milliseconds: f64,
    lm_head_milliseconds: f64,
    total_milliseconds: f64,
    embedding_share: f64,
    transformer_share: f64,
    final_norm_share: f64,
    lm_head_share: f64,
}

fn parse_usize(index: usize, default: usize) -> Result<usize, Box<dyn Error>> {
    match std::env::args().nth(index) {
        Some(value) => Ok(value.parse::<usize>()?),
        None => Ok(default),
    }
}

fn milliseconds(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1_000.0 / iterations as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let artifact = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/small-lm-8m"));
    let prompt_length = parse_usize(2, 32)?;
    let iterations = parse_usize(3, 3)?;

    if prompt_length == 0 || iterations == 0 {
        return Err("prompt length and iteration count must both be positive".into());
    }

    let config = ModelConfig::from_json_path(artifact.join("config.json"))?;
    if prompt_length > config.context_length {
        return Err(format!(
            "prompt length {prompt_length} exceeds context length {}",
            config.context_length
        )
        .into());
    }

    let inputs = ReferenceInputs::from_json_path(artifact.join("reference_inputs.json"))?;
    let seed_tokens = inputs
        .sequences()
        .first()
        .ok_or("reference input is empty")?;
    if seed_tokens.is_empty() {
        return Err("reference input contains no tokens".into());
    }
    let prompt: Vec<u32> = seed_tokens
        .iter()
        .copied()
        .cycle()
        .take(prompt_length)
        .collect();

    let mut model = SmallLMModel::from_config(config)?;
    model.load_weights(artifact.join("model.safetensors"))?;

    // Warm every stage once so one-time Rayon and code-path initialization is
    // not charged to the measurements below.
    let embeddings = model.embed_tokens(&prompt)?;
    let transformed = model.run_transformer_blocks(&embeddings, prompt_length)?;
    let normalized = model.apply_final_norm(&transformed, prompt_length)?;
    let logits = model.project_tied_logits(&normalized, prompt_length)?;
    std::hint::black_box(logits);

    let mut embedding_total = Duration::ZERO;
    let mut transformer_total = Duration::ZERO;
    let mut final_norm_total = Duration::ZERO;
    let mut lm_head_total = Duration::ZERO;

    for _ in 0..iterations {
        let started = Instant::now();
        let embeddings = model.embed_tokens(&prompt)?;
        embedding_total += started.elapsed();

        let started = Instant::now();
        let transformed = model.run_transformer_blocks(&embeddings, prompt_length)?;
        transformer_total += started.elapsed();

        let started = Instant::now();
        let normalized = model.apply_final_norm(&transformed, prompt_length)?;
        final_norm_total += started.elapsed();

        let started = Instant::now();
        let logits = model.project_tied_logits(&normalized, prompt_length)?;
        lm_head_total += started.elapsed();
        std::hint::black_box(logits);
    }

    let embedding_ms = milliseconds(embedding_total, iterations);
    let transformer_ms = milliseconds(transformer_total, iterations);
    let final_norm_ms = milliseconds(final_norm_total, iterations);
    let lm_head_ms = milliseconds(lm_head_total, iterations);
    let total_ms = embedding_ms + transformer_ms + final_norm_ms + lm_head_ms;

    let report = StageProfile {
        model: model.config().model_name.clone(),
        rayon_threads: rayon::current_num_threads(),
        prompt_tokens: prompt_length,
        iterations,
        embedding_milliseconds: embedding_ms,
        transformer_milliseconds: transformer_ms,
        final_norm_milliseconds: final_norm_ms,
        lm_head_milliseconds: lm_head_ms,
        total_milliseconds: total_ms,
        embedding_share: embedding_ms / total_ms,
        transformer_share: transformer_ms / total_ms,
        final_norm_share: final_norm_ms / total_ms,
        lm_head_share: lm_head_ms / total_ms,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
