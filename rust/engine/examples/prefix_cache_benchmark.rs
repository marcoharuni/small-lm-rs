//! Measure cold prompt prefill against exact and partial prefix-cache reuse.

use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use smalllm_engine::sampler::SamplingConfig;
use smalllm_engine::{GenerationSession, ModelConfig, PrefixCache, SmallLMModel};

const DEFAULT_ITERATIONS: usize = 10;
const BASE_PROMPT_TOKENS: usize = 64;
const EXTENSION_TOKENS: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/small-lm-8m"));
    let iterations = env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_ITERATIONS);
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }

    let config = ModelConfig::from_json_path(artifact_dir.join("config.json"))?;
    let base_prompt = prompt_tokens(BASE_PROMPT_TOKENS, config.vocab_size);
    let extended_prompt = prompt_tokens(BASE_PROMPT_TOKENS + EXTENSION_TOKENS, config.vocab_size);

    let mut model = SmallLMModel::from_config(config)?;
    model.load_weights(artifact_dir.join("model.safetensors"))?;

    // Warm allocator/Rayon/model paths before collecting timings.
    let mut warm_cache = PrefixCache::default();
    let _ = prefill(&model, &base_prompt, &mut warm_cache)?;

    let mut cold_base_total = Duration::ZERO;
    let mut exact_hit_total = Duration::ZERO;
    let mut cold_extended_total = Duration::ZERO;
    let mut partial_hit_total = Duration::ZERO;

    for _ in 0..iterations {
        let mut cache = PrefixCache::default();

        let started = Instant::now();
        let cold_base_token = prefill(&model, &base_prompt, &mut cache)?;
        cold_base_total += started.elapsed();

        let started = Instant::now();
        let exact_hit_token = prefill(&model, &base_prompt, &mut cache)?;
        exact_hit_total += started.elapsed();
        if exact_hit_token != cold_base_token {
            return Err("exact prefix hit changed the sampled token".into());
        }

        let started = Instant::now();
        let partial_hit_token = prefill(&model, &extended_prompt, &mut cache)?;
        partial_hit_total += started.elapsed();

        let mut cold_cache = PrefixCache::default();
        let started = Instant::now();
        let cold_extended_token = prefill(&model, &extended_prompt, &mut cold_cache)?;
        cold_extended_total += started.elapsed();
        if partial_hit_token != cold_extended_token {
            return Err("partial prefix hit changed the sampled token".into());
        }
    }

    let divisor = iterations as f64;
    let cold_base_ms = milliseconds(cold_base_total, divisor);
    let exact_hit_ms = milliseconds(exact_hit_total, divisor);
    let cold_extended_ms = milliseconds(cold_extended_total, divisor);
    let partial_hit_ms = milliseconds(partial_hit_total, divisor);

    println!("iterations={iterations}");
    println!("base_prompt_tokens={}", base_prompt.len());
    println!("extended_prompt_tokens={}", extended_prompt.len());
    println!("reused_partial_tokens={}", base_prompt.len());
    println!("mean_cold_base_ms={cold_base_ms:.3}");
    println!("mean_exact_hit_ms={exact_hit_ms:.3}");
    println!("exact_hit_speedup_x={:.2}", cold_base_ms / exact_hit_ms);
    println!("mean_cold_extended_ms={cold_extended_ms:.3}");
    println!("mean_partial_hit_ms={partial_hit_ms:.3}");
    println!(
        "partial_hit_speedup_x={:.2}",
        cold_extended_ms / partial_hit_ms
    );
    println!("exact_token_match=true");
    println!("partial_token_match=true");

    Ok(())
}

fn prefill(
    model: &SmallLMModel,
    prompt: &[u32],
    cache: &mut PrefixCache,
) -> Result<u32, Box<dyn std::error::Error>> {
    let session = GenerationSession::prefill_with_prefix_cache(
        model,
        prompt,
        1,
        None,
        SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        },
        cache,
    )?;
    session
        .generated_token_ids()
        .first()
        .copied()
        .ok_or_else(|| "prefill produced no token".into())
}

fn prompt_tokens(length: usize, vocab_size: usize) -> Vec<u32> {
    (0..length)
        .map(|index| ((index * 97 + 17) % vocab_size) as u32)
        .collect()
}

fn milliseconds(duration: Duration, divisor: f64) -> f64 {
    duration.as_secs_f64() * 1_000.0 / divisor
}
