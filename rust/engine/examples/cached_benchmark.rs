//! Release benchmark for NileMini prompt prefill and KV-cached decoding.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nilemini_engine::parity::ReferenceInputs;
use nilemini_engine::{ModelConfig, NileMiniModel};
use serde::Serialize;

#[derive(Serialize)]
struct BenchmarkReport {
    model: String,
    rayon_threads: usize,
    prompt_tokens: usize,
    decode_tokens: usize,
    model_load_milliseconds: f64,
    prefill_milliseconds: f64,
    prefill_tokens_per_second: f64,
    time_to_first_token_milliseconds: f64,
    first_token_id: u32,
    decode_milliseconds: f64,
    decode_milliseconds_per_token: f64,
    decode_tokens_per_second: f64,
    inter_token_latency_milliseconds: f64,
    peak_rss_mebibytes: Option<f64>,
    final_cache_length: usize,
}

fn parse_usize(index: usize, default: usize) -> Result<usize, Box<dyn Error>> {
    match std::env::args().nth(index) {
        Some(value) => Ok(value.parse::<usize>()?),
        None => Ok(default),
    }
}

fn greedy_token(logits: &[f32]) -> Result<u32, Box<dyn Error>> {
    let (index, _) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or("cannot select a token from empty logits")?;
    Ok(u32::try_from(index)?)
}

fn peak_rss_mebibytes() -> Option<f64> {
    // Linux exposes the process high-water resident set in /proc/self/status.
    // Other platforms simply report null rather than inventing a value.
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<f64>().ok()?;
    Some(kib / 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let artifact = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/nilemini-8m-situ"));
    let prompt_length = parse_usize(2, 32)?;
    let decode_tokens = parse_usize(3, 32)?;

    let config = ModelConfig::from_json_path(artifact.join("config.json"))?;
    if prompt_length == 0 || decode_tokens == 0 {
        return Err("prompt and decode token counts must both be positive".into());
    }
    if prompt_length + decode_tokens > config.context_length {
        return Err(format!(
            "requested {} total tokens exceeds context length {}",
            prompt_length + decode_tokens,
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
    let decode_input: Vec<u32> = seed_tokens
        .iter()
        .copied()
        .cycle()
        .skip(prompt_length % seed_tokens.len())
        .take(decode_tokens)
        .collect();

    let model_load_started = Instant::now();
    let mut model = NileMiniModel::from_config(config)?;
    model.load_weights(artifact.join("model.safetensors"))?;
    let model_load_elapsed = model_load_started.elapsed();

    let mut cache = model.allocate_kv_cache(model.config().context_length)?;

    // Warm a tiny projection path before timing the declared workload so Rayon
    // thread-pool initialization is not charged to prefill or TTFT.
    let mut warm_cache = model.allocate_kv_cache(2)?;
    let warm_logits = model.forward_prefill_with_cache(&prompt[..1], &mut warm_cache)?;
    std::hint::black_box(warm_logits);

    let prefill_started = Instant::now();
    let prefill_logits = model.forward_prefill_with_cache(&prompt, &mut cache)?;
    let prefill_elapsed = prefill_started.elapsed();

    let vocab_size = model.config().vocab_size;
    let final_logits = &prefill_logits[prefill_logits.len() - vocab_size..];
    let first_token = greedy_token(final_logits)?;
    let ttft_elapsed = prefill_started.elapsed();
    std::hint::black_box(first_token);
    std::hint::black_box(prefill_logits);

    let decode_started = Instant::now();
    for token in decode_input {
        let logits = model.forward_cached_token(token, &mut cache)?;
        std::hint::black_box(logits);
    }
    let decode_elapsed = decode_started.elapsed();

    let prefill_seconds = prefill_elapsed.as_secs_f64();
    let decode_seconds = decode_elapsed.as_secs_f64();
    let decode_ms_per_token = decode_seconds * 1_000.0 / decode_tokens as f64;

    let report = BenchmarkReport {
        model: model.config().model_name.clone(),
        rayon_threads: rayon::current_num_threads(),
        prompt_tokens: prompt_length,
        decode_tokens,
        model_load_milliseconds: model_load_elapsed.as_secs_f64() * 1_000.0,
        prefill_milliseconds: prefill_seconds * 1_000.0,
        prefill_tokens_per_second: prompt_length as f64 / prefill_seconds,
        time_to_first_token_milliseconds: ttft_elapsed.as_secs_f64() * 1_000.0,
        first_token_id: first_token,
        decode_milliseconds: decode_seconds * 1_000.0,
        decode_milliseconds_per_token: decode_ms_per_token,
        decode_tokens_per_second: decode_tokens as f64 / decode_seconds,
        inter_token_latency_milliseconds: decode_ms_per_token,
        peak_rss_mebibytes: peak_rss_mebibytes(),
        final_cache_length: cache.sequence_length(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
