//! Release benchmark for NileMini prompt prefill and cached token decoding.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use nilemini_engine::parity::ReferenceInputs;
use nilemini_engine::{ModelConfig, NileMiniModel};
use serde::Serialize;

#[derive(Serialize)]
struct BenchmarkReport {
    prompt_tokens: usize,
    decode_tokens: usize,
    prefill_milliseconds: f64,
    decode_milliseconds: f64,
    decode_milliseconds_per_token: f64,
    final_cache_length: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let artifact = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/nilemini-8m-situ"));
    let config = ModelConfig::from_json_path(artifact.join("config.json"))?;
    let inputs = ReferenceInputs::from_json_path(artifact.join("reference_inputs.json"))?;
    let sequence = inputs
        .sequences()
        .first()
        .ok_or("reference input is empty")?;
    if sequence.len() < 2 {
        return Err("benchmark requires at least two reference tokens".into());
    }
    let prompt_length = sequence.len() - 1;
    let mut model = NileMiniModel::from_config(config)?;
    model.load_weights(artifact.join("model.safetensors"))?;
    let mut cache = model.allocate_kv_cache(model.config().context_length)?;

    let prefill_started = Instant::now();
    let prefill_logits =
        model.forward_prefill_with_cache(&sequence[..prompt_length], &mut cache)?;
    let prefill_elapsed = prefill_started.elapsed();
    std::hint::black_box(prefill_logits);

    let decode_started = Instant::now();
    let decode_logits = model.forward_cached_token(sequence[prompt_length], &mut cache)?;
    let decode_elapsed = decode_started.elapsed();
    std::hint::black_box(decode_logits);

    let decode_tokens = 1_usize;
    let report = BenchmarkReport {
        prompt_tokens: prompt_length,
        decode_tokens,
        prefill_milliseconds: prefill_elapsed.as_secs_f64() * 1_000.0,
        decode_milliseconds: decode_elapsed.as_secs_f64() * 1_000.0,
        decode_milliseconds_per_token: decode_elapsed.as_secs_f64() * 1_000.0
            / decode_tokens as f64,
        final_cache_length: cache.sequence_length(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
