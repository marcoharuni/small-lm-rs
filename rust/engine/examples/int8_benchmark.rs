//! Repeatable FP32/INT8 benchmark for one SmallLM prompt and cached decode step.

use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use smalllm_engine::{DecodeBackend, ModelConfig, QuantizedDecodeModel, SmallLMModel};

const PROMPT: [u32; 10] = [1, 17, 42, 101, 256, 511, 1024, 2048, 4096, 7];
const DEFAULT_ITERATIONS: usize = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = env::args().nth(1).unwrap_or_else(|| "fp32".to_owned());
    let artifact_dir = env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/small-lm-8m"));
    let iterations = env::args()
        .nth(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_ITERATIONS);
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }

    let config = ModelConfig::from_json_path(artifact_dir.join("config.json"))?;
    match mode.as_str() {
        "fp32" => {
            let mut model = SmallLMModel::from_config(config)?;
            model.load_weights(artifact_dir.join("model.safetensors"))?;
            benchmark("fp32", &model, iterations)?;
        }
        "int8" => {
            let mut model = QuantizedDecodeModel::from_config(config)?;
            model.load_weights(artifact_dir.join("model.int8.safetensors"))?;
            benchmark("int8", &model, iterations)?;
        }
        _ => return Err("mode must be either 'fp32' or 'int8'".into()),
    }
    Ok(())
}

fn benchmark<B: DecodeBackend>(
    label: &str,
    model: &B,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Warm up once so allocator/thread-pool startup does not dominate averages.
    let _ = run_once(model)?;

    let mut prefill_total = Duration::ZERO;
    let mut decode_total = Duration::ZERO;
    let mut top1_checksum = 0_u64;

    for _ in 0..iterations {
        let measurement = run_once(model)?;
        prefill_total += measurement.prefill;
        decode_total += measurement.decode;
        top1_checksum = top1_checksum.wrapping_add(measurement.decode_top1 as u64);
    }

    let divisor = iterations as f64;
    println!("mode={label}");
    println!("iterations={iterations}");
    println!(
        "mean_prefill_ms={:.3}",
        prefill_total.as_secs_f64() * 1_000.0 / divisor
    );
    println!(
        "mean_decode_ms={:.3}",
        decode_total.as_secs_f64() * 1_000.0 / divisor
    );
    println!("decode_top1_checksum={top1_checksum}");
    Ok(())
}

struct Measurement {
    prefill: Duration,
    decode: Duration,
    decode_top1: usize,
}

fn run_once<B: DecodeBackend>(model: &B) -> Result<Measurement, Box<dyn std::error::Error>> {
    let capacity = PROMPT.len() + 1;
    let mut cache = model.allocate_kv_cache(capacity)?;

    let started = Instant::now();
    let logits = model.forward_prefill_with_cache(&PROMPT, &mut cache)?;
    let prefill = started.elapsed();

    let next_token = argmax(final_row(&logits, model.config().vocab_size)?) as u32;
    let started = Instant::now();
    let decode_logits = model.forward_cached_token(next_token, &mut cache)?;
    let decode = started.elapsed();

    Ok(Measurement {
        prefill,
        decode,
        decode_top1: argmax(&decode_logits),
    })
}

fn final_row(logits: &[f32], vocab_size: usize) -> Result<&[f32], &'static str> {
    if vocab_size == 0 || logits.len() < vocab_size || logits.len() % vocab_size != 0 {
        return Err("invalid flattened logit tensor");
    }
    Ok(&logits[logits.len() - vocab_size..])
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or_default()
}

#[allow(dead_code)]
fn _assert_path(_: &Path) {}
