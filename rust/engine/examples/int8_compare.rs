//! Compare FP32 and INT8 SmallLM execution on the same token sequence.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use smalllm_engine::{ModelConfig, QuantizedDecodeModel, SmallLMModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/small-lm-8m"));
    let config = ModelConfig::from_json_path(artifact_dir.join("config.json"))?;

    let mut fp32 = SmallLMModel::from_config(config.clone())?;
    fp32.load_weights(artifact_dir.join("model.safetensors"))?;
    let mut int8 = QuantizedDecodeModel::from_config(config.clone())?;
    int8.load_weights(artifact_dir.join("model.int8.safetensors"))?;

    let prompt = [1_u32, 17, 42, 101, 256, 511, 1024, 2048, 4096, 7];
    let capacity = prompt.len() + 1;
    let mut fp32_cache = fp32.allocate_kv_cache(capacity)?;
    let mut int8_cache = int8.allocate_kv_cache(capacity)?;

    let started = Instant::now();
    let fp32_logits = fp32.forward_prefill_with_cache(&prompt, &mut fp32_cache)?;
    let fp32_prefill = started.elapsed();

    let started = Instant::now();
    let int8_logits = int8.forward_prefill_with_cache(&prompt, &mut int8_cache)?;
    let int8_prefill = started.elapsed();

    let prefill_metrics = compare(&fp32_logits, &int8_logits)?;
    print_metrics("prefill", prefill_metrics);
    println!(
        "fp32_prefill_ms={:.3}",
        fp32_prefill.as_secs_f64() * 1_000.0
    );
    println!(
        "int8_prefill_ms={:.3}",
        int8_prefill.as_secs_f64() * 1_000.0
    );

    let next_token = argmax(final_row(&fp32_logits, config.vocab_size)?) as u32;
    let started = Instant::now();
    let fp32_decode = fp32.forward_cached_token(next_token, &mut fp32_cache)?;
    let fp32_decode_time = started.elapsed();

    let started = Instant::now();
    let int8_decode = int8.forward_cached_token(next_token, &mut int8_cache)?;
    let int8_decode_time = started.elapsed();

    let decode_metrics = compare(&fp32_decode, &int8_decode)?;
    print_metrics("decode", decode_metrics);
    println!(
        "fp32_decode_ms={:.3}",
        fp32_decode_time.as_secs_f64() * 1_000.0
    );
    println!(
        "int8_decode_ms={:.3}",
        int8_decode_time.as_secs_f64() * 1_000.0
    );
    println!(
        "decode_top1_match={}",
        argmax(&fp32_decode) == argmax(&int8_decode)
    );

    Ok(())
}

#[derive(Clone, Copy)]
struct Metrics {
    count: usize,
    max_abs: f32,
    mean_abs: f64,
    rmse: f64,
    cosine: f64,
}

fn compare(reference: &[f32], candidate: &[f32]) -> Result<Metrics, &'static str> {
    if reference.is_empty() || reference.len() != candidate.len() {
        return Err("comparison tensors must have the same non-zero length");
    }

    let mut max_abs = 0.0_f32;
    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut reference_norm = 0.0_f64;
    let mut candidate_norm = 0.0_f64;

    for (&reference_value, &candidate_value) in reference.iter().zip(candidate) {
        let difference = f64::from(reference_value) - f64::from(candidate_value);
        max_abs = max_abs.max(difference.abs() as f32);
        absolute_sum += difference.abs();
        squared_sum += difference * difference;
        dot += f64::from(reference_value) * f64::from(candidate_value);
        reference_norm += f64::from(reference_value).powi(2);
        candidate_norm += f64::from(candidate_value).powi(2);
    }

    let count = reference.len();
    let denominator = (reference_norm * candidate_norm).sqrt();
    let cosine = if denominator > 0.0 {
        dot / denominator
    } else {
        1.0
    };
    Ok(Metrics {
        count,
        max_abs,
        mean_abs: absolute_sum / count as f64,
        rmse: (squared_sum / count as f64).sqrt(),
        cosine,
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

fn print_metrics(label: &str, metrics: Metrics) {
    println!("{label}_values={}", metrics.count);
    println!("{label}_max_abs={:.9}", metrics.max_abs);
    println!("{label}_mean_abs={:.9}", metrics.mean_abs);
    println!("{label}_rmse={:.9}", metrics.rmse);
    println!("{label}_cosine={:.9}", metrics.cosine);
}
