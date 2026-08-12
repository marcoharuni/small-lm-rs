//! Report lazy paged-KV allocation as sequence length grows.

use std::env;
use std::mem::size_of;
use std::path::PathBuf;

use smalllm_engine::{KvCache, KvCacheConfig, ModelConfig};

const CHECKPOINTS: [usize; 7] = [0, 1, 16, 17, 32, 128, 512];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/small-lm-8m"));
    let model_config = ModelConfig::from_json_path(artifact_dir.join("config.json"))?;
    let cache_config = KvCacheConfig::from_model(&model_config, model_config.context_length)?;
    let width = cache_config
        .num_key_value_heads
        .checked_mul(cache_config.head_dimension)
        .ok_or("KV width overflows usize")?;
    let bytes_per_layer_token = width
        .checked_mul(2)
        .and_then(|value| value.checked_mul(size_of::<f32>()))
        .ok_or("KV bytes per layer token overflow usize")?;
    let dense_payload_bytes = cache_config
        .num_layers
        .checked_mul(cache_config.max_sequence_length)
        .and_then(|value| value.checked_mul(bytes_per_layer_token))
        .ok_or("dense KV payload size overflows usize")?;

    println!("layers={}", cache_config.num_layers);
    println!("max_sequence_length={}", cache_config.max_sequence_length);
    println!("kv_heads={}", cache_config.num_key_value_heads);
    println!("head_dimension={}", cache_config.head_dimension);
    println!("kv_width={width}");
    println!("dense_payload_bytes={dense_payload_bytes}");

    for target_length in CHECKPOINTS
        .into_iter()
        .filter(|length| *length <= cache_config.max_sequence_length)
    {
        let mut cache = KvCache::allocate(cache_config)?;
        if target_length > 0 {
            let elements = target_length
                .checked_mul(width)
                .ok_or("benchmark append size overflows usize")?;
            let keys = vec![0.0_f32; elements];
            let values = vec![0.0_f32; elements];
            for layer_index in 0..cache_config.num_layers {
                cache.append_layer(layer_index, &keys, &values, target_length)?;
            }
        }

        let allocated_payload_bytes = cache
            .allocated_token_capacity()
            .checked_mul(bytes_per_layer_token)
            .ok_or("allocated KV payload size overflows usize")?;
        let saved_percent = if dense_payload_bytes == 0 {
            0.0
        } else {
            100.0 * (1.0 - allocated_payload_bytes as f64 / dense_payload_bytes as f64)
        };

        println!(
            "tokens={target_length} pages={} allocated_token_capacity={} allocated_payload_bytes={} dense_payload_saved_percent={saved_percent:.3}",
            cache.allocated_pages(),
            cache.allocated_token_capacity(),
            allocated_payload_bytes,
        );
    }

    Ok(())
}
