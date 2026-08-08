//! Release command for measuring JAX-to-Rust logit parity.

use std::error::Error;
use std::io::{self, Write};
use std::path::PathBuf;

use nilemini_engine::parity::{run_reference_parity, ParityThresholds};
use nilemini_engine::{ModelConfig, NileMiniModel};

fn main() -> Result<(), Box<dyn Error>> {
    let artifact = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/nilemini-8m-situ"));

    let config = ModelConfig::from_json_path(artifact.join("config.json"))?;
    let mut model = NileMiniModel::from_config(config)?;
    model.load_weights(artifact.join("model.safetensors"))?;

    let report = run_reference_parity(
        &model,
        artifact.join("reference_inputs.json"),
        artifact.join("reference_outputs.safetensors"),
        ParityThresholds::default(),
    )?;

    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &report)?;
    writeln!(output)?;
    Ok(())
}
