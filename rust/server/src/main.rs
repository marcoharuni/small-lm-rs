//! Local language-model HTTP server.

#![forbid(unsafe_code)]

mod api;
mod batch_worker;
mod errors;
mod schema;
mod streaming;
mod worker;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use clap::Parser;
use errors::ServerError;
use tokio::net::TcpListener;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use worker::InferenceWorker;

#[derive(Clone, Debug, Parser)]
#[command(
    name = "small-lm-server",
    about = "Serve a local language model through an OpenAI-compatible HTTP API"
)]
struct Args {
    /// Directory containing config, tokenizer, generation metadata, and weights.
    #[arg(long, default_value = "artifacts/small-lm-8m")]
    model_dir: PathBuf,

    /// Maximum number of generation sequences kept active by the batch scheduler.
    #[arg(long, default_value_t = 8)]
    max_active_sequences: usize,

    /// Interface on which the HTTP server listens.
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,

    /// TCP port on which the HTTP server listens.
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), ServerError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    info!(
        model_dir = %args.model_dir.display(),
        max_active_sequences = args.max_active_sequences,
        "loading model artifacts"
    );
    let worker = InferenceWorker::from_artifact_dir(&args.model_dir, args.max_active_sequences)
        .map_err(|source| ServerError::Engine { source })?;

    let address = SocketAddr::new(args.host, args.port);
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;

    info!(
        %address,
        model = worker.model_name(),
        max_active_sequences = args.max_active_sequences,
        "SmallLM continuous-batching server ready"
    );
    axum::serve(listener, api::router(worker))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(ServerError::Serve)
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        warn!(%error, "failed to install shutdown signal handler");
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::path::PathBuf;

    use clap::Parser;

    use super::Args;

    #[test]
    fn command_line_defaults_are_local_and_predictable() {
        let args = Args::try_parse_from(["small-lm-server"]).expect("default arguments parse");

        assert_eq!(args.model_dir, PathBuf::from("artifacts/small-lm-8m"));
        assert_eq!(args.max_active_sequences, 8);
        assert_eq!(args.host, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(args.port, 8080);
    }

    #[test]
    fn batch_capacity_is_configurable() {
        let args = Args::try_parse_from([
            "small-lm-server",
            "--max-active-sequences",
            "4",
        ])
        .expect("batch capacity parses");

        assert_eq!(args.max_active_sequences, 4);
    }
}
