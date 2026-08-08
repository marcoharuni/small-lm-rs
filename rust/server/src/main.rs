//! Local language-model HTTP server.

#![forbid(unsafe_code)]

mod api;
mod errors;
mod schema;
mod streaming;
mod worker;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use clap::Parser;
use errors::ServerError;
use nilemini_engine::GenerationService;
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
    #[arg(long, default_value = "artifacts/nilemini-8m-situ")]
    model_dir: PathBuf,

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
    info!(model_dir = %args.model_dir.display(), "loading model artifacts");
    let service = GenerationService::from_artifact_dir(&args.model_dir)
        .map_err(|source| ServerError::Engine { source })?;
    let worker = InferenceWorker::from_service(service);

    let address = SocketAddr::new(args.host, args.port);
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;

    info!(%address, model = worker.model_name(), "SmallLM server ready");
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

        assert_eq!(args.model_dir, PathBuf::from("artifacts/nilemini-8m-situ"));
        assert_eq!(args.host, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(args.port, 8080);
    }
}
