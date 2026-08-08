//! Structured HTTP and process-level server errors.

use std::io;
use std::net::SocketAddr;

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use nilemini_engine::EngineError;
use thiserror::Error;

use crate::schema::ErrorResponse;

/// Failures returned through the HTTP API.
#[derive(Debug, Error)]
pub enum ApiError {
    /// The router has no loaded inference service.
    #[error("{feature} is not implemented because model inference is not ready")]
    NotImplemented {
        /// Unavailable feature.
        feature: &'static str,
    },

    /// The client supplied an invalid request.
    #[error("invalid request: {message}")]
    BadRequest {
        /// Failed request constraint.
        message: String,
    },

    /// A blocking inference task failed or panicked.
    #[error("inference worker failed: {message}")]
    Worker {
        /// Worker failure details.
        message: String,
    },

    /// The inference engine failed unexpectedly.
    #[error("inference engine error: {source}")]
    Engine {
        /// Underlying engine failure.
        #[source]
        source: EngineError,
    },
}

impl ApiError {
    /// Construct an explicit unavailable-feature failure.
    #[must_use]
    pub const fn not_implemented(feature: &'static str) -> Self {
        Self::NotImplemented { feature }
    }

    /// Construct an invalid-request error.
    #[must_use]
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest {
            message: message.into(),
        }
    }

    /// Convert an Axum JSON rejection into the stable API envelope.
    #[must_use]
    pub fn from_json_rejection(rejection: JsonRejection) -> Self {
        Self::bad_request(rejection.body_text())
    }
}

impl From<EngineError> for ApiError {
    fn from(source: EngineError) -> Self {
        match source {
            EngineError::InvalidInput { .. } | EngineError::InvalidConfiguration { .. } => {
                Self::BadRequest {
                    message: source.to_string(),
                }
            }
            _ => Self::Engine { source },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, code) = match &self {
            Self::NotImplemented { .. } => (
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented_error",
                "not_implemented",
            ),
            Self::BadRequest { .. } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_request",
            ),
            Self::Worker { .. } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "worker_error",
            ),
            Self::Engine { .. } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "engine_error",
            ),
        };
        let body = ErrorResponse::new(self.to_string(), error_type, code);
        (status, Json(body)).into_response()
    }
}

/// Failures which prevent the HTTP process from starting or serving.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Model artifacts could not be loaded.
    #[error("failed to initialize inference engine: {source}")]
    Engine {
        /// Underlying engine failure.
        #[source]
        source: EngineError,
    },

    /// The configured network address could not be bound.
    #[error("failed to bind HTTP server at {address}: {source}")]
    Bind {
        /// Requested socket address.
        address: SocketAddr,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },

    /// The Axum serving loop terminated with an error.
    #[error("HTTP server failed: {0}")]
    Serve(#[source] io::Error),
}
