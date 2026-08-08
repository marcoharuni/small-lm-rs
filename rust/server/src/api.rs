//! Axum router and HTTP handlers.

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::errors::ApiError;
use crate::schema::{
    ChatCompletionRequest, CompletionRequest, HealthResponse, ModelListResponse, ModelObject,
};
use crate::streaming;
use crate::worker::InferenceWorker;

const CHAT_UI: &str = include_str!("../static/index.html");

#[derive(Clone, Debug)]
struct AppState {
    worker: InferenceWorker,
}

/// Build the HTTP application around an inference worker.
pub fn router(worker: InferenceWorker) -> Router {
    Router::new()
        .route("/", get(chat_ui))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState { worker })
}

async fn chat_ui() -> Html<&'static str> {
    Html(CHAT_UI)
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let ready = state.worker.is_ready();
    let model = ready.then(|| state.worker.model_name().to_owned());
    Json(HealthResponse::new(ready, model))
}

async fn models(State(state): State<AppState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_owned(),
        data: vec![ModelObject {
            id: state.worker.model_name().to_owned(),
            object: "model".to_owned(),
            owned_by: "small-lm-rs".to_owned(),
        }],
    })
}

async fn completions(
    State(state): State<AppState>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(ApiError::from_json_rejection)?;
    let stream = request.stream;
    let response = state.worker.complete_text(request).await?;
    if stream {
        Ok(streaming::completion_response(&response))
    } else {
        Ok(Json(response).into_response())
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(ApiError::from_json_rejection)?;
    let stream = request.stream;
    let response = state.worker.complete_chat(request).await?;
    if stream {
        Ok(streaming::chat_response(&response))
    } else {
        Ok(Json(response).into_response())
    }
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use super::router;
    use crate::schema::{ErrorResponse, HealthResponse, ModelListResponse};
    use crate::worker::InferenceWorker;

    #[tokio::test]
    async fn root_serves_browser_chat_ui() {
        let response = router(InferenceWorker::new())
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("router response");

        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert!(content_type.starts_with("text/html"));

        let bytes = to_bytes(response.into_body(), 128 * 1024)
            .await
            .expect("read UI body");
        let body = String::from_utf8(bytes.to_vec()).expect("UTF-8 UI");
        assert!(body.contains("SmallLM"));
        assert!(body.contains("/v1/chat/completions"));
    }

    #[tokio::test]
    async fn health_endpoint_reports_unloaded_test_worker() {
        let response = router(InferenceWorker::new())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("router response");

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 4_096)
            .await
            .expect("read response body");
        let body = serde_json::from_slice::<HealthResponse>(&bytes).expect("valid health JSON");
        assert!(!body.ready);
    }

    #[tokio::test]
    async fn models_endpoint_lists_the_configured_identifier() {
        let response = router(InferenceWorker::new())
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("valid request"),
            )
            .await
            .expect("router response");

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 4_096)
            .await
            .expect("read response body");
        let body = serde_json::from_slice::<ModelListResponse>(&bytes).expect("valid model list");
        assert_eq!(body.data[0].id, "nilemini-8m-situ");
        assert_eq!(body.data[0].owned_by, "small-lm-rs");
    }

    #[tokio::test]
    async fn unloaded_worker_fails_with_structured_not_implemented_error() {
        let body = serde_json::json!({
            "model": "nilemini-8m-situ",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1
        });
        let response = router(InferenceWorker::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("router response");

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let bytes = to_bytes(response.into_body(), 8_192)
            .await
            .expect("read error body");
        let body = serde_json::from_slice::<ErrorResponse>(&bytes).expect("valid error JSON");
        assert_eq!(body.error.code, "not_implemented");
    }

    #[tokio::test]
    async fn unknown_json_fields_return_structured_bad_request() {
        let body = serde_json::json!({
            "model": "nilemini-8m-situ",
            "prompt": "Hello",
            "unsupported": true
        });
        let response = router(InferenceWorker::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("valid request"),
            )
            .await
            .expect("router response");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), 8_192)
            .await
            .expect("read error body");
        let body = serde_json::from_slice::<ErrorResponse>(&bytes).expect("valid error JSON");
        assert_eq!(body.error.code, "invalid_request");
    }
}
