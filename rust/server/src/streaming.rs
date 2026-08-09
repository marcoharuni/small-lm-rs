//! OpenAI-compatible Server-Sent Events framing.

use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::Response;
use serde_json::json;

use crate::schema::{ChatCompletionResponse, CompletionResponse};

/// Convert a completed chat response into SSE frames ending in `[DONE]`.
#[must_use]
pub fn chat_response(response: &ChatCompletionResponse) -> Response {
    let choice = &response.choices[0];
    let mut body = String::new();

    append_event(
        &mut body,
        &json!({
            "id": &response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": &response.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null
            }]
        }),
    );

    for character in choice.message.content.chars() {
        append_event(
            &mut body,
            &json!({
                "id": &response.id,
                "object": "chat.completion.chunk",
                "created": response.created,
                "model": &response.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": character.to_string()},
                    "finish_reason": null
                }]
            }),
        );
    }

    append_event(
        &mut body,
        &json!({
            "id": &response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": &response.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": &choice.finish_reason
            }]
        }),
    );
    body.push_str("data: [DONE]\n\n");
    event_stream_response(body)
}

/// Convert a completed text response into SSE frames ending in `[DONE]`.
#[must_use]
pub fn completion_response(response: &CompletionResponse) -> Response {
    let choice = &response.choices[0];
    let mut body = String::new();

    for character in choice.text.chars() {
        append_event(
            &mut body,
            &json!({
                "id": &response.id,
                "object": "text_completion",
                "created": response.created,
                "model": &response.model,
                "choices": [{
                    "text": character.to_string(),
                    "index": 0,
                    "finish_reason": null
                }]
            }),
        );
    }

    append_event(
        &mut body,
        &json!({
            "id": &response.id,
            "object": "text_completion",
            "created": response.created,
            "model": &response.model,
            "choices": [{
                "text": "",
                "index": 0,
                "finish_reason": &choice.finish_reason
            }]
        }),
    );
    body.push_str("data: [DONE]\n\n");
    event_stream_response(body)
}

fn append_event(body: &mut String, value: &serde_json::Value) {
    body.push_str("data: ");
    body.push_str(&value.to_string());
    body.push_str("\n\n");
}

fn event_stream_response(body: String) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from(body))
        .expect("static SSE response headers are valid")
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::header;

    use super::chat_response;
    use crate::schema::{ChatCompletionChoice, ChatCompletionResponse, ChatMessage, Usage};

    #[tokio::test]
    async fn chat_stream_contains_chunks_and_done_sentinel() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-test".to_owned(),
            object: "chat.completion".to_owned(),
            created: 1,
            model: "smalllm-8m-situ".to_owned(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_owned(),
                    content: "Hi".to_owned(),
                },
                finish_reason: "length".to_owned(),
            }],
            usage: Usage::new(3, 1),
        };

        let response = chat_response(&response);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let bytes = to_bytes(response.into_body(), 16 * 1024)
            .await
            .expect("read SSE body");
        let text = String::from_utf8(bytes.to_vec()).expect("UTF-8 SSE");
        assert!(text.contains("chat.completion.chunk"));
        assert!(text.contains("\"content\":\"H\""));
        assert!(text.contains("data: [DONE]"));
    }

    #[test]
    fn body_type_remains_axum_compatible() {
        let _: Body = Body::empty();
    }
}
