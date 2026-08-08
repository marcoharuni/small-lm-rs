# NileMini HTTP API

The Rust server loads the model, tokenizer, and generation metadata once at startup and exposes both a local browser chat page and an OpenAI-compatible API subset.

```bash
cargo run --release --package nilemini-server -- \
  --model-dir artifacts/nilemini-8m-situ \
  --host 127.0.0.1 \
  --port 8080
```

Keep the server on loopback. Authentication, TLS, quotas, and multi-tenant hardening are not implemented.

## Browser chat

Open:

```text
http://127.0.0.1:8080/
```

The page is embedded directly into the Rust server binary and talks to the same `/health` and `/v1/chat/completions` endpoints documented below. No Node.js frontend, separate web server, or CORS configuration is required.

The UI keeps the current conversation in browser memory, exposes `max_tokens` and `temperature`, shows token usage returned by the API, and includes a clear-conversation control.

## `GET /health`

```json
{
  "status": "ok",
  "service": "nilemini-server",
  "ready": true,
  "model": "nilemini-8m-situ"
}
```

## `GET /v1/models`

Returns an OpenAI-shaped model list containing `nilemini-8m-situ`.

## `POST /v1/chat/completions`

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "nilemini-8m-situ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 0
  }'
```

Supported roles are `system`, `user`, and `assistant`. The frozen chat template allows a system message only at index zero and requires the final message to be from the user.

## `POST /v1/completions`

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "nilemini-8m-situ",
    "prompt": "Hello",
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

## Sampling and limits

- `max_tokens`: default 16, accepted range 1 through 128
- `temperature`: default 0; zero is greedy
- `top_p`: default 1, accepted range `(0, 1]`
- `top_k`: default 0; zero disables top-k filtering
- `seed`: default 0
- unknown JSON fields receive HTTP 400

The engine separately enforces the 512-token total context limit.

## Streaming

Set `"stream": true` to receive `text/event-stream` frames ending with:

```text
data: [DONE]
```

The current synchronous CPU service completes generation before emitting the SSE body. The wire contract is client-compatible; token-time delivery and disconnect-aware cancellation remain later optimizations.

## Concurrency

HTTP handling is asynchronous. CPU inference runs on Tokio's blocking pool and is serialized through the loaded service. Each generation invocation creates a fresh KV cache, so decoding state is never shared between requests.

## End-to-end smoke test

```bash
./scripts/smoke_server.sh artifacts/nilemini-8m-situ
```

For a short presentation workflow, see [`DEMO.md`](DEMO.md).
