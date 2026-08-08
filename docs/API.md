# HTTP API

The Rust server loads the model and tokenizer once at startup and serves both a browser chat and an OpenAI-compatible API subset.

```bash
./scripts/run_server.sh
```

The default address is `127.0.0.1:8080`.

## Browser chat

Open:

```text
http://127.0.0.1:8080/
```

The page is embedded in the Rust binary and uses the same chat-completions endpoint as API clients.

## Health

```text
GET /health
```

A ready server returns the loaded model identifier and `service: small-lm-rs`.

## Models

```text
GET /v1/models
```

The bundled artifact uses the compatibility identifier `nilemini-8m-situ`.

## Chat completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "nilemini-8m-situ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

Supported roles are `system`, `user`, and `assistant`.

## Text completions

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

## Generation controls

- `max_tokens`: 1–128, default 16
- `temperature`: 0 selects greedy decoding
- `top_p`: `(0, 1]`
- `top_k`: 0 disables top-k filtering
- `seed`: deterministic sampler seed

The model enforces a 512-token total context limit.

## Streaming

`"stream": true` returns `text/event-stream` frames ending in:

```text
data: [DONE]
```

Generation currently completes before the buffered SSE body is emitted. True token-time streaming is not implemented yet.

## Concurrency

HTTP handling is asynchronous. CPU inference runs on Tokio's blocking pool and is serialized through the loaded generation service. Each request gets its own KV cache.
