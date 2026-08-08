# NileMini-8M-SiTU Demo Walkthrough

This is a short, reproducible demo flow for NileMini-8M-SiTU v1.0.1.

## 1. Show the project

Open the repository root and point out:

- JAX/Flax NNX training code in `src/nilemini/`
- final trained artifact in `artifacts/nilemini-8m-situ/`
- Rust inference engine in `rust/engine/`
- Rust HTTP server in `rust/server/`

State the core result:

> NileMini-8M-SiTU is a 7,999,744-parameter decoder-only language model trained on 140,017,664 FineWeb-Edu tokens, lightly instruction-tuned on SmolTalk, exported to FP32 SafeTensors, and executed independently by a Rust CPU engine.

## 2. Verify the artifact

```bash
cd artifacts/nilemini-8m-situ
sha256sum -c SHA256SUMS
cd ../..
```

Expected result: every artifact file reports `OK`.

## 3. Show JAX ↔ Rust parity

```bash
cargo run --release -p nilemini-engine --example parity -- \
  artifacts/nilemini-8m-situ
```

Important measured values:

- 81,920 logits compared
- max absolute error: 0.0528898239
- mean absolute error: 0.0052069233
- RMSE: 0.0074163827
- cosine similarity: 0.9999967275
- top-1 agreement: 10/10
- final-position top-1 match: true
- accepted: true
- violations: none

## 4. Start the Rust server

```bash
cargo run --release -p nilemini-server -- \
  --model-dir artifacts/nilemini-8m-situ \
  --host 127.0.0.1 \
  --port 8080
```

The server should log that `nilemini-8m-situ` is ready on `127.0.0.1:8080`.

## 5. Chat in the browser

Open:

```text
http://127.0.0.1:8080/
```

The web page talks directly to the same Rust `/v1/chat/completions` endpoint.

Suggested prompts:

```text
Hello
```

```text
What is a computer?
```

```text
Write one short sentence about the Nile River.
```

Keep expectations realistic: this is an 8M-parameter research/engineering model with a deliberately small SFT pass.

## 6. Show the API directly

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nilemini-8m-situ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

Also show:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

## 7. Close with the engineering story

A concise closing statement:

> The important result is the full path: deterministic JAX training, framework-independent SafeTensors export, an independent Rust Transformer implementation, numerical JAX-to-Rust parity, KV-cached CPU generation, and both browser and OpenAI-compatible API interfaces around the same trained model.

## Suggested video length

A strong demo can be 3–5 minutes:

1. repository overview — 30 seconds
2. artifact checksum — 20 seconds
3. parity result — 40 seconds
4. start server — 20 seconds
5. browser chat — 60–90 seconds
6. API curl — 30 seconds
7. architecture/training results and closing — 40 seconds
