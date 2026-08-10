#!/usr/bin/env bash

MODEL_DIR="${1:-artifacts/small-lm-8m}"
HOST="${SMALLLM_SMOKE_HOST:-127.0.0.1}"
PORT="${SMALLLM_SMOKE_PORT:-18080}"
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="${TMPDIR:-/tmp}/small-lm-server-smoke.log"

echo "Building the release server before the readiness timer starts..."
cargo build --release -q -p smalllm-server || exit $?

SERVER_BINARY="target/release/smalllm-server"
if [[ ! -x "$SERVER_BINARY" ]]; then
  echo "Release server binary was not created: $SERVER_BINARY"
  exit 1
fi

"$SERVER_BINARY" \
  --model-dir "$MODEL_DIR" \
  --max-active-sequences 8 \
  --host "$HOST" \
  --port "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 300); do
  if curl -fsS "$BASE_URL/health" >/tmp/small-lm-health.json 2>/dev/null; then
    if python3 - <<'PYREADY'
import json
from pathlib import Path

data = json.loads(Path("/tmp/small-lm-health.json").read_text())
raise SystemExit(0 if data.get("ready") is True else 1)
PYREADY
    then
      ready=1
      break
    fi
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited during startup:"
    cat "$LOG_FILE"
    exit 1
  fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "Server did not become ready in time."
  echo "Server log:"
  if [[ -s "$LOG_FILE" ]]; then
    cat "$LOG_FILE"
  else
    echo "(log file is empty)"
  fi
  exit 1
fi

curl -fsS "$BASE_URL/v1/models" >/tmp/small-lm-models.json || exit $?

curl -fsS \
  -H 'content-type: application/json' \
  -d '{
    "model":"small-lm-8m",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":1,
    "temperature":0.0,
    "top_p":1.0,
    "top_k":0,
    "seed":0
  }' \
  "$BASE_URL/v1/chat/completions" >/tmp/small-lm-chat.json || exit $?

curl -fsS \
  -H 'content-type: application/json' \
  -d '{
    "model":"small-lm-8m",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":1,
    "temperature":0.0,
    "stream":true
  }' \
  "$BASE_URL/v1/chat/completions" >/tmp/small-lm-chat.sse || exit $?

concurrent_pids=()
for index in 0 1 2 3; do
  curl -fsS \
    -H 'content-type: application/json' \
    -d "{
      \"model\":\"small-lm-8m\",
      \"messages\":[{\"role\":\"user\",\"content\":\"Concurrent request ${index}\"}],
      \"max_tokens\":4,
      \"temperature\":0.0,
      \"top_p\":1.0,
      \"top_k\":0,
      \"seed\":0
    }" \
    "$BASE_URL/v1/chat/completions" >"/tmp/small-lm-concurrent-${index}.json" &
  concurrent_pids+=("$!")
done

for pid in "${concurrent_pids[@]}"; do
  if ! wait "$pid"; then
    echo "Concurrent HTTP request failed."
    cat "$LOG_FILE"
    exit 1
  fi
done

python3 - <<'PY'
import json
from pathlib import Path

health = json.loads(Path("/tmp/small-lm-health.json").read_text())
assert health["status"] == "ok"
assert health["ready"] is True
assert health["model"] == "small-lm-8m"

models = json.loads(Path("/tmp/small-lm-models.json").read_text())
assert models["object"] == "list"
assert models["data"][0]["id"] == "small-lm-8m"

chat = json.loads(Path("/tmp/small-lm-chat.json").read_text())
assert chat["object"] == "chat.completion"
assert chat["model"] == "small-lm-8m"
assert chat["choices"][0]["message"]["role"] == "assistant"
assert chat["usage"]["completion_tokens"] == 1
assert chat["usage"]["total_tokens"] == chat["usage"]["prompt_tokens"] + 1

sse = Path("/tmp/small-lm-chat.sse").read_text()
assert "chat.completion.chunk" in sse
assert "data: [DONE]" in sse

concurrent = []
for index in range(4):
    response = json.loads(Path(f"/tmp/small-lm-concurrent-{index}.json").read_text())
    assert response["object"] == "chat.completion"
    assert response["model"] == "small-lm-8m"
    assert response["usage"]["completion_tokens"] == 4
    assert response["choices"][0]["finish_reason"] in {"length", "stop"}
    concurrent.append({
        "id": response["id"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "finish_reason": response["choices"][0]["finish_reason"],
    })

assert len({item["id"] for item in concurrent}) == 4

print(json.dumps(
    {
        "health": health,
        "model": models["data"][0]["id"],
        "finish_reason": chat["choices"][0]["finish_reason"],
        "usage": chat["usage"],
        "sse_done": True,
        "concurrent_requests": concurrent,
    },
    indent=2,
))
PY
