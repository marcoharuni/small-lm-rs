#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-artifacts/nilemini-8m-situ}"
HOST="${NILEMINI_SMOKE_HOST:-127.0.0.1}"
PORT="${NILEMINI_SMOKE_PORT:-18080}"
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="${TMPDIR:-/tmp}/nilemini-server-smoke.log"

echo "Building the release server before the readiness timer starts..."
cargo build --release -q -p nilemini-server

SERVER_BINARY="target/release/nilemini-server"
if [[ ! -x "$SERVER_BINARY" ]]; then
  echo "Release server binary was not created: $SERVER_BINARY"
  exit 1
fi

"$SERVER_BINARY" \
  --model-dir "$MODEL_DIR" \
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
  if curl -fsS "$BASE_URL/health" >/tmp/nilemini-health.json 2>/dev/null; then
    if python3 - <<'PYREADY'
import json
from pathlib import Path

data = json.loads(Path("/tmp/nilemini-health.json").read_text())
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

curl -fsS "$BASE_URL/v1/models" >/tmp/nilemini-models.json

curl -fsS \
  -H 'content-type: application/json' \
  -d '{
    "model":"nilemini-8m-situ",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":1,
    "temperature":0.0,
    "top_p":1.0,
    "top_k":0,
    "seed":0
  }' \
  "$BASE_URL/v1/chat/completions" >/tmp/nilemini-chat.json

curl -fsS \
  -H 'content-type: application/json' \
  -d '{
    "model":"nilemini-8m-situ",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":1,
    "temperature":0.0,
    "stream":true
  }' \
  "$BASE_URL/v1/chat/completions" >/tmp/nilemini-chat.sse

python3 - <<'PY'
import json
from pathlib import Path

health = json.loads(Path("/tmp/nilemini-health.json").read_text())
assert health["status"] == "ok"
assert health["ready"] is True
assert health["model"] == "nilemini-8m-situ"

models = json.loads(Path("/tmp/nilemini-models.json").read_text())
assert models["object"] == "list"
assert models["data"][0]["id"] == "nilemini-8m-situ"

chat = json.loads(Path("/tmp/nilemini-chat.json").read_text())
assert chat["object"] == "chat.completion"
assert chat["model"] == "nilemini-8m-situ"
assert chat["choices"][0]["message"]["role"] == "assistant"
assert chat["usage"]["completion_tokens"] == 1
assert chat["usage"]["total_tokens"] == chat["usage"]["prompt_tokens"] + 1

sse = Path("/tmp/nilemini-chat.sse").read_text()
assert "chat.completion.chunk" in sse
assert "data: [DONE]" in sse

print(json.dumps(
    {
        "health": health,
        "model": models["data"][0]["id"],
        "finish_reason": chat["choices"][0]["finish_reason"],
        "usage": chat["usage"],
        "sse_done": True,
    },
    indent=2,
))
PY
