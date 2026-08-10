#!/usr/bin/env python3
"""Benchmark SmallLM's continuous-batching HTTP server with concurrent clients."""

from __future__ import annotations

import concurrent.futures
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/small-lm-8m")
HOST = os.environ.get("SMALLLM_BENCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("SMALLLM_BENCH_PORT", "18081"))
BASE_URL = f"http://{HOST}:{PORT}"
MAX_TOKENS = int(os.environ.get("SMALLLM_BENCH_MAX_TOKENS", "8"))
ROUNDS = int(os.environ.get("SMALLLM_BENCH_ROUNDS", "3"))
CONCURRENCY_LEVELS = (1, 2, 4, 8)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def request_json(path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["content-type"] = "application/json"
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def wait_ready(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            health = request_json("/health")
            if health.get("ready") is True:
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise RuntimeError("server did not become ready within 120 seconds")


def read_rss_kib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return 0
    return 0


def run_one(index: int) -> tuple[float, int]:
    payload = {
        "model": "small-lm-8m",
        "messages": [
            {
                "role": "user",
                "content": f"Give a short greeting for request {index}.",
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 0,
    }
    started = time.perf_counter()
    response = request_json("/v1/chat/completions", payload)
    latency = time.perf_counter() - started
    if response.get("object") != "chat.completion":
        raise RuntimeError(f"unexpected response object: {response!r}")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"missing usage object: {response!r}")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise RuntimeError(f"invalid completion token count: {response!r}")
    return latency, completion_tokens


def benchmark_level(concurrency: int) -> dict[str, object]:
    latencies: list[float] = []
    total_tokens = 0
    wall_started = time.perf_counter()
    request_index = 0
    for _ in range(ROUNDS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_one, request_index + i) for i in range(concurrency)]
            request_index += concurrency
            for future in futures:
                latency, completion_tokens = future.result()
                latencies.append(latency)
                total_tokens += completion_tokens
    wall_seconds = time.perf_counter() - wall_started
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "generated_tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "aggregate_tokens_per_second": total_tokens / wall_seconds,
        "request_latency_ms": {
            "mean": 1000.0 * sum(latencies) / len(latencies),
            "p50": 1000.0 * median(latencies),
            "p95": 1000.0 * percentile(latencies, 0.95),
            "p99": 1000.0 * percentile(latencies, 0.99),
            "min": 1000.0 * min(latencies),
            "max": 1000.0 * max(latencies),
        },
    }


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    os.chdir(REPO_ROOT)
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise SystemExit(f"missing model artifact: {MODEL_DIR / 'model.safetensors'}")

    subprocess.run(
        ["cargo", "build", "--release", "--quiet", "-p", "smalllm-server"],
        check=True,
    )
    binary = REPO_ROOT / "target/release/smalllm-server"
    process = subprocess.Popen(
        [
            str(binary),
            "--model-dir",
            str(MODEL_DIR),
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--max-active-sequences",
            "8",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    peak_rss_kib = 0
    stop_monitor = threading.Event()

    def monitor_memory() -> None:
        nonlocal peak_rss_kib
        while not stop_monitor.is_set():
            peak_rss_kib = max(peak_rss_kib, read_rss_kib(process.pid))
            time.sleep(0.01)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()

    try:
        wait_ready(process)
        run_one(-1)  # warmup
        results = [benchmark_level(level) for level in CONCURRENCY_LEVELS]
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    metadata = {
        "git_head": git_value("rev-parse", "HEAD"),
        "git_state": "clean" if not git_value("status", "--porcelain") else "dirty",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "max_active_sequences": 8,
        "max_tokens_per_request": MAX_TOKENS,
        "rounds_per_concurrency": ROUNDS,
        "peak_server_rss_mib": peak_rss_kib / 1024.0,
        "note": (
            "HTTP request latency is measured end-to-end. The current SSE adapter buffers the "
            "completed generation, so these results intentionally do not report HTTP TTFT/TPOT."
        ),
    }
    report = {"metadata": metadata, "results": results}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
