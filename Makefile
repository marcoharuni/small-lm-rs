.DEFAULT_GOAL := test

.PHONY: setup format lint typecheck test-python test-rust test build-rust run-server clean

setup:
	uv sync

format:
	uv run ruff format .
	cargo fmt --all

lint:
	uv run ruff check .
	cargo clippy --workspace --all-targets -- -D warnings

typecheck:
	uv run mypy

test-python:
	uv run pytest

test-rust:
	cargo test --workspace

test: test-python test-rust

build-rust:
	cargo build --workspace

run-server:
	cargo run --package smalllm-server

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	cargo clean
