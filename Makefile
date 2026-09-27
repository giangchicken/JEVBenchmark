# `make check` is the gate. It excludes tests marked `integration`, which call a real
# model endpoint.
.PHONY: check lint types test integration

check: lint types test

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy --strict benchmark

test:
	uv run pytest -q -m "not integration"

integration:
	uv run pytest -q -m integration
