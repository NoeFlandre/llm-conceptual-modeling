.PHONY: sync test lint typecheck verify doctor ci

sync:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run ty check

verify:
	uv run lcm verify all --json

doctor:
	uv run lcm doctor --json

ci:
	uv run ruff check .
	uv run ty check
	uv run pytest
