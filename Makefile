.PHONY: sync sync-locked lock-check format-check lint typecheck test verify doctor guardrails cli-smoke ci

sync:
	uv sync --dev

sync-locked:
	uv sync --locked --dev

lock-check:
	uv lock --check

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run ty check

test:
	uv run pytest

verify:
	uv run lcm verify all --json

doctor:
	uv run lcm doctor --json

guardrails:
	uv run python scripts/check_repo_guardrails.py

cli-smoke:
	uv run lcm doctor --json
	uv run lcm generate algo1 --fixture-only --json
	uv run pytest tests/verification/test_cli_ci_smoke.py

ci:
	$(MAKE) lock-check
	$(MAKE) lint
	$(MAKE) guardrails
	$(MAKE) cli-smoke
	$(MAKE) test
	$(MAKE) verify
