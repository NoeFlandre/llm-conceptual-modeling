# llm-conceptual-modeling

Offline rewrite of the conceptual modeling research workflows with regression tests against legacy artifacts.

## Current Scope

The new codebase currently covers the first verified slice:

- shared connection evaluation logic for Algorithms 1 and 2
- offline Algorithm 1 metrics evaluation
- offline Algorithm 2 metrics evaluation
- CLI entrypoints for `lcm eval algo1` and `lcm eval algo2`
- regression tests against a committed legacy fixture

Live LLM generation is intentionally out of scope at this stage.

## Setup

```bash
uv sync --dev
```

## Verify

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

## Run

```bash
uv run lcm eval algo1 \
  --input tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv \
  --output /tmp/metrics_sg1_sg2.csv
```
