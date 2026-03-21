# llm-conceptual-modeling

Offline rewrite of the conceptual modeling research workflows with regression tests against legacy artifacts.

## Current Scope

The new codebase currently covers the deterministic offline workflows for all three algorithms:

- shared connection evaluation logic for Algorithms 1 and 2
- offline Algorithm 1 metrics evaluation
- offline Algorithm 2 metrics evaluation
- offline Algorithm 3 recall recomputation
- ALGO1 factorial analysis from evaluated metrics CSVs
- ALGO2 factorial analysis from evaluated metrics CSVs
- ALGO3 factorial analysis from evaluated metrics CSVs
- self-verification commands: `lcm doctor` and `lcm verify legacy-parity`
- offline generation stubs for `algo1`, `algo2`, and `algo3`
- CLI entrypoints for `lcm eval algo1`, `lcm eval algo2`, `lcm eval algo3`, `lcm factorial algo1`, `lcm factorial algo2`, `lcm factorial algo3`, `lcm doctor`, `lcm verify legacy-parity`, and `lcm generate`
- regression tests against committed legacy fixtures

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

uv run lcm eval algo3 \
  --input tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --output /tmp/method3_results_evaluated_gpt5.csv

uv run lcm factorial algo1 \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --output /tmp/factorial_analysis_algo1_gpt_5_without_error.csv

uv run lcm factorial algo2 \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --output /tmp/factorial_analysis_gpt_5_algo2_without_error.csv

uv run lcm factorial algo3 \
  --input tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv \
  --output /tmp/factorial_analysis_results_gpt5_without_error.csv

uv run lcm doctor --json

uv run lcm verify legacy-parity --json

uv run lcm generate algo1 --json

uv run lcm generate algo3 --fixture-only --json
```
