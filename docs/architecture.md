# Architecture

## Goals

The repository preserves the legacy project behavior for deterministic offline workflows while making future changes safer.

## Layout

- `src/llm_conceptual_modeling/algo1`, `algo2`, `algo3`: algorithm-specific adapters and remaining behavior differences.
- `src/llm_conceptual_modeling/common`: shared graph loading, evaluation logic, factorial helpers, CSV schema checks, and typed models.
- `src/llm_conceptual_modeling/commands`: CLI handlers separated from domain logic.
- `tests/fixtures/legacy`: committed oracle artifacts copied from the old codebase.
- `paper/`: local manuscript sources used for project context only. This directory is intentionally ignored and is not required for verification.

## Safety Model

- Deterministic workflows are protected by full-file parity tests against legacy fixtures.
- `lcm doctor` checks basic repository health.
- `lcm verify legacy-parity` reruns all migrated offline workflows against oracle fixtures.
- `lcm verify all` combines health checks and parity checks into one machine-readable gate.

## Explicit Boundary

Live LLM calls are intentionally not executed in automated verification. Generation code is represented as offline manifests and migrated experiment structure, not validated provider behavior.
