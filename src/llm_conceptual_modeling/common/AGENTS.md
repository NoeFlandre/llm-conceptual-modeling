# Common Module Guide

This module contains shared primitives used across the package.

Purpose:

- Centralize reusable graph, evaluation, schema, retry, and literal-parsing helpers.
- Keep shared behavior here instead of duplicating logic in `algo1`, `algo2`, `algo3`, or `commands`.

Key files:

- `anthropic.py`: Anthropic client wrapper used for debugging and experimental paths.
- `mistral.py`: shared Mistral client utilities and prompt-format helpers.
- `retry.py`: retry helper shared by provider clients.
- `graph_data.py`: loads canonical graph fixtures and thesaurus data.
- `connection_eval.py`: derives valid connections between subgraphs.
- `evaluation_core.py`: shared CSV evaluation logic.
- `factorial_core.py`: shared factorial-analysis logic.
- `baseline.py`: direct cross-subgraph edge baseline helpers.
- `csv_schema.py`: dataframe schema validation helpers.
- `literals.py`: safe parsing helpers for stored literal values.
- `model_catalog.py`: paper-model aliases and provider-aware model resolution.
- `types.py`: shared dataclasses and type aliases.

Working rules:

- Keep functions small and single-purpose.
- Prefer adding a new helper here only when it is reused by multiple modules.
- If a file approaches 300 lines, split by responsibility before adding more code.
- If this directory grows too much, split along the concern boundaries above.
