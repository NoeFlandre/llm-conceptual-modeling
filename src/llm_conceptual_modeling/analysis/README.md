# `analysis`

Deterministic offline analysis helpers live here.

## Contents

- grouped summaries and confidence intervals
- hypothesis-testing and bundle generation
- stability and replication-budget analysis
- figure-ready exports
- output-validity, variability, baseline, and variance-decomposition reports

## Module map

Primary bundle orchestrators (entry points):

- `stability_bundle.py` — replication stability bundle assembly
- `baseline_bundle.py` — non-LLM baseline comparison bundle
- `variability_bundle.py`, `hypothesis_bundle.py`, etc.

Sub-packages and extracted helpers:

- `_algo3_stability.py` — algo3-specific pair-aware stability: raw input resolution,
  pair-aware condition frames, level stability/variability aggregation, metric overview rows
- `_stability_helpers.py` — pure helpers: `slugify`, `frame_to_overview_records`,
  `patch_algorithm_rows`

## Notes

These modules should stay deterministic and file-driven. If a new analysis needs
shared logic, prefer adding a focused helper in `common/` rather than expanding a
single large file.
