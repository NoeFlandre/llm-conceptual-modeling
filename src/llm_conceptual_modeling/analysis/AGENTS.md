# Analysis Module Guide

This module owns reviewer-facing post-processing and reporting helpers.

Purpose:

- Produce grouped summaries, stability tables, figure-ready rows, hypothesis tests, failure analysis, and baseline comparisons.
- Keep this layer deterministic and CSV-oriented.
- Reuse shared schema and type helpers from `common`.

Key files:

- `summary.py`: grouped metric summaries.
- `stability.py`: grouped metric stability reporting.
- `figures.py`: figure-ready metric row exports.
- `failures.py`: row-level failure analysis.
- `hypothesis.py`: paired-factor hypothesis tests.
- `baseline_comparison.py`: comparisons against baseline runs.

Working rules:

- Keep the analysis layer independent from CLI parsing and from algorithm-specific orchestration.
- Prefer small, composable data transforms over large monolithic report builders.
- If `baseline_comparison.py` or `hypothesis.py` grows further, split by statistic or report type before adding more code.
- Keep the file set small by introducing subdirectories if new analysis families are added.
