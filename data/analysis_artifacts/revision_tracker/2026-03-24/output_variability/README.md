# Raw Output Variability Audit Bundle

This directory contains the organized artifacts for the output-variability revision item.

## Purpose

The reviewer asked for a mechanism-level explanation of run-to-run variability. This bundle
captures a proxy analysis: do repeated runs produce the same edge sets, or do they drift
into different edges, and if so, how much?

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Cross-algorithm summary with mean Jaccard, exact-match rate, edge counts,
  and breadth expansion ratio per algorithm.
- `algorithm_output_variability_summary.csv`
  Top-level cross-algorithm summary.
- `model_output_variability_summary.csv`
  Model-level summary grouped by algorithm and model.
- `output_variability_extremes.csv`
  Most extreme source files by Jaccard and breadth expansion ratio.
- `<algo>/condition_output_variability.csv`
  Per-model, per-source-file variability detail.
- `<algo>/by_depth.csv` (ALGO3 only)
  ALGO3 variability broken down by search Depth.
- `<algo>/by_word_count.csv` (ALGO3 only)
  ALGO3 variability broken down by word budget.

## Key Interpretation

- **ALGO1 and ALGO2**: mean pairwise Jaccard ≈ 1.000, exact-match rate ≈ 0.999,
  breadth expansion ratio ≈ 1.00. These methods are near-deterministic.
- **ALGO3**: mean pairwise Jaccard ≈ 0.077, exact-match rate ≈ 0.001,
  breadth expansion ratio ≈ 4.13. Repeated runs produce substantially different
  edge sets — only about 8% overlap on average.
