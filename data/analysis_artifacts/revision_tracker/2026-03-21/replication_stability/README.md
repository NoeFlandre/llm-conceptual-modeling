# Replication Stability Audit Bundle

This directory contains the organized artifacts for the replication-stability revision item.

## Purpose

The reviewer asked for a principled justification of the five-replication decision. This bundle
captures repetition-level stability analysis over the five recorded runs in the imported corpus.
Rather than a formal power analysis, this bundle shows how much the evaluated metrics actually
moved across repetitions — confirming that five runs are enough to reveal which methods are
stable and which are not.

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Cross-algorithm stability summary (CV and range-width) by algorithm and metric.
- `variability_incidence_by_algorithm.csv`
  Cross-algorithm count and share of conditions that changed across repetitions.
- `overall_metric_stability_by_algorithm.csv`
  Coefficient-of-variation and range-width summaries by algorithm and metric.
- `<algorithm>/condition_stability.csv`
  Per-file, per-condition stability statistics across the five repetitions.
- `<algorithm>/<factor>_stability_by_level.csv`
  Aggregated stability summaries grouped by a specific factor level.
- `<algorithm>/<factor>_variability_incidence.csv`
  Incidence of any run-to-run variation grouped by a specific factor level.

## Key Interpretation

- **ALGO1 and ALGO2 are nearly repetition-stable**: median CVs are 0.0 across most conditions.
- **ALGO2 is especially stable when Convergence = 1**: zero varying conditions.
- **ALGO3 is orders of magnitude noisier**: median CV on recall is 3.87, meaning noise is
  nearly 4 times the signal size — not a small or marginal difference.
