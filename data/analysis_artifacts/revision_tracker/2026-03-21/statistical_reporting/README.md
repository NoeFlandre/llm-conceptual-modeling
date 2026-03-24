# Statistical Reporting Audit Bundle

This directory contains the organized artifacts for the statistical-reporting revision item.

## Purpose

The reviewer asked for confidence intervals and stronger statistical reporting. This bundle captures
the descriptive evidence used to answer that request across all imported evaluated result files.

## Layout

- `bundle_manifest.csv`
  Index of every generated summary file and overview file.
- `bundle_overview.csv`
  One row per algorithm, factor, and metric with global means and per-file winner counts.
- `<algorithm>/<factor_slug>/grouped_metric_summary.csv`
  The full grouped descriptive summary produced by `lcm analyze summary`.
- `<algorithm>/<factor_slug>/metric_overview.csv`
  A compact reviewer-facing overview for that factor.

## Interpretation

The grouped summary files preserve per-source-file provenance. The metric overview files compress
that evidence into global means and file-level winner counts so the revision document can cite the
most informative patterns without repeating entire CSVs inline.
