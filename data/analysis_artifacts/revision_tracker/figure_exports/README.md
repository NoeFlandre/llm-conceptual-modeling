# Figure-Ready Exports Audit Bundle

This directory contains plot-ready metric exports and distributional summaries for
the figure-ready revision item.

## Purpose

The reviewer asked for confidence intervals and distributional plots alongside mean
performance. This bundle produces deterministic, long-format metric rows and
per-model distributional summaries that can be consumed directly by plotting tools
without ad hoc post-processing.

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Distributional summary (n, mean, std, 95% CI, median, q1, q3, min, max) per
  model and metric across all algorithms.
- `<algorithm>_metric_rows.csv`
  Long-format metric rows for all models of that algorithm, suitable for
  box plots, violin plots, and faceted model comparisons.
- `<algorithm>/<model>/distributional_summary.csv`
  Distributional summary for a specific model, including 95% confidence
  intervals on the mean using the t-distribution.

## Interpretation

The distributional summaries show both central tendency (mean, median) and
spread (std, CI, quartiles) per model per metric. When the 95% CI does not
cross zero for a metric, the evaluated performance is reliably above or
below the reference regardless of repetition noise.
