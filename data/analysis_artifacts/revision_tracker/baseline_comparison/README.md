# Non-LLM Baseline Comparison Bundle

This directory contains the organized artifacts for the non-LLM baseline comparison revision item.

## Purpose

The reviewer asked for a non-LLM comparator to contextualize the value proposition of using
LLMs despite their inherent variability. The baseline is the **random-k** strategy:
for each LLM output row, the baseline samples exactly k edges uniformly at random from the
mother graph, where k equals the number of edges the LLM proposed in that row.  No information
about which edges are cross-subgraph is used.

This gives a fair, volume-matched comparison: both the LLM and the baseline are allowed
the same number of guesses, and both are evaluated against the ground-truth cross edges.

## Baseline Strategy

**random-k** (seed=42):

1. For each LLM output row, extract k = number of edges the LLM proposed.
2. Sample exactly k edges uniformly at random from the mother graph.
3. Evaluate the sampled edges against the ground-truth cross edges (same as the LLM).
4. Compare: LLM metric vs. baseline metric for the same row, then average across rows.

This answers: does the LLM find true cross edges better than random guessing,
when both are allowed the same number of guesses?

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `baseline_advantage_summary.csv`
  Cross-model summary: for each algorithm and metric, how many models beat the baseline.
- `<algo>_model_vs_baseline.csv`
  Per-model comparison: mean LLM metric vs mean baseline metric, with delta.
- `all_models_vs_baseline.csv`
  Combined all-model comparison across all three algorithms.

## Interpretation

A positive `mean_delta` means the LLM outperforms random guessing on that metric.
A negative `mean_delta` means random guessing is more effective.
The comparison is fair: both methods propose the same number of edges per row.
