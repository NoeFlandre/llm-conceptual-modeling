# Non-LLM Baseline Comparison Bundle

This directory contains the organized artifacts for the non-LLM baseline comparison revision item.

## Purpose

The reviewer asked for non-LLM comparators to contextualize the value proposition of using
LLMs despite their inherent variability. The bundle compares two baseline strategies:

- `random-k`, sampled from all admissible cross-subgraph pairs with five deterministic replications.
- `wordnet-ontology-match`, a volume-matched direct lexical matching baseline.

Each baseline proposes exactly `k` edges for each row, where `k` is the number of
scored cross-subgraph connections produced by the corresponding LLM output row after adding its
generated edges to the source and target graphs. This avoids rewarding a method for simply
emitting more raw edges when only a smaller number of source-target connections are scored.

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `row_level_baseline_comparison.csv`
  Auditable row-level comparison with the scored `k`, random-k repetition index, LLM metric,
  baseline metric, and delta.
- `per_model_baseline_summary.csv`
  Manuscript-facing per-model summary. Use this file when asking whether a top LLM can
  outperform a baseline rather than relying only on a cross-model frontier mean.
- `baseline_advantage_summary.csv`
  Cross-model summary by algorithm, baseline strategy, and metric.
- `<algo>_model_vs_baseline.csv`
  Per-model grouped comparison with `baseline_strategy`, `llm_mean`,
  `baseline_mean`, confidence interval columns for the baseline mean, and `mean_delta`.
- `all_models_vs_baseline.csv`
  Combined comparison across all algorithms and baseline strategies.

## Interpretation

A positive `mean_delta` means the LLM outperforms the named non-LLM baseline on that metric.
A negative `mean_delta` means the baseline is more effective.
WordNet should be interpreted as a clean direct lexical matching baseline: it does not propose
new intermediate concept nodes and therefore does not solve the same generative task as the LLM
methods.
