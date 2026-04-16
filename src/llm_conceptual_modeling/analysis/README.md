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
- `variability_bundle.py`, `hypothesis_bundle.py`, `output_validity_bundle.py`,
  `variance_decomposition.py` — bundle assembly and report generation

Sub-packages and extracted helpers:

- `_algo3_stability.py` — algo3-specific pair-aware stability: raw input resolution,
  pair-aware condition frames, level stability/variability aggregation, metric overview rows
- `_stability_helpers.py` — pure helpers: `_slugify`, `frame_to_overview_records`,
  `patch_algorithm_rows`
- `_stability_budget.py` — replication-budget overview record construction for
  stability bundles
- `_variance_decomposition_spec.py` — data model (`ALGORITHM_SPECS`, `AlgorithmSpec`),
  decode helpers (`decode_condition_bits`, `decode_decoding_columns`,
  `coerce_analysis_frame`), and LaTeX table rendering for variance decomposition
- `_variance_decomposition_math.py` — variance-decomposition basis construction,
  orthogonality checks, and sum-of-squares helpers
- `_color_mapping.py` — model-family color maps, canonical label resolution,
  and release-rank ordering for figure rendering
- `_path_helpers.py` — path-triplet extraction from aggregated-directory layouts
  and results-root discovery
- `_plot_frames.py` — plot-frame builders for aggregated distributions,
  factor effects, variability summaries, and main-metric rows
- `_figure_stats.py` — figure-ready long-format melting and distributional
  summary helpers
- `_hypothesis_bundle_helpers.py` — hypothesis-test significance summaries and
  factor-overview aggregation helpers
- `_summary_helpers.py` — statistical summary overview construction and level
  stringification helpers
- `_bundle_stats.py` — pure DataFrame transformers for output-validity and
  breadth bundle assembly (`_build_validity_summary`, `_build_breadth_distribution`,
  `_build_parsed_edge_quartiles`, `_build_failure_rates`, `_build_parsed_edge_counts`,
  `_extract_model`)

## Notes

These modules should stay deterministic and file-driven. If a new analysis needs
shared logic, prefer adding a focused helper in `common/` rather than expanding a
single large file.
