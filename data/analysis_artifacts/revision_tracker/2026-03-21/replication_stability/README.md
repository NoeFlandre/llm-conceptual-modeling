# Replication Stability Audit Artifacts

These files support the replication-stability findings recorded in `paper/revision-tracker.md`.

## Source Data

The source data comes from the imported evaluated result files under `data/results/`:

- `data/results/algo1/*/evaluated/*.csv`
- `data/results/algo2/*/evaluated/*.csv`
- `data/results/algo3/*/evaluated/*.csv`

## Files

- `algo1_condition_stability.csv`
  Per-file, per-condition stability statistics for ALGO1 across the five repetitions.
- `algo1_explanation_stability_by_level.csv`
  Aggregated ALGO1 stability summaries grouped by `Explanation`.
- `algo2_condition_stability.csv`
  Per-file, per-condition stability statistics for ALGO2 across the five repetitions.
- `algo2_convergence_stability_by_level.csv`
  Aggregated ALGO2 stability summaries grouped by `Convergence`.
- `algo2_convergence_variability_incidence.csv`
  Incidence of any run-to-run variation in ALGO2 grouped by `Convergence`.
- `algo2_explanation_stability_by_level.csv`
  Aggregated ALGO2 stability summaries grouped by `Explanation`.
- `algo3_condition_stability.csv`
  Per-file, per-condition stability statistics for ALGO3 across the five repetitions.
- `algo3_depth_stability_by_level.csv`
  Aggregated ALGO3 stability summaries grouped by `Depth`.
- `algo3_depth_variability_incidence.csv`
  Incidence of any run-to-run variation in ALGO3 grouped by `Depth`.
- `algo3_number_of_words_stability_by_level.csv`
  Aggregated ALGO3 stability summaries grouped by `Number of Words`.
- `overall_metric_stability_by_algorithm.csv`
  Cross-algorithm summary of coefficient of variation and range width by metric.
- `variability_incidence_by_algorithm.csv`
  Cross-algorithm count and share of conditions that changed across repetitions.

## Interpretation Notes

These artifacts are descriptive and deterministic. They do not justify a replication count by themselves, but they do show how much the observed evaluated metrics moved across the five recorded repetitions for each condition.
