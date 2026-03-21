# Revision Tracker Audit Artifacts

These files are the concrete analysis artifacts used to support the findings recorded in `paper/revision-tracker.md`.

## Scope

The source data for these artifacts comes from the imported primary result files in `data/results/`, which were copied from the restored legacy project while excluding archives, scripts, caches, and playground backups.

## Files

- `algo1_explanation_directionality.csv`
  File-level mean comparisons for ALGO1 under the `Explanation` factor.
- `algo2_convergence_directionality.csv`
  File-level mean comparisons for ALGO2 under the `Convergence` factor.
- `algo2_explanation_directionality.csv`
  File-level mean comparisons for ALGO2 under the `Explanation` factor.
- `algo3_depth_summary.csv`
  Mean recall summaries for ALGO3 grouped by `Depth`.
- `algo3_number_of_words_summary.csv`
  Mean recall summaries for ALGO3 grouped by `Number of Words`.
- `all_row_level_failures.csv`
  Row-level failure classification across all imported raw result files.
- `failure_counts_by_model.csv`
  Aggregated failure-category counts by algorithm and model.
- `failure_rates_by_model.csv`
  Aggregated failure rates by algorithm and model.
- `parsed_edge_counts_by_model.csv`
  Aggregated parsed edge-count statistics by algorithm and model.

## Audit Use

The intent of this directory is auditability, not polished presentation. The tracker should only claim findings that can be traced back to one or more files in this directory.
