# Revision Tracker Audit Artifacts

These files are the concrete analysis artifacts used to support the findings recorded in `paper/revision-tracker.md`.

## Scope

The source data for these artifacts comes from the imported primary result files in `data/results/`, which were copied from the restored legacy project while excluding archives, scripts, caches, and playground backups.

## Files

- `statistical_reporting/`
  Organized descriptive-statistics bundle for the statistical-reporting revision item.
  It contains per-algorithm, per-factor grouped summaries plus compact metric overviews and
  bundle-level manifest and overview files.
- `all_row_level_failures.csv`
  Row-level failure classification across all imported raw result files.
- `failure_counts_by_model.csv`
  Aggregated failure-category counts by algorithm and model.
- `failure_rates_by_model.csv`
  Aggregated failure rates by algorithm and model.
- `parsed_edge_counts_by_model.csv`
  Aggregated parsed edge-count statistics by algorithm and model.
- `replication_stability/`
  Stability artifacts derived from evaluated CSVs to quantify how much metrics moved across the recorded repetitions.
- `hypothesis_testing/`
  Audited paired-test outputs and adjusted p-value summaries for selected factor comparisons on the imported evaluated corpus.
- `figure_exports/`
  Tidy long-format metric exports for external plotting across the imported evaluated corpus.

## Audit Use

The intent of this directory is auditability, not polished presentation. The tracker should only claim findings that can be traced back to one or more files in this directory.
