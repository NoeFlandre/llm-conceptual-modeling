# Figure-Export Audit Artifacts

These files support the plot-ready export findings recorded in `paper/revision-tracker.md`.

## Source Data

The source data comes from the imported evaluated result files under `data/results/`.

## Files

- `algo1_metric_rows.csv`
  Tidy row-level metric exports for ALGO1 across all imported models and evaluated files.
- `algo2_metric_rows.csv`
  Tidy row-level metric exports for ALGO2 across all imported models and evaluated files.
- `algo3_metric_rows.csv`
  Tidy row-level metric exports for ALGO3 across all imported models and evaluated files.

## Intended Use

Each file is already in a long format suitable for external plotting tools. The key columns are:

- `source_input`
  Original evaluated CSV path for provenance.
- `algorithm`
  Parsed algorithm identifier from the source path.
- `model`
  Parsed model identifier from the source path.
- factor and repetition columns
  Preserved identifier columns from the original evaluated rows.
- `metric`
  Metric name.
- `value`
  Numeric metric value.
