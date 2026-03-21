# Post-Revision Debug Artifacts

This directory contains exploratory and post-freeze artifacts created after the repository was frozen for pre-revision reproducibility.

## Versioning Policy

- The frozen pre-revision repository state is tagged as `pre-revision-freeze-2026-03-21`.
- Post-freeze debugging and live-provider experiments run on branch `post-revision-debug`.
- Imported historical results remain under `data/results/`.
- New live-provider probes and their audit artifacts belong under `data/analysis_artifacts/post_revision_debug/`.

## Layout

- `mistral/2026-03-21/ad_hoc/`
  One-off exploratory calls used to validate model IDs, output formats, and individual row behavior.
- `mistral/2026-03-21/representative_matrix_v1/`
  The first canonical audited matrix run over representative ALGO1, ALGO2, and ALGO3 rows.

## Logging And Audit Files

Canonical runs are expected to produce:

- `manifest.json`
  The exact probe configuration and model list.
- `run.log`
  Human-readable execution log.
- `events.jsonl`
  Structured event stream for each run.
- `prompt.txt`
  The prompt snapshot for each probe row.
- `*_response.json`
  Raw provider responses.
- `probe_results.csv`
  Row-level scored outputs against the historical imported rows.
- `probe_summary_by_algorithm_and_model.csv`
  Aggregated score deltas.
- `findings.md`
  Human-readable summary of the run.
