# Open-Weight Sweep Canonical Results

This directory is the canonical local mirror of the full open-weight Qwen and
Mistral sweep used in the paper-facing ablation analyses.

It contains:

- `runs/`
  - per-run artifacts for Algorithms 1, 2, and 3 across the finished sweep
- `aggregated/`
  - grouped raw/evaluated/factorial/stability exports
- `variance_decomposition/`
  - the maintained sweep-level variance-decomposition bundle
- `ledger.json`
  - canonical finished-run ledger used by audit and resume tooling
- `batch_summary.csv`
  - consolidated per-run summary rows

Use this tree for the full sweep. Use
`data/results/open_weights/hf-map-extension-canonical/` for the later
three-map extension batch.
