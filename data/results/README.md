# Results Layout

This directory is split into three root buckets:

- `frontier/`
  - frontier-model experiment outputs imported from `data/results`
- `open_weights/`
  - paper-facing Qwen/Mistral outputs and the canonical finished-run ledger
- `archives/`
  - preserved operational workdirs, OLMO artifacts, and stale shard trees

## What Belongs Where

### `frontier/`
- `algo1/`
- `algo2/`
- `algo3/`

Each algorithm directory contains the frontier model family outputs, grouped by model name and kept out of the code-data tree.

### `open_weights/`
- `hf-paper-batch-canonical/`
  - canonical finished-run ledger and paper-facing outputs
  - `runs/` artifacts for the finalized Qwen/Mistral study
  - `variance_decomposition/` outputs for the paper

### `archives/`
- `olmo/`
  - OLMO batch workdirs and smoke-run artifacts kept for reference
- `operational/`
  - stale `*-current` workdirs and canonical batch leftovers
  - sync logs and other provenance-bearing runtime files
- `stale-shards/`
  - older shard layouts and drained shard trees

## Cleanup Policy

- Keep finished run artifacts, ledgers, runtime configs, and paper outputs.
- Remove or archive operational clutter such as:
  - `.DS_Store`
  - `*.pid`
  - `preview/` and `preview_resume/`
  - `worker-queues/`
  - scratch/debug files that do not explain a finished result
- Failed-only trees may be archived for provenance, but they should not remain mixed into the top-level results layout.

The goal is that someone can open `results/`, immediately find frontier outputs, the open-weight paper outputs, and then browse `archives/` only when operational history is needed.
