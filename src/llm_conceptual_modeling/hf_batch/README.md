# `hf_batch`

Batch-planning and run-artifact helpers for the paper batch live here.

## Contents

- experiment planning and prompt building
- batch monitoring and status snapshots
- run-artifact reading and writing
- output aggregation and utility helpers
- type definitions for batch execution
- `spec_path`: pure spec-identity and run-directory path-resolution helpers

## Maintenance Rule

Keep this package focused on the batch surface itself. Execution, worker, and
resume behavior belong in their own packages.
