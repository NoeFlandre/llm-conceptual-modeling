# `hf_state`

Canonical batch-state helpers live here.

## Contents

- active-model discovery
- ledger refresh
- shard-manifest generation
- resume-state and snapshot helpers

## Maintenance Rule

This is the source of truth for persisted batch state. Keep it free of
execution logic and heavy orchestration concerns.
