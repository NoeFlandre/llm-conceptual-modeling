# `hf_pipeline`

Algorithm execution and shared result metrics live here.

## Contents

- `algo1`, `algo2`, `algo3` runtime adapters
- shared structural metrics and result sanitization
- worker-ready helper code used by the `hf_experiments.py` facade

## Maintenance Rule

Keep algorithm execution helpers small and testable. If a helper becomes a
generic orchestration concern, move it to the more specific package.
