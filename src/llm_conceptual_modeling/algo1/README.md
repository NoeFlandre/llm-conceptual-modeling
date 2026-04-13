# `algo1`

Algorithm 1 implements the direct edge-linking workflow.

## Main Modules

- `generation.py`: prompt assembly and runtime execution
- `evaluation.py`: post-processing of raw outputs
- `factorial.py`: factorial design helpers
- `baseline.py`: deterministic baseline generation
- `cove.py`: verification prompt logic
- `method.py` and `mistral.py`: algorithm-specific wiring

## Maintenance Rule

Keep Method 1-specific logic here. Shared parsing, codecs, and result handling
belong in `common/` or the batch/execution packages.
