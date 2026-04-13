# `algo3`

Algorithm 3 implements the tree-expansion workflow.

## Main Modules

- `generation.py`: prompt assembly and tree expansion
- `evaluation.py`: post-processing of raw outputs
- `factorial.py`: factorial design helpers
- `baseline.py`: deterministic baseline generation
- `method.py`, `mistral.py`, `tree.py`: algorithm-specific runtime logic

## Maintenance Rule

Keep tree traversal and recall-specific logic here. Shared execution, resume,
and worker code should remain in the dedicated orchestration packages.
