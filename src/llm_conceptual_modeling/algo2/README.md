# `algo2`

Algorithm 2 implements the iterative label-expansion workflow.

## Main Modules

- `generation.py`: label expansion and edge generation
- `evaluation.py`: post-processing of raw outputs
- `factorial.py`: factorial design helpers
- `baseline.py`: deterministic baseline generation
- `embeddings.py`: embedding and convergence helpers
- `expansion.py`, `thesaurus.py`, `method.py`, `mistral.py`: algorithm-specific wiring

## Maintenance Rule

Keep convergence and expansion logic here. General worker/runtime helpers should
stay in the shared execution or pipeline packages.
