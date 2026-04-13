# `llm_conceptual_modeling`

This package contains the code that reproduces and analyzes the manuscript's
experiments.

## How The Package Is Organized

- `algo1/`, `algo2/`, `algo3/`: algorithm-specific generation, evaluation, and
  factorial-analysis logic.
- `common/`: shared parsing, graph, schema, baseline, and utility code.
- `commands/`: CLI command handlers.
- `hf_batch/`: batch planning, prompts, outputs, monitoring, and artifact helpers.
- `hf_pipeline/`: algorithm execution and shared metric helpers.
- `hf_resume/`, `hf_state/`, `hf_drain/`, `hf_execution/`, `hf_worker/`,
  `hf_tail/`: orchestration, resume, worker, and execution helpers.
- `analysis/`, `verification/`: offline analysis and repository health checks.

## Read First

1. [../../README.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/README.md)
2. [../docs/README.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/README.md)
3. The package README for the area you are editing.

## Maintenance Rule

Keep shared behavior in the smallest package that truly owns it. Leave root-level
modules only as compatibility facades when older imports still need to work.
