# Algo3 Module Guide

This module implements Algorithm 3, the tree-structured label-expansion workflow.

Purpose:

- Build and run the Method 3 experiment pipeline.
- Keep tree expansion, prompt construction, probing, evaluation, and factorial post-processing local to Algorithm 3.
- Reuse shared primitives from `common` rather than duplicating them here.

Key files:

- `generation.py`: offline generation manifest for Algorithm 3.
- `baseline.py`: structural baseline results for comparison runs.
- `tree.py`: tree expansion data structures and traversal logic.
- `method.py`: orchestration helpers that execute Method 3.
- `factorial.py`: factorial-analysis helpers for Algorithm 3 outputs.
- `experiment.py`: experiment-spec construction and batch execution.
- `mistral.py`: Method 3 prompt formatting and Mistral client integration.
- `probe.py`: probe run orchestration and artifact logging.
- `evaluation.py`: raw output evaluation and normalization.

Working rules:

- Keep the tree model, prompt construction, and execution flow separated.
- Move shared retry, schema, graph, or baseline logic into `common` instead of growing this module.
- If `mistral.py`, `probe.py`, or `evaluation.py` grows too large, split by concern before adding more logic.
- Keep the module aligned with the existing Algorithm 3 terminology so future agents can map code to paper concepts quickly.
