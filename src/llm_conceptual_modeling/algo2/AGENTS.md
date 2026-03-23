# Algo2 Module Guide

This module implements Algorithm 2, the label-expansion plus edge-suggestion workflow.

Purpose:

- Keep the Method 2 pipeline localized here.
- Separate label expansion, edge suggestion, thesaurus normalization, probing, and evaluation.
- Reuse shared helpers from `common` when the logic is not specific to Algorithm 2.

Key files:

- `generation.py`: offline generation manifest for Algorithm 2.
- `baseline.py`: direct cross-subgraph baseline output generation.
- `embeddings.py`: Mistral embedding client and similarity helpers.
- `expansion.py`: iterative label-expansion data structures and control flow.
- `method.py`: Method 2 orchestration and execution result types.
- `factorial.py`: factorial-analysis configuration and dispatch.
- `experiment.py`: experiment-spec construction and batch execution.
- `mistral.py`: Method 2 prompt formatting and Mistral client integration.
- `probe.py`: probe run orchestration and artifact logging.
- `evaluation.py`: evaluation wrapper for Algorithm 2 outputs.
- `thesaurus.py`: thesaurus normalization and edge-term cleanup helpers.

Working rules:

- Keep the embedding, expansion, and edge-suggestion concerns separated.
- Keep thesaurus normalization isolated from prompt construction.
- Move graph, retry, schema, and shared factorial logic into `common` instead of growing this module.
- If `mistral.py`, `probe.py`, or `experiment.py` grows further, split by subworkflow before adding more code.
