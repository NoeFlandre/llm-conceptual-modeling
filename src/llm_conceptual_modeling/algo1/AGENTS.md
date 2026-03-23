# Algo1 Module Guide

This module implements Algorithm 1, the direct edge-linking workflow with CoVe verification.

Purpose:

- Keep the Method 1 pipeline localized here.
- Separate direct edge generation, CoVe verification, probing, evaluation, and factorial post-processing.
- Reuse shared helpers from `common` when the logic is not specific to Algorithm 1.

Key files:

- `generation.py`: offline generation manifest for Algorithm 1.
- `baseline.py`: direct cross-subgraph baseline results.
- `cove.py`: chain-of-verification prompt and vote filtering helpers.
- `method.py`: Method 1 orchestration and execution result types.
- `factorial.py`: factorial-analysis configuration and dispatch.
- `experiment.py`: experiment-spec construction and batch execution.
- `mistral.py`: Method 1 prompt formatting and Mistral client integration.
- `probe.py`: probe run orchestration and artifact logging.
- `evaluation.py`: evaluation wrapper for Algorithm 1 outputs.

Working rules:

- Keep the direct generation path and the verification path separate.
- Keep CoVe logic isolated from generic CLI orchestration.
- Move graph, retry, schema, and shared factorial logic into `common` instead of growing this module.
- If `mistral.py`, `probe.py`, or `experiment.py` grows further, split by subworkflow before adding more code.
