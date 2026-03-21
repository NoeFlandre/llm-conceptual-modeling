# Reviewer Guide

This document is intended for readers who want to verify the repository quickly without first reading the implementation details.

## What This Repository Reproduces

The repository reproduces the deterministic analysis pipeline for the study on LLM variability in conceptual modeling. In practice, this means it can:

- evaluate raw outputs for Algorithms 1 and 2
- recompute evaluated recall outputs for Algorithm 3
- run factorial-analysis post-processing for all three algorithms
- verify that these deterministic outputs match committed reference artifacts

## What It Does Not Reproduce Automatically

The repository does not issue live calls to external LLM providers as part of the reproducible workflow.

This is an explicit methodological boundary. The repository preserves the experiment structure and generation manifests, but not runtime-validated historical provider behavior.

## Fastest End-To-End Check

From the repository root:

```bash
uv sync --dev
uv run lcm verify all --json
```

Expected outcome:

- the command exits successfully
- the JSON report has `"status": "ok"`
- the doctor checks report repository health
- the legacy parity checks report six deterministic workflows as `passed`

## Full Local Quality Gate

```bash
uv run pytest
uv run ruff check .
uv run ty check
uv run lcm verify all --json
```

This is the same verification surface enforced in CI, with the addition of the parity gate.

## Representative Workflow Commands

Algorithm 1 evaluation:

```bash
uv run lcm eval algo1 \
  --input tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv \
  --output /tmp/metrics_sg1_sg2.csv
```

Algorithm 3 evaluation:

```bash
uv run lcm eval algo3 \
  --input tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --output /tmp/method3_results_evaluated_gpt5.csv
```

Algorithm 2 factorial analysis:

```bash
uv run lcm factorial algo2 \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --output /tmp/factorial_analysis_gpt_5_algo2_without_error.csv
```

Generation-manifest inspection:

```bash
uv run lcm generate algo1 --json
```

## Files Worth Inspecting

- [README.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/README.md)
  Project overview, command reference, reproducibility notes.
- [docs/architecture.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/architecture.md)
  Internal structure and verification model.
- [tests/fixtures/legacy](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/fixtures/legacy)
  Oracle CSV artifacts used for parity checks.
- [src/llm_conceptual_modeling/verification.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/verification.py)
  Repository-level verification entry points.

## Interpretation Notes

If `verify all` passes, that confirms the deterministic offline workflows remain consistent with the committed reference fixtures in this repository.

It does not confirm that current external LLM providers would reproduce the same raw outputs today. That question depends on provider availability, model drift, and serving behavior outside the scope of this offline package.
