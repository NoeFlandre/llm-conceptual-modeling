# Reviewer Guide

This document is intended for readers who want to verify the repository quickly without first reading the implementation details.

## What This Repository Reproduces

The repository reproduces the deterministic analysis pipeline for the study on LLM variability in conceptual modeling. In practice, this means it can:

- evaluate raw outputs for Algorithms 1 and 2
- recompute evaluated recall outputs for Algorithm 3
- run factorial-analysis post-processing for all three algorithms
- generate a deterministic structural baseline for all three algorithms
- export grouped descriptive statistics and confidence intervals from evaluated CSVs
- classify malformed, empty, and valid raw outputs for failure inspection
- compare imported model outputs against the deterministic baseline
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

Paper-aligned live-backed execution for Method 2:

```bash
uv run lcm generate algo2 \
  --model mistral-small-2603 \
  --embedding-model mistral-embed-2312 \
  --pair sg1_sg2 \
  --output-root /tmp/algo2_runs \
  --json
```

This executable Method 2 path uses the confirmed `0.01` convergence threshold and the curated domain thesaurus tracked in `data/inputs/algo2_thesaurus.json`.
Add `--resume` to any live-backed `generate` or `probe` command to reuse the existing run state and checkpoint files when rerunning a partially completed output directory.

To audit the paper-facing contract in one place, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from llm_conceptual_modeling.cli import main; raise SystemExit(main(['audit', 'paper-alignment', '--json']))"
```

The resulting JSON report lists the method contracts, the confirmed Method 2 `0.01` convergence threshold, the Mistral embedding model, resume support, fixture-backed metric schemas, and probe checkpointing evidence.

Grouped descriptive summaries:

```bash
uv run lcm analyze summary \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --group-by Explanation \
  --metric accuracy \
  --metric recall \
  --output /tmp/algo1_summary.csv
```

Failure classification:

```bash
uv run lcm analyze failures \
  --input tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --result-column Results \
  --output /tmp/algo3_failures.csv
```

Replication-stability summary:

```bash
uv run lcm analyze stability \
  --input data/results/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --group-by Explanation \
  --group-by Example \
  --group-by Counterexample \
  --group-by Array/List\(1/-1\) \
  --group-by Tag/Adjacency\(1/-1\) \
  --group-by Convergence \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo2_stability.csv
```

Paired hypothesis test:

```bash
uv run lcm analyze hypothesis \
  --input data/results/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --factor Explanation \
  --pair-by Repetition \
  --pair-by Example \
  --pair-by Counterexample \
  --pair-by Array/List\(1/-1\) \
  --pair-by Tag/Adjacency\(1/-1\) \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo1_explanation_hypothesis.csv
```

Benjamini-Hochberg correction is used for these outputs because the repository evaluates sets of related factor-level tests rather than one isolated primary test. This keeps the reported signal from being dominated by false positives while remaining less conservative than a familywise-error correction such as Bonferroni.

Figure-ready metric export:

```bash
uv run lcm analyze figures \
  --input data/results/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --id-column Repetition \
  --id-column Explanation \
  --id-column Example \
  --id-column Counterexample \
  --id-column Array/List\(1/-1\) \
  --id-column Tag/Adjacency\(1/-1\) \
  --id-column Convergence \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo2_metric_rows.csv
```

Deterministic structural baseline:

```bash
uv run lcm baseline algo2 \
  --pair sg1_sg2 \
  --output /tmp/algo2_baseline_sg1_sg2.csv
```

Baseline comparison:

```bash
uv run lcm analyze baseline-comparison \
  --baseline-input data/baselines/direct-cross-graph/algo2/evaluated/metrics_sg1_sg2.csv \
  --baseline-input data/baselines/direct-cross-graph/algo2/evaluated/metrics_sg2_sg3.csv \
  --baseline-input data/baselines/direct-cross-graph/algo2/evaluated/metrics_sg3_sg1.csv \
  --input data/results/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input data/results/algo2/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input data/results/algo2/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo2_baseline_comparison.csv
```

The baseline comparison is auditable because it is file-based and deterministic. It should still be interpreted with care: the `direct-cross-graph` baseline uses mother-graph structure directly, so it is a structural heuristic comparator rather than a learned alternative to the LLM workflows.

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
