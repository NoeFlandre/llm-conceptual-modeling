# llm-conceptual-modeling

`llm-conceptual-modeling` is a research software package for studying how large language models support the conceptual modeling task of combining causal maps.

The repository accompanies the manuscript:

_On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps_.

The current codebase provides a reproducible, testable implementation of the deterministic parts of the study pipeline:

- evaluation of raw outputs for Algorithm 1
- evaluation of raw outputs for Algorithm 2
- recall recomputation for Algorithm 3
- factorial analysis for Algorithms 1, 2, and 3
- repository-level verification commands for parity, smoke checks, and machine-readable reporting

## Research Context

The project studies how prompting choices, model families, model versions, and stochastic variability affect the quality of conceptual-model combination workflows. The task is instantiated on causal maps represented as labeled directed graphs. Across the repository, the term "algorithm" refers to one experimental prompting strategy for generating links or intermediary concepts between graph partitions.

The package is designed so that deterministic outputs can be reproduced locally and checked against committed reference fixtures. This makes the repository suitable for manuscript revision, reviewer inspection, and future extensions where the experimental design must remain auditable.

## What A Reviewer Can Reproduce

Without calling any external LLM provider, a reviewer can:

- rerun the evaluation workflow for each algorithm on committed fixture inputs
- rerun the factorial-analysis post-processing used in the paper
- verify that the produced CSV outputs match committed reference artifacts
- inspect the dataset paths and experiment manifests used by the generation layer
- run repository-wide health and parity checks from the command line

The repository includes the input graph data under [data/inputs](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/inputs) and regression fixtures under [tests/fixtures/legacy](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/fixtures/legacy).

## Scope And Boundaries

This repository fully covers the deterministic offline workflows used to process and analyze experiment outputs.

It does not currently execute live LLM calls as part of the reproducible pipeline. The generation layer is represented as an offline manifest surface that records:

- algorithm name
- required input data
- subgraph-pair coverage
- factorial condition counts
- replication counts
- prompt previews

This means the repository can reproduce and verify the offline analysis pipeline, but it does not claim runtime equivalence for historical provider behavior.

## Quick Start

Install the project and development tools with `uv`:

```bash
uv sync --dev
```

Run the full repository verification gate:

```bash
uv run lcm verify all --json
```

Run the full local quality gate:

```bash
uv run pytest
uv run ruff check .
uv run ty check
uv run lcm verify all --json
```

The [Makefile](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/Makefile) exposes the same common tasks through `make test`, `make lint`, `make typecheck`, `make verify`, and `make ci`.

## Command-Line Interface

The repository exposes a single CLI, `lcm`, for evaluation, factorial analysis, verification, and generation-manifest inspection.

### Evaluate Algorithm Outputs

Algorithm 1:

```bash
uv run lcm eval algo1 \
  --input tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv \
  --output /tmp/metrics_sg1_sg2.csv
```

Algorithm 2:

```bash
uv run lcm eval algo2 \
  --input tests/fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv \
  --output /tmp/metrics_sg1_sg2.csv
```

Algorithm 3:

```bash
uv run lcm eval algo3 \
  --input tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --output /tmp/method3_results_evaluated_gpt5.csv
```

### Run Factorial Analysis

Algorithm 1:

```bash
uv run lcm factorial algo1 \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --output /tmp/factorial_analysis_algo1_gpt_5_without_error.csv
```

Algorithm 2:

```bash
uv run lcm factorial algo2 \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --output /tmp/factorial_analysis_gpt_5_algo2_without_error.csv
```

Algorithm 3:

```bash
uv run lcm factorial algo3 \
  --input tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv \
  --output /tmp/factorial_analysis_results_gpt5_without_error.csv
```

### Verification Commands

Basic repository health:

```bash
uv run lcm doctor --json
```

Deterministic workflow parity:

```bash
uv run lcm verify legacy-parity --json
```

Combined health and parity gate:

```bash
uv run lcm verify all --json
```

### Generation Manifests

The `generate` commands do not call providers. They expose the experimental contract for each algorithm in a machine-readable form:

```bash
uv run lcm generate algo1 --json
uv run lcm generate algo2 --json
uv run lcm generate algo3 --fixture-only --json
```

## Reproducibility And Verification

The repository is designed to make wrongness visible rather than implicit.

- [tests/test_contracts.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_contracts.py) checks full-file parity and CLI verification contracts.
- [tests/test_csv_schemas.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_csv_schemas.py) checks output CSV schemas.
- [tests/test_failures.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_failures.py) covers malformed inputs and failure modes.
- [tests/test_graph_data.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_graph_data.py) checks graph-data and manifest invariants.
- [tests/test_snapshots.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_snapshots.py) checks stable JSON command outputs.

Continuous integration is configured in [.github/workflows/ci.yml](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/.github/workflows/ci.yml).

A concise reviewer workflow is available in [docs/reviewer-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/reviewer-guide.md).

## Repository Structure

- [src/llm_conceptual_modeling](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling): package source code
- [src/llm_conceptual_modeling/algo1](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/algo1), [algo2](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/algo2), [algo3](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/algo3): algorithm-specific workflows
- [src/llm_conceptual_modeling/common](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/common): shared evaluation, factorial-analysis, graph, and schema code
- [src/llm_conceptual_modeling/commands](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/commands): CLI handlers
- [data/inputs](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/inputs): graph input files used by the generation layer
- [tests/fixtures/legacy](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/fixtures/legacy): committed oracle fixtures for regression checks
- [docs/architecture.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/architecture.md): implementation-oriented structure and safety notes
- [docs/reviewer-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/reviewer-guide.md): short reviewer-oriented reproduction guide

## Documentation Notes

The local `paper/` directory may be used during manuscript revision, but it is intentionally ignored by git and is not required to run or verify the software in this repository.
