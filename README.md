# llm-conceptual-modeling

`llm-conceptual-modeling` is a research software package for studying how large language models support the conceptual modeling task of combining causal maps.

The repository accompanies the manuscript:

_On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps_.

The current codebase provides a reproducible, testable implementation of the deterministic parts of the study pipeline:

- evaluation of raw outputs for Algorithm 1
- evaluation of raw outputs for Algorithm 2
- recall recomputation for Algorithm 3
- factorial analysis for Algorithms 1, 2, and 3
- deterministic structural baseline generation for Algorithms 1, 2, and 3
- grouped descriptive summaries and confidence-interval exports from evaluated CSVs
- row-level failure classification for raw outputs
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

Imported multi-model experiment outputs used for revision work are stored under [data/results](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results). Audit artifacts that support tracker findings are stored under [data/analysis_artifacts](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/analysis_artifacts).

The repository also maintains a hard split between the frozen pre-revision state and later live-debug work:

- pre-revision frozen tag: `pre-revision-freeze-2026-03-21`
- post-revision debug branch: `post-revision-debug`

Live-provider debugging artifacts are stored under [post_revision_debug](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/analysis_artifacts/post_revision_debug) so they never overwrite the imported historical corpus.

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

## Citation

If you use this software in academic work, please cite both the repository and the accompanying manuscript using `CITATION.cff`:

```yaml
cff-version: 1.2.0
title: "On the variability of generative artificial intelligence methods in conceptual modeling"
message: "If you use this software, please cite the repository and the accompanying manuscript."
authors:
  - family-names: Flandre
    given-names: "Noe Y."
    affiliation: "Old Dominion University (ODU) VMASC; IMT Ales"
  - family-names: Daumas
    given-names: Cedric
    affiliation: "Old Dominion University (ODU) VMASC; IMT Ales"
  - family-names: Giabbanelli
    given-names: "Philippe J."
    affiliation: "Old Dominion University (ODU) VMASC"
repository-code: "https://github.com/NoeFlandre/llm-conceptual-modeling"
license: MIT
```

Full details in [CITATION.cff](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/CITATION.cff).

## License

MIT License. Copyright (c) 2026 Noe Flandre. See the [LICENSE](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/LICENSE) file for details.

## Docker

The project includes a [Dockerfile](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/Dockerfile) for containerized execution:

```bash
# Build the image
docker build -t llm-conceptual-modeling .

# Run the full test suite
docker run --rm llm-conceptual-modeling uv run pytest

# Run the verification gate
docker run --rm llm-conceptual-modeling uv run lcm verify all --json

# Run a specific CLI command
docker run --rm llm-conceptual-modeling uv run lcm doctor --json
```

The image uses `python:3.12-slim` as the base and installs `uv` for fast dependency management.

## Command-Line Interface

The repository exposes a single CLI, `lcm`, for evaluation, factorial analysis, verification, and generation-manifest inspection.

It also exposes `analyze` workflows for reviewer-facing post-processing that do not require any new LLM calls.

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

### Reviewer-Facing Analysis Commands

Grouped descriptive summaries with confidence-interval exports:

```bash
uv run lcm analyze summary \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --group-by Explanation \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo1_summary.csv
```

Raw-output failure classification:

```bash
uv run lcm analyze failures \
  --input tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --result-column Results \
  --output /tmp/algo3_failures.csv
```

Replication-stability summaries across repeated evaluated runs:

```bash
uv run lcm analyze stability \
  --input data/results/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --group-by Explanation \
  --group-by Example \
  --group-by Counterexample \
  --group-by Array/List\\(1/-1\\) \
  --group-by Tag/Adjacency\\(1/-1\\) \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo1_stability.csv
```

Paired factor-level hypothesis tests with Benjamini-Hochberg correction:

```bash
uv run lcm analyze hypothesis \
  --input data/results/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --factor Convergence \
  --pair-by Repetition \
  --pair-by Explanation \
  --pair-by Example \
  --pair-by Counterexample \
  --pair-by Array/List\(1/-1\) \
  --pair-by Tag/Adjacency\(1/-1\) \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo2_convergence_hypothesis.csv
```

The hypothesis workflow uses Benjamini-Hochberg correction because this repository runs families of related factor-level tests across multiple files and metrics. The goal is to limit false discoveries without using a correction so conservative that it erases potentially real effects in this exploratory revision-analysis setting.

Figure-ready long-format metric export:

```bash
uv run lcm analyze figures \
  --input data/results/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv \
  --id-column Repetition \
  --id-column Example \
  --id-column Counter-Example \
  --id-column Number\ of\ Words \
  --id-column Depth \
  --id-column Source\ Subgraph\ Name \
  --id-column Target\ Subgraph\ Name \
  --metric Recall \
  --output /tmp/algo3_metric_rows.csv
```

Deterministic structural baseline generation:

```bash
uv run lcm baseline algo1 --pair sg1_sg2 --output /tmp/algo1_baseline_sg1_sg2.csv
uv run lcm baseline algo2 --pair sg1_sg2 --output /tmp/algo2_baseline_sg1_sg2.csv
uv run lcm baseline algo3 --pair subgraph_1_to_subgraph_3 --output /tmp/algo3_baseline.csv
```

Baseline-vs-model comparison from evaluated CSVs:

```bash
uv run lcm analyze baseline-comparison \
  --baseline-input data/baselines/direct-cross-graph/algo1/evaluated/metrics_sg1_sg2.csv \
  --baseline-input data/baselines/direct-cross-graph/algo1/evaluated/metrics_sg2_sg3.csv \
  --baseline-input data/baselines/direct-cross-graph/algo1/evaluated/metrics_sg3_sg1.csv \
  --input data/results/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input data/results/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input data/results/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo1_baseline_comparison.csv
```

The tracked revision artifacts for this comparison live under [data/analysis_artifacts/revision_tracker/2026-03-21/baseline_comparison](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/analysis_artifacts/revision_tracker/2026-03-21/baseline_comparison). The current baseline is intentionally simple and deterministic: it uses mother-graph structure directly, so it should be read as a graph heuristic comparator rather than as a substitute for live LLM generation.

### Generation Manifests

The `generate` commands expose the experimental contract for each algorithm in a machine-readable form by default. When given explicit runtime arguments, they execute the corresponding live-backed experiment path:

```bash
uv run lcm generate algo1 --json
uv run lcm generate algo2 --json
uv run lcm generate algo3 --fixture-only --json
```

When explicit execution arguments are provided, `generate` also runs a live-backed experiment path for the selected algorithm. The repo keeps the existing Mistral defaults, but it now also supports the paper-faithful OpenRouter-backed model aliases and Method 2 embedding path:

```bash
uv run lcm generate algo1 \
  --model mistral-small-2603 \
  --pair sg1_sg2 \
  --output-root /tmp/algo1_runs \
  --json

uv run lcm generate algo1 \
  --provider openrouter \
  --model paper:gpt-5 \
  --pair sg1_sg2 \
  --output-root /tmp/algo1_runs \
  --json

uv run lcm generate algo2 \
  --model mistral-small-2603 \
  --embedding-provider openrouter \
  --embedding-model paper:text-embedding-3-large \
  --pair sg1_sg2 \
  --output-root /tmp/algo2_runs \
  --json

uv run lcm generate algo3 \
  --model mistral-small-2603 \
  --pair subgraph_1_to_subgraph_3 \
  --output-root /tmp/algo3_runs \
  --json
```

Method 2 uses the confirmed cosine-similarity thresholds `0.01` and `0.02` in the executable path and the tracked domain thesaurus under `data/inputs/algo2_thesaurus.json`. The default embedding backend remains Mistral, but the OpenRouter / `text-embedding-3-large` path is now available when you want the paper-faithful embedding configuration.

The paper model aliases exposed by the CLI are:

- `paper:deepseek-v3-0324` -> `deepseek/deepseek-chat-v3-0324`
- `paper:deepseek-v3.1` -> `deepseek/deepseek-chat-v3.1`
- `paper:gemini-2.0-flash` -> `google/gemini-2.0-flash-001`
- `paper:gemini-2.5-pro` -> `google/gemini-2.5-pro`
- `paper:gpt-4o` -> `openai/gpt-4o-2024-05-13`
- `paper:gpt-5` -> `openai/gpt-5`
- `paper:text-embedding-3-large` -> `text-embedding-3-large`

All executable `generate` and `probe` paths accept `--resume` and write `run.log`, `state.json`, and `execution_checkpoint.json` alongside the existing prompt and summary artifacts. The reusable Mistral chat and embedding clients are now backed by the official `mistralai` SDK, and they retry transient transport failures with exponential backoff before surfacing a hard error, so the failure path is logged and checkpointed rather than silent.

## Post-Revision Debugging

Post-freeze live-provider probing is documented in [post-revision-debug.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/post-revision-debug.md).

The current audited live-debug workflow uses structured Mistral outputs and writes:

- `manifest.json`
- `run.log`
- `state.json`
- `execution_checkpoint.json`
- `events.jsonl`
- per-row prompts
- raw provider responses
- scored CSV summaries

The first canonical run is stored under:

- [representative_matrix_v1](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/representative_matrix_v1)

## Reproducibility And Verification

The repository is designed to make wrongness visible rather than implicit.

- [tests/test_contracts.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_contracts.py) checks full-file parity and CLI verification contracts.
- [tests/test_csv_schemas.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_csv_schemas.py) checks output CSV schemas.
- [tests/test_failures.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_failures.py) covers malformed inputs and failure modes.
- [tests/test_graph_data.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_graph_data.py) checks graph-data and manifest invariants.
- [tests/test_snapshots.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_snapshots.py) checks stable JSON command outputs.
- [tests/test_analysis_summary.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_analysis_summary.py) checks grouped descriptive-statistics exports.
- [tests/test_analysis_failures.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_analysis_failures.py) checks raw-output failure classification.
- [tests/test_analysis_stability.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_analysis_stability.py) checks grouped replication-stability exports.
- [tests/test_analysis_hypothesis.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_analysis_hypothesis.py) checks paired factor-level hypothesis tests and adjusted p-values.
- [tests/test_analysis_figures.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_analysis_figures.py) checks tidy figure-export rows and path-derived metadata.
- [tests/test_baseline.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/test_baseline.py) checks the deterministic structural baseline heuristics and CLI outputs.
- `uv run lcm audit paper-alignment --json` emits a compact machine-readable report covering the paper-facing method contracts, the confirmed Method 2 convergence thresholds (`0.01` and `0.02`), the tracked Mistral embedding model, resume support, fixture-backed metric schemas, and probe checkpointing evidence.

Continuous integration is configured in [.github/workflows/ci.yml](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/.github/workflows/ci.yml).

A concise reviewer workflow is available in [docs/reviewer-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/reviewer-guide.md).

## Repository Structure

- [src/llm_conceptual_modeling](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling): package source code
- [src/llm_conceptual_modeling/algo1](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/algo1), [algo2](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/algo2), [algo3](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/algo3): algorithm-specific workflows
- [src/llm_conceptual_modeling/common](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/common): shared code including the Mistral chat-client (`common/mistral.py`), evaluation, factorial-analysis, graph, and schema utilities
- [src/llm_conceptual_modeling/commands](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/commands): CLI handlers
- [src/llm_conceptual_modeling/cli.py](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/src/llm_conceptual_modeling/cli.py): CLI entry point (`lcm`)
- [data/inputs](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/inputs): graph input files used by the generation layer
- [data/results](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results): imported primary experiment outputs across algorithms and models
- [data/baselines](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/baselines): deterministic structural baseline outputs (graph-heuristic comparators)
- [data/analysis_artifacts](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/analysis_artifacts): audit artifacts backing revision-tracker findings
- [tests/fixtures/legacy](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests/fixtures/legacy): committed oracle fixtures for regression checks
- [tests/](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/tests): full test suite including contracts, CSV schemas, graph data, snapshots, analysis, and baseline tests
- [scripts/](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/scripts): post-revision live-provider debugging tooling
- [docs/architecture.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/architecture.md): implementation-oriented architecture and safety notes
- [docs/algo1-method1-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/algo1-method1-guide.md): exhaustive Method 1 runtime and prompt guide
- [docs/algo2-method2-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/algo2-method2-guide.md): exhaustive Method 2 runtime and prompt guide
- [docs/algo3-method3-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/algo3-method3-guide.md): exhaustive Method 3 runtime and prompt guide
- [docs/reviewer-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/reviewer-guide.md): short reviewer-oriented reproduction guide
- [docs/post-revision-debug.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/post-revision-debug.md): live-provider debugging documentation
- [Dockerfile](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/Dockerfile): container build definition
- [CITATION.cff](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/CITATION.cff): academic citation metadata
- [LICENSE](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/LICENSE): MIT License
- [Makefile](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/Makefile): common task shortcuts
- [uv.lock](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/uv.lock): pinned dependency lock file for reproducible environments

> **Always use `uv sync`** to restore the exact environment from `uv.lock`. Do not use `pip install` directly.

## Documentation

- [docs/architecture.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/architecture.md) — implementation architecture, design decisions, and safety notes for contributors
- [docs/reviewer-guide.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/reviewer-guide.md) — concise reviewer-oriented reproduction guide
- [docs/post-revision-debug.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/post-revision-debug.md) — live-provider probing workflow and artifact layout

## Documentation Notes

The local `paper/` directory may be used during manuscript revision, but it is intentionally ignored by git and is not required to run or verify the software in this repository.
