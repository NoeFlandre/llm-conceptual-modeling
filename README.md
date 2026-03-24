# llm-conceptual-modeling

`llm-conceptual-modeling` is a research software package accompanying the manuscript:

_On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps_.

The repository is publication-focused. Its reproducible surface is the deterministic offline pipeline used to process, analyze, and verify experiment outputs that are already stored in the repository.

## What This Repository Reproduces

Without issuing any new LLM calls, the codebase can:

- evaluate raw outputs for Algorithm 1 and Algorithm 2
- recompute evaluated outputs for Algorithm 3
- run factorial-analysis post-processing for all three algorithms
- generate deterministic structural baselines for all three algorithms
- export grouped descriptive summaries, failure classifications, figure-ready tables, stability summaries, and output-variability summaries
- verify deterministic parity against committed reference fixtures

This repository does not claim runtime equivalence for historical provider behavior.

## Data And Artifacts

- `data/inputs/`
  Input graph files and tracked auxiliary inputs.
- `data/results/`
  Imported primary experiment outputs across algorithms and model families.
- `data/baselines/`
  Deterministic structural baseline outputs.
- `data/analysis_artifacts/`
  Audited offline artifacts supporting revision and analysis findings.
- `tests/fixtures/legacy/`
  Reference fixtures used for deterministic parity checks.

## Quick Start

Install the pinned environment:

```bash
uv sync --dev
```

Run the main verification gate:

```bash
uv run lcm verify all --json
```

Run the local quality gate:

```bash
uv run pytest
uv run ruff check .
uv run ty check
uv run lcm verify all --json
```

The [Makefile](Makefile) mirrors the common tasks through `make test`, `make lint`, `make typecheck`, `make verify`, and `make ci`.

## Core CLI Workflows

Evaluate raw outputs:

```bash
uv run lcm eval algo1 \
  --input tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv \
  --output /tmp/metrics_sg1_sg2.csv

uv run lcm eval algo2 \
  --input tests/fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv \
  --output /tmp/metrics_sg1_sg2.csv

uv run lcm eval algo3 \
  --input tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --output /tmp/method3_results_evaluated_gpt5.csv
```

Run factorial analysis:

```bash
uv run lcm factorial algo1 \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --output /tmp/factorial_analysis_algo1.csv

uv run lcm factorial algo2 \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg2_sg3.csv \
  --input tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg3_sg1.csv \
  --output /tmp/factorial_analysis_algo2.csv

uv run lcm factorial algo3 \
  --input tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv \
  --output /tmp/factorial_analysis_algo3.csv
```

Run offline analysis helpers:

```bash
uv run lcm analyze summary \
  --input tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv \
  --group-by Explanation \
  --metric accuracy \
  --metric recall \
  --metric precision \
  --output /tmp/algo1_summary.csv

uv run lcm analyze failures \
  --input tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --result-column Results \
  --output /tmp/algo3_failures.csv

uv run lcm analyze variability \
  --input data/results/algo3/gpt-5/raw/method3_results_gpt5.csv \
  --group-by Example \
  --group-by Counter-Example \
  --group-by Number\ of\ Words \
  --group-by Depth \
  --group-by Source\ Subgraph\ Name \
  --group-by Target\ Subgraph\ Name \
  --result-column Results \
  --output /tmp/algo3_output_variability.csv
```

Generate deterministic baselines:

```bash
uv run lcm baseline algo1 --pair sg1_sg2 --output /tmp/algo1_baseline.csv
uv run lcm baseline algo2 --pair sg1_sg2 --output /tmp/algo2_baseline.csv
uv run lcm baseline algo3 --pair subgraph_1_to_subgraph_3 --output /tmp/algo3_baseline.csv
```

Verification commands:

```bash
uv run lcm doctor --json
uv run lcm verify legacy-parity --json
uv run lcm verify all --json
```

## Repository Structure

- `src/llm_conceptual_modeling/`
  Package source code.
- `src/llm_conceptual_modeling/algo1`, `algo2`, `algo3`
  Algorithm-specific workflows.
- `src/llm_conceptual_modeling/common`
  Shared evaluation, factorial-analysis, graph, parsing, and schema logic.
- `src/llm_conceptual_modeling/commands`
  CLI handlers.
- `src/llm_conceptual_modeling/analysis`
  Deterministic offline analysis helpers.
- `docs/architecture.md`
  Implementation-oriented overview for contributors.

## Reproducibility Notes

- Use `uv sync` to restore the environment from `uv.lock`.
- The deterministic parity surface is checked by `lcm verify legacy-parity`.
- The repository-wide machine-readable gate is `lcm verify all --json`.
- The local `paper/` directory may exist during manuscript revision, but it is not required to run or verify the software in this repository.

## Citation

If you use this software in academic work, cite the repository and the manuscript using [CITATION.cff](CITATION.cff).

## License

MIT License. See [LICENSE](LICENSE).
