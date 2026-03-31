# llm-conceptual-modeling

`llm-conceptual-modeling` is a research software package accompanying the manuscript:

_On the variability of generative artificial intelligence methods in conceptual modeling: an experimental evaluation on combining causal maps_.

The repository is publication-focused. Its reproducible surface is the deterministic offline pipeline used to process, analyze, and verify experiment outputs.

The source code lives in GitHub. The full experiment data payload lives separately in the Hugging Face bucket [`NoeFlandre/llm-variability-conceptual-modeling`](https://huggingface.co/NoeFlandre/llm-variability-conceptual-modeling).

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
  Input graph files and auxiliary lexical resources used by the offline workflows.
  The canonical copy is published in the Hugging Face bucket.
- `data/results/`
  Imported primary experiment outputs across algorithms and model families.
  The canonical copy is published in the Hugging Face bucket.
- `data/baselines/`
  Deterministic structural baseline outputs.
- `data/analysis_artifacts/`
  Audited offline artifacts supporting revision and analysis findings.
  The canonical copy is published in the Hugging Face bucket.
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

## Code/Data Split

Repository roles:

- GitHub repository: code, regression fixtures, tests, and contributor documentation
- Hugging Face bucket: canonical `inputs/`, `results/`, and `analysis_artifacts/`

Bucket repository:

- [NoeFlandre/llm-variability-conceptual-modeling](https://huggingface.co/NoeFlandre/llm-variability-conceptual-modeling)

To point the CLI defaults at a local clone or download of the bucket repository:

```bash
export LCM_INPUTS_ROOT="/path/to/llm-variability-conceptual-modeling/inputs"
export LCM_RESULTS_ROOT="/path/to/llm-variability-conceptual-modeling/results"
export LCM_ANALYSIS_ARTIFACTS_ROOT="/path/to/llm-variability-conceptual-modeling/analysis_artifacts"
```

You can also override paths per command with `--results-root` and explicit `--output-dir`.

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
  --input "$LCM_RESULTS_ROOT/algo3/gpt-5/raw/method3_results_gpt5.csv" \
  --group-by Example \
  --group-by Counter-Example \
  --group-by Number\ of\ Words \
  --group-by Depth \
  --group-by Source\ Subgraph\ Name \
  --group-by Target\ Subgraph\ Name \
  --result-column Results \
  --output /tmp/algo3_output_variability.csv

uv run lcm analyze replication-budget \
  --input data/analysis_artifacts/revision_tracker/replication_stability/algo1_condition_stability.csv \
  --output /tmp/algo1_replication_budget.csv
```

Generate deterministic baselines:

```bash
uv run lcm baseline algo1 --pair sg1_sg2 --output /tmp/algo1_baseline.csv
uv run lcm baseline algo2 --pair sg1_sg2 --output /tmp/algo2_baseline.csv
uv run lcm baseline algo3 --pair subgraph_1_to_subgraph_3 --output /tmp/algo3_baseline.csv

uv run lcm baseline algo1 \
  --pair sg1_sg2 \
  --strategy wordnet-ontology-match \
  --output /tmp/algo1_wordnet_baseline.csv

uv run lcm baseline algo1 \
  --pair sg1_sg2 \
  --strategy edit-distance \
  --output /tmp/algo1_edit_distance_baseline.csv
```

Available baseline strategies:

- `random-uniform-subset`
  Sample 20 cross-subgraph node pairs uniformly from all possible source-target combinations.
- `direct-cross-graph`
  Return the true cross-subgraph edges present in the mother graph.
- `wordnet-ontology-match`
  Rank cross-subgraph node pairs by overlap in the tracked WordNet-derived label lexicon.
- `edit-distance`
  Rank cross-subgraph node pairs by normalized label similarity.

Verification commands:

```bash
uv run lcm doctor --json
uv run lcm verify legacy-parity --json
uv run lcm verify all --json
```

Validate and preview the checked-in HF run configuration before renting a GPU:

```bash
uv run lcm run validate-config \
  --config configs/hf_transformers_paper_batch.yaml \
  --output-dir /tmp/hf-paper-batch-preview
```

Run the new local-`transformers` GPU batch:

```bash
uv run lcm run paper-batch \
  --config configs/hf_transformers_paper_batch.yaml \
  --resume

uv run lcm analyze plots \
  --results-root /workspace/results/hf-paper-batch \
  --output-dir /workspace/results/hf-paper-batch/plots
```

The YAML file is the execution source of truth for:

- chat models and embedding model
- decoding algorithms and their parameters
- temperature, seed, per-schema token budgets, and context-window safety margin
- algorithm prompt fragments and DOE-controlled optional prompt elements
- output root and replication count

Remote GPU workflow:

- bootstrap and CUDA verification: [docs/vast-ai-transformers.md](docs/vast-ai-transformers.md)
- helper scripts:
  - `scripts/vast/bootstrap_gpu_host.sh`
  - `scripts/vast/sync_repo_to_vast.sh`
  - `scripts/vast/fetch_results_from_vast.sh`

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
- Commands that need externalized data should use `LCM_INPUTS_ROOT`, `LCM_RESULTS_ROOT`, and `LCM_ANALYSIS_ARTIFACTS_ROOT` when the bucket contents are stored outside the repository checkout.
- The local `paper/` directory may exist during manuscript revision, but it is not required to run or verify the software in this repository.

## Citation

If you use this software in academic work, cite the repository and the manuscript using [CITATION.cff](CITATION.cff).

## License

MIT License. See [LICENSE](LICENSE).
