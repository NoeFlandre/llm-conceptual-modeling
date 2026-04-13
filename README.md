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
- `data/results/frontier/`
  Imported frontier-model experiment outputs grouped by algorithm.
- `data/results/open_weights/`
  Canonical Qwen/Mistral paper-facing outputs and the variance-decomposition bundle.
- `data/results/archives/`
  Preserved OLMO artifacts, stale batch workdirs, and drained shard trees.
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

Variance decomposition for the open-weight ablation study:

```bash
uv run python generate_variance_decomposition.py
```

This recomputes the deterministic Qwen/Mistral variance-decomposition bundle from
the canonical finished-run ledger and writes the maintained artifacts under:

- `data/results/open_weights/hf-paper-batch-canonical/variance_decomposition/`

That folder contains:

- `variance_decomposition.csv`
- `variance_decomposition_algo1.csv`
- `variance_decomposition_algo2.csv`
- `variance_decomposition_algo3.csv`
- `variance_decomposition.tex`
- `variance_decomposition_algo1.tex`
- `variance_decomposition_algo2.tex`
- `variance_decomposition_algo3.tex`

Method and artifact details are documented in
[docs/variance-decomposition.md](docs/variance-decomposition.md).

Local experiment-result hygiene is documented in [data/results/README.md](data/results/README.md).
In short: keep frontier outputs under `data/results/frontier/`, open-weight paper outputs under
`data/results/open_weights/`, archive old operational workdirs under `data/results/archives/`, and
leave the top-level `results/` folder free of maintained outputs.
The source tree itself should also stay free of Finder junk (`.DS_Store`) and other ad hoc
temporary artifacts; keep scratch work isolated under `tmp/` or a throwaway worktree instead of
committing it into the repo layout.
On macOS, Finder may recreate local `.DS_Store` files transiently; the hygiene test guards against
tracked occurrences in the maintained source buckets.
One-off helper scripts should also stay out of the repo root unless they are part of the
documented CLI or `scripts/` surface.
Accidental copied worktrees or SSH-named mirror directories belong outside the repository root,
not alongside the maintained source tree.

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

uv run lcm run status \
  --results-root /workspace/results/hf-paper-batch \
  --json
```

The YAML file is the execution source of truth for:

- chat models and embedding model
- decoding algorithms and their parameters
- per-model thinking-mode control declarations
- temperature, seed, per-schema initial generation budgets, and context-window safety margin
- algorithm prompt fragments and DOE-controlled optional prompt elements
- output root and replication count

You can also inspect the HF planning surface through the existing `generate` command by pointing it
at the same config:

```bash
uv run lcm generate algo1 \
  --provider hf-transformers \
  --config configs/hf_transformers_paper_batch.yaml \
  --json
```

The HF batch writes both per-decoding-condition outputs and combined model-level outputs under
`aggregated/<algorithm>/<model>/combined/`. The combined surface is the one that now carries the
decoding-factor DOE analysis and the explicit residual `Error` row in `factorial.csv`.

Remote GPU workflow:

- bootstrap and CUDA verification: [docs/vast-ai-transformers.md](docs/vast-ai-transformers.md)
- long-form operator handoff and failure playbook: [qwen-mistral-remote-ops-handoff.md](qwen-mistral-remote-ops-handoff.md)
- helper scripts:
  - `scripts/vast/bootstrap_gpu_host.sh`
  - `scripts/vast/sync_repo_to_vast.sh`
  - `scripts/vast/fetch_results_from_vast.sh`
  - `scripts/vast/prepare_and_resume_qwen_algo1_tail.sh`
  - `scripts/vast/finalize_qwen_algo1_tail.sh`

Dedicated last-10 Qwen tail workflow:

- The final Qwen-only `algo1` tail has a dedicated isolated workflow and should not be run through the generic multi-model resume path.
- The source of truth is the canonical `data/results/open_weights/hf-paper-batch-canonical/ledger.json`
  records surface, not the physical run tree.
- Use the dedicated local prep commands first:

```bash
uv run lcm run prepare-qwen-algo1-tail \
  --canonical-results-root /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical \
  --tail-results-root /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  --remote-output-root /workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail \
  --json

uv run lcm run qwen-algo1-tail-preflight \
  --repo-root /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  --canonical-results-root /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical \
  --tail-results-root /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  --json
```

- The dedicated preflight is expected to resolve to exactly `10` pending Qwen identities and `can_resume=true`.
- The dedicated remote launcher is:

```bash
bash scripts/vast/prepare_and_resume_qwen_algo1_tail.sh \
  root@HOST \
  PORT \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /workspace/llm-conceptual-modeling \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical \
  /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  /workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail
```

- If the fresh host stalls in cold model download, inspect `/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/blobs/*.incomplete` and the Xet logs before touching parser/runtime code.
- If the `.incomplete` files stop growing and Xet logs stop advancing, relaunch the dedicated tail with:
  - `HF_HUB_DISABLE_XET=1`
  - `HF_HOME=/workspace/.hf_home`
- After the remote tail reaches `10 finished / 0 pending / 0 failed`, pull the isolated root locally and finalize with:

```bash
bash scripts/vast/finalize_qwen_algo1_tail.sh \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /private/tmp/qwen-tail-live/hf-paper-batch-qwen-algo1-tail \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/data/results/open_weights/hf-paper-batch-canonical
```

- The final correctness check is not the whole-study top-line `pending_count`. It is:
  - isolated tail `batch_status.json` shows `10 finished / 0 pending`
  - canonical run dirs for the 10 target identities have `state.json = finished`
  - canonical `ledger.json.records` marks those 10 identities as `finished`

## Repository Structure

- `src/llm_conceptual_modeling/`
  Package source code.
- `src/llm_conceptual_modeling/algo1`, `algo2`, `algo3`
  Algorithm-specific workflows.
- `src/llm_conceptual_modeling/common`
  Shared evaluation, factorial-analysis, graph, parsing, and schema logic.
- `src/llm_conceptual_modeling/hf_pipeline`
  Algorithm execution, shared result metrics, and worker-ready helpers used by the legacy `hf_experiments.py` facade.
- `src/llm_conceptual_modeling/hf_batch`
  Batch-planning, prompt-building, outputs, monitoring, run-artifact, and type helpers grouped under one namespace.
- `src/llm_conceptual_modeling/hf_state`
  Canonical batch-state helpers for active model discovery, shard-manifest generation, and ledger refresh.
- `src/llm_conceptual_modeling/hf_drain`
  Drain planning and runtime supervisor logic for the remaining-results sweep.
- `src/llm_conceptual_modeling/hf_execution`
  Local worker execution helpers, subprocess fallback, and persistent-session orchestration.
- `src/llm_conceptual_modeling/verification`
  Doctor, legacy-parity, and repository verification logic.
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
