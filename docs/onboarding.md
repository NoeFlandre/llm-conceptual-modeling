# Codebase Onboarding

This is the primary operating manual for the repository.

If you are new here, read this file first. It explains what the codebase does,
how the modules are organized, how to verify changes, how to keep the tree
maintainable, and which habits matter most when you touch code, docs, tests, or
data artifacts.

## What This Repository Is For

`llm-conceptual-modeling` is the reproducible software companion to the
manuscript on variability in generative AI methods for conceptual modeling.

The repository is used to:

- evaluate experiment outputs for Algorithms 1, 2, and 3
- generate deterministic baselines and factorial summaries
- run offline analysis, summaries, figures, hypothesis tests, and stability checks
- support batch planning, result syncing, resume, and worker orchestration for
  the paper-facing Hugging Face runs
- preserve archived operational and frontier-model outputs without letting them
  pollute the main paper-facing result tree

The source code lives in GitHub. The canonical experiment payload lives in the
Hugging Face bucket:

- [`NoeFlandre/llm-variability-conceptual-modeling`](https://huggingface.co/NoeFlandre/llm-variability-conceptual-modeling)

## Core Mental Model

Think of the project in four layers:

1. `src/llm_conceptual_modeling/`
   - executable code, split by concern
2. `data/`
   - tracked inputs, paper-facing outputs, archives, and deterministic artifacts
3. `docs/`
   - long-form guides, operating rules, and maintainability notes
4. `tests/`
   - executable contracts that keep the repo reproducible

The source tree is intentionally organized into small packages instead of one
flat pile of scripts:

- `algo1/`, `algo2/`, `algo3/`
  - algorithm-specific workflows and analysis helpers
- `analysis/`
  - deterministic offline analysis, summaries, plots, and tables
- `commands/`
  - CLI entry points
- `common/`
  - shared parsing, schemas, I/O, type helpers, and low-level utilities
- `hf_batch/`, `hf_state/`, `hf_resume/`, `hf_drain/`, `hf_execution/`,
  `hf_worker/`, `hf_tail/`, `hf_config/`, `hf_pipeline/`
  - batch planning, canonical state, resume logic, drain logic, execution,
    worker behavior, final-tail handling, runtime config, and algorithm
    execution helpers
- `verification/`
  - health checks and parity checks

Each major package directory has a local `README.md`. That local README should
be the first stop before editing the package internals.

## What The Codebase Reproduces

Without issuing new LLM calls, the repository can:

- evaluate raw outputs for Algorithms 1 and 2
- recompute evaluated outputs for Algorithm 3
- run deterministic factorial-analysis post-processing for all three algorithms
- generate deterministic structural baselines
- export summaries, failure classifications, figure-ready tables, stability
  summaries, and output-variability summaries
- generate variance decomposition outputs for the open-weight ablation study
- verify deterministic parity against committed fixtures and maintained results

The repository does **not** promise historical provider replay equivalence. It
promises deterministic, externally checkable behavior for the maintained data
and workflows.

## Canonical Data And Results Layout

The tracked data tree is the source of truth for local development:

- `data/inputs/`
  - canonical input payloads and resources
- `data/results/frontier/`
  - archived frontier-model experiment outputs
- `data/results/open_weights/`
  - canonical Qwen/Mistral paper-facing outputs
- `data/results/archives/`
  - preserved OLMO, operational, and stale-shard archives
- `data/analysis_artifacts/`
  - revision, stability, and analysis artifacts
- `data/baselines/`
  - deterministic baseline artifacts

Paper-facing open-weight results live under:

- `data/results/open_weights/hf-paper-batch-canonical/`

That canonical tree contains:

- `ledger.json`
- `batch_status.json`
- `runtime_config.yaml`
- run directories under `runs/`
- variance-decomposition outputs under `variance_decomposition/`

When reasoning about completion state, prefer the ledger and the batch metadata
over the physical directory count alone.

Archive buckets are for preservation, not active use:

- frontier experiment outputs go in `data/results/frontier/`
- OLMO and old operational trees go in `data/results/archives/`

The goal is to keep the paper-facing tree clean and easy to navigate while
preserving provenance where it matters.

## How To Work Safely

Use these rules on every change:

- Run the baseline tests first.
- Prefer red/green TDD for code changes.
- Keep the smallest correct diff that solves the actual problem.
- Do not guess about semantics that can be checked from the code or fixtures.
- Make wrongness visible through tests, fixtures, schemas, and CLI checks.
- Delete dead code, stale docs, obsolete scripts, and duplicate paths.
- Prefer package-local READMEs over a single giant architecture note.
- Keep files small and focused. If a file becomes a grab-bag, split it.
- Treat every inference as provisional until you verify it.

The repository has been shaped by repeated operational work. That means there are
often several similar files or helper paths. Before you add another helper,
check whether an existing package already owns that responsibility.

## Standard Development Loop

1. Inspect the relevant code, docs, and tests.
2. Run the narrowest existing verification command.
3. Add or update a focused failing test.
4. Implement the smallest correct fix.
5. Run the focused tests again.
6. Update docs when behavior, layout, or operating assumptions change.
7. Run the broader gate before handoff.

This repository is easy to damage with broad refactors. Keep the loop short and
keep the verification local.

## Verification Workflow

Use the repository’s normal toolchain:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ty check
uv run lcm verify all --json
```

Prefer the narrowest useful command first, then expand outward:

- a focused unit test before a full suite
- a CLI smoke check before a broad doctor run
- a fixture-based regression before a live run

If a task touches analysis outputs, verify both the machine-readable artifact
and the relevant regression test. If it touches orchestration, verify the
runtime path, the ledger/state path, and the result snapshot together.

## Repository Standards

The repo should stay:

- reproducible
- deterministic where possible
- documented close to the code
- split by concern into small modules
- easy to verify locally
- free of stale helpers and duplicated logic

The standard toolchain is:

- `uv` for environment and dependency management
- `ruff` for linting and formatting
- `ty` for type checking
- `pytest` for tests

Keep dependencies minimal. If a dependency is only used by one abandoned path,
remove it.

## Source Tree Guide

The package boundaries exist for a reason. Use the local README in each package
before you edit internals.

- `src/llm_conceptual_modeling/algo1/`
  - Algorithm 1 evaluation, generation, factorial analysis, and baseline logic
- `src/llm_conceptual_modeling/algo2/`
  - Algorithm 2 evaluation, generation, factorial analysis, embeddings, and
    baseline logic
- `src/llm_conceptual_modeling/algo3/`
  - Algorithm 3 tree-expansion, evaluation, generation, and factorial logic
- `src/llm_conceptual_modeling/analysis/`
  - summaries, figures, stability, variability, failure categorization,
    variance decomposition, and paper-facing tables
- `src/llm_conceptual_modeling/commands/`
  - CLI wiring and command grouping
- `src/llm_conceptual_modeling/common/`
  - core helpers shared by the rest of the repo
- `src/llm_conceptual_modeling/hf_batch/`
  - batch planning, artifact handling, monitoring, types, and prompts
- `src/llm_conceptual_modeling/hf_state/`
  - batch state, ledger, resume state, and shard-manifest helpers
- `src/llm_conceptual_modeling/hf_resume/`
  - user-facing resume wrappers and preflight checks
- `src/llm_conceptual_modeling/hf_drain/`
  - remaining-run drain planning and supervisor logic
- `src/llm_conceptual_modeling/hf_execution/`
  - subprocess execution helpers and runtime orchestration
- `src/llm_conceptual_modeling/hf_worker/`
  - worker request, policy, state, result, entrypoint, and persistent-worker
    logic
- `src/llm_conceptual_modeling/hf_tail/`
  - final-tail preparation and preflight helpers
- `src/llm_conceptual_modeling/hf_config/`
  - runtime configuration helpers
- `src/llm_conceptual_modeling/hf_pipeline/`
  - shared algorithm execution and metrics helpers
- `src/llm_conceptual_modeling/verification/`
  - repo health and parity checks

Flat compatibility modules still exist where needed, but they should remain
thin. When you see a large root-level module, treat it as a candidate for a
package split.

## Refactoring Rules

When you are improving code structure:

- split by responsibility, not by arbitrary file count
- keep public facades thin and stable
- move logic into focused package modules
- remove dead wrappers after callers are repointed
- prefer one small package with a clear README over one giant “utils” file
- keep line count low per file, but do not create tiny files that obscure the
  flow of responsibility

The target is not maximal fragmentation. The target is clear ownership:

- one package boundary for one coherent concern
- one file for one focused job when practical
- one public import surface per concern

If a refactor adds more files, keep them grouped in subfolders and document the
new boundaries immediately.

## Documentation Rules

Use this order when looking for information:

1. `README.md`
2. `docs/onboarding.md`
3. the package-local `README.md`
4. the narrow implementation file
5. the matching test

Keep docs close to the code they describe. Prefer short package READMEs for
local orientation and reserve the onboarding guide for cross-cutting rules and
operating knowledge.

Do not keep long-lived revision notes once the stable package README or the
onboarding guide covers the same ground. Delete or archive stale docs instead of
letting them accumulate.

## Remote Execution Lessons

The remote batch and tail workflows produced a few stable lessons:

- `current_run` is a claim signal, not proof of useful progress
- `batch_status.json`, watcher status, and `ledger.json` are different views and
  can temporarily disagree
- a worker may look idle while it is still loading a model
- model-switch churn can dominate time if the queue is not ordered carefully
- a parser bug should become a regression test before it becomes a one-off fix
- remote deployment should be hash-verified before trusting the host
- if a Qwen download or startup stalls, inspect the runtime and logs before
  changing the algorithm logic

For remote GPU work, the stable sequence is:

1. sync the repository
2. verify the runtime environment
3. validate the results root
4. launch the batch or tail workflow
5. watch the status, logs, and ledger
6. finalize and sync locally

The main operational rule is: do not reason from one artifact in isolation.
Use the ledger, batch status, logs, and run directories together.

## Results Hygiene

The maintained result tree should stay clean and understandable:

- keep paper-facing open-weight results in `data/results/open_weights/`
- keep imported frontier outputs in `data/results/frontier/`
- keep archives in `data/results/archives/`
- keep scratch files, previews, and operational leftovers out of the maintained
  tree

Do not delete provenance-bearing artifacts unless you are sure they are
redundant. If a result is archived, preserve the files needed to explain it:

- `ledger.json`
- `batch_status.json`
- `runtime_config.yaml`
- run logs
- run summaries
- sync logs

If a folder is only an operational checkpoint or temporary workdir, archive it
or remove it once you have verified that no finished result depends on it.

## Quality Expectations

The codebase should remain:

- maintainable by a fresh contributor
- easy to navigate from package README to implementation
- explicit about what is canonical and what is archive
- mechanically verifiable by tests and CLI commands
- free of stale experiment-only clutter in tracked docs or source files

When you find a bug, treat it as a documentation opportunity too. Add the
regression test, fix the code, and update the relevant guide so the next person
does not rediscover the same issue.

## Future Work: HTML Causal Maps And New Experiments

The next likely expansion is HTML causal-map ingestion plus more Qwen and
Mistral experiments over the existing decoding strategies and design-of-
experiments surface.

Before adding that work:

- define the input contract in `data/inputs/`
- add a package-local README for the new module boundary
- add a narrow failing test before implementing the feature
- keep the new code path aligned with the existing deterministic analysis and
  verification conventions
- avoid reintroducing a flat helper dump under `src/`

The point is to make future experiments easy to add without turning the repo
back into an unreadable script pile.
