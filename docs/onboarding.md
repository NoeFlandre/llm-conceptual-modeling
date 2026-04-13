# Codebase Onboarding

This document is the primary orientation guide for anyone working in this
repository.

It explains what the codebase does, how the pieces fit together, how to work
safely, and which rules matter most when you are modifying code, docs, tests,
or experiment artifacts.

## What This Repository Is For

`llm-conceptual-modeling` is the reproducible software companion to the
manuscript on variability in generative AI for conceptual modeling.

The repository contains:

- the offline evaluation and analysis pipeline
- deterministic baselines and parity fixtures
- batch planning, resume, and worker orchestration for paper experiments
- documentation for the manuscript-facing workflow
- verification checks that keep the repo reproducible and maintainable

The large experiment payload lives in the separate Hugging Face bucket:

- [NoeFlandre/llm-variability-conceptual-modeling](https://huggingface.co/NoeFlandre/llm-variability-conceptual-modeling)

## Core Mental Model

Think of the project in three layers:

1. `src/llm_conceptual_modeling/`
   - all executable code
   - organized by concern, not by historical accident
2. `data/`
   - tracked local mirror of the canonical inputs, results, and derived artifacts
3. `docs/`
   - long-form guides for contributors and operators

The source tree is split by responsibility:

- `algo1/`, `algo2/`, `algo3/`
  - algorithm-specific workflows
- `common/`
  - shared parsing, graph, schema, and utility code
- `commands/`
  - CLI entrypoints and argument wiring
- `hf_batch/`, `hf_state/`, `hf_resume/`, `hf_drain/`, `hf_execution/`, `hf_worker/`, `hf_tail/`
  - batch planning, persistence, resume, execution, worker, and tail-run helpers
- `hf_pipeline/`
  - algorithm execution and shared result metrics
- `analysis/`
  - deterministic offline analysis
- `verification/`
  - repository health checks

Each major package directory has its own `README.md`. Read the local package
README before editing that package.

## What The Codebase Reproduces

Without issuing new LLM calls, the repository can:

- evaluate raw outputs for Algorithms 1 and 2
- recompute evaluated outputs for Algorithm 3
- run factorial analysis for all three algorithms
- generate deterministic structural baselines
- export descriptive summaries, hypothesis tests, stability reports, figures,
  failure classifications, and variability summaries
- verify deterministic parity against committed fixtures and the maintained
  results tree

The codebase does **not** promise historical provider replay equivalence.
It promises deterministic, externally checkable software behavior.

## Working Rules

Use these rules whenever you touch the repository:

- Run the baseline tests first.
- Prefer red/green TDD.
- Keep changes minimal and reviewable.
- Do not guess about semantics that can be verified from the code or fixtures.
- Delete dead code and stale docs when a cleaner source of truth exists.
- Prefer package-local `README.md` files over a single giant architecture note.
- Keep every file as small and focused as the design allows.
- Make wrongness obvious with tests, fixtures, schemas, and CLI checks.

If a task can be expressed as a narrow failing test, do that first.

## Standard Development Loop

1. Inspect the relevant code and tests.
2. Run the narrowest existing verification command.
3. Add or update a focused failing test.
4. Implement the smallest correct fix.
5. Run the focused tests again.
6. Update docs when behavior or layout changes.
7. Run the broader gate before handoff.

This project is too easy to break by broad refactors. Keep the loop short.

## Quality Requirements

The repository should remain:

- reproducible
- deterministic where possible
- documented close to the code
- split by concern into small modules
- easy to verify locally
- free of stale helper scripts and duplicated logic

Use `uv`, `ruff`, `ty`, and `pytest` where relevant. Keep dependencies minimal.

## Current Repository Structure

### Source

- `src/llm_conceptual_modeling/algo1`, `algo2`, `algo3`
  - algorithm-specific workflows and evaluation helpers
- `src/llm_conceptual_modeling/common`
  - shared utilities and shared model/data helpers
- `src/llm_conceptual_modeling/commands`
  - CLI surface
- `src/llm_conceptual_modeling/hf_batch`
  - batch planning, monitoring, and artifact helpers
- `src/llm_conceptual_modeling/hf_state`
  - canonical batch state, resume state, ledger, and shard-manifest helpers
- `src/llm_conceptual_modeling/hf_resume`
  - user-facing resume wrappers
- `src/llm_conceptual_modeling/hf_drain`
  - remaining-run drain planning and supervisor logic
- `src/llm_conceptual_modeling/hf_execution`
  - execution mechanics and subprocess monitoring
- `src/llm_conceptual_modeling/hf_pipeline`
  - algorithm execution and shared metrics
- `src/llm_conceptual_modeling/hf_worker`
  - worker process implementation and policy helpers
- `src/llm_conceptual_modeling/hf_tail`
  - dedicated final-tail prep and preflight helpers
- `src/llm_conceptual_modeling/analysis`
  - deterministic offline analysis
- `src/llm_conceptual_modeling/verification`
  - health checks and parity logic

### Data

- `data/inputs/`
  - canonical input payloads
- `data/results/frontier/`
  - archived frontier-model experiment outputs
- `data/results/open_weights/`
  - canonical Qwen/Mistral paper-facing outputs
- `data/results/archives/`
  - preserved OLMO, operational, and stale-shard archives
- `data/analysis_artifacts/`
  - revision and analysis outputs
- `data/baselines/`
  - deterministic baseline artifacts

## Results Handling

The maintained paper-facing results are under:

- `data/results/open_weights/hf-paper-batch-canonical/`

That canonical tree contains:

- `ledger.json`
- `batch_status.json`
- `runtime_config.yaml`
- run directories under `runs/`
- variance-decomposition outputs under `variance_decomposition/`

Do not trust the physical run tree alone when reasoning about completion state.
The ledger is the source of truth for finished work.

## Remote Execution Lessons

The remote batch and tail workflows taught a few stable rules:

- `current_run` is a claim signal, not proof of useful progress.
- `batch_status.json`, watcher status, and `ledger.json` are different views.
- A worker can look idle while it is still loading a model.
- If a Qwen download stalls, check Xet and blob growth before rewriting parser code.
- A failure family should become a test before it becomes a patch.
- Remote deployment should be hash-verified before trusting a host.

For remote GPU work, the high-level sequence is:

1. sync the repository
2. verify the runtime
3. validate the results root
4. launch the batch
5. watch the status and the ledger
6. finalize and sync locally

## Documentation Conventions

Use this order when looking for information:

1. `README.md`
2. `docs/onboarding.md`
3. the package-local `README.md`
4. the narrow implementation file
5. the matching test

Long, one-off revision notes should not become permanent architecture docs.
Delete or archive them once the stable package README or onboarding guide covers
the same ground.

## Future Experimental Work

The next likely expansion of the codebase is HTML causal-map ingestion and more
experiments over Qwen and Mistral with the existing decoding strategies.

Before adding that work:

- define the input contract in `data/inputs/`
- add a package-local README for the new module boundary
- add a narrow failing test before implementing the feature
- keep the new code path aligned with the existing deterministic analysis and
  verification conventions

The goal is to make future experiments easy to add without turning the repo
back into a flat pile of scripts.
