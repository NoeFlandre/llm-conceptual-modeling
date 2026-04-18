# Open-Weight Map-Extension Batch Implementation Plan

> **For noeflandre:** REQUIRED SUB-SKILL: Use test-driven-development to implement each story below.

**Goal:** make the scoped open-weight map-extension experiment batch runnable end-to-end from an SSH GPU instance, using the existing HF batch infrastructure, with resumable execution, deterministic planning, local/remote monitoring, graph-source-aware outputs, and post hoc CI replication sufficiency analysis.

**Status:** Plan only. No implementation code should be written before the first red test in each story.

## Decision Summary

The new batch extends the prior open-weight experiments from one causal map to three selected maps while holding the rest of the design tightly scoped:

- Causal maps: `babs_johnson`, `clarice_starling`, `philip_marlowe`.
- Models: `Qwen/Qwen3.5-9B`, `mistralai/Ministral-3-8B-Instruct-2512`.
- Algorithm: `algo3` only.
- Decoding: fixed to `beam_num_beams_6`.
- Varied prompt factors: example, number of words, depth.
- Fixed prompt factor: counterexample off.
- Subgraph pairs: the three existing Algo 3 directed pairs.
- Replications: 5 upfront, with CI sufficiency assessed post hoc.

Expected run count:

```text
3 maps * 2 models * 1 algorithm * 1 decoding * 3 pairs * 8 prompt conditions * 5 replications = 720 runs
```

## Non-Goals

- Do not change frontier-model or OpenRouter/Mistral SDK workflows.
- Do not rewrite the HF runtime, worker, or resume engine.
- Do not add a new scheduler or experiment runner.
- Do not change default graph behavior or existing default run-directory layout.
- Do not generate additional experiments beyond the 720-run scoped batch.
- Do not hide GPU/runtime failures with fallback behavior.

## Existing Code To Reuse

- Config loading and preview: `src/llm_conceptual_modeling/hf_config/run_config.py`.
- Batch planning: `src/llm_conceptual_modeling/hf_batch/planning.py`.
- Run identity and output paths: `src/llm_conceptual_modeling/hf_batch/spec_path.py`.
- Run manifests and seeds: `src/llm_conceptual_modeling/hf_batch/utils.py`.
- Execution and resume: `src/llm_conceptual_modeling/hf_experiments.py`.
- CLI run commands: `src/llm_conceptual_modeling/commands/run.py`.
- Vast/SSH wrapper: `scripts/vast/prepare_and_resume_hf_batch.sh`.
- Replication sufficiency analysis: `src/llm_conceptual_modeling/analysis/replication_budget_summary.py`.
- Decision context: `docs/open-weight-map-extension-decision.md`.

## Important Current Constraints

- `common.graph_data.load_default_graph()` hardcodes the original Giabbanelli and Macewan graph and category groupings.
- `HFRunConfig` currently accepts only `inputs.graph_source: default`.
- Config-based planning currently calls `load_default_graph()` inside each algorithm planner.
- `HFRunSpec` does not carry graph identity.
- `run_dir_for_spec()` does not include graph identity, so multiple maps would collide unless run identity is extended.
- `algo3_prompt_config()` currently expects `include_counterexample` in `prompt_factors`; removing the counterexample factor requires a fixed-runtime-field mechanism.
- Existing resume, ledger, and CI summaries identify runs using algorithm/model/decoding/pair/condition/replication; graph identity must be included for the new batch.

## Target Artifacts

- `data/inputs/open_weight_map_extension/babs_johnson_categories.csv`
- `data/inputs/open_weight_map_extension/babs_johnson_edges.csv`
- `data/inputs/open_weight_map_extension/clarice_starling_categories.csv`
- `data/inputs/open_weight_map_extension/clarice_starling_edges.csv`
- `data/inputs/open_weight_map_extension/philip_marlowe_categories.csv`
- `data/inputs/open_weight_map_extension/philip_marlowe_edges.csv`
- `data/inputs/open_weight_map_extension/manifest.yaml`
- `configs/hf_transformers_open_weight_map_extension.yaml`
- Updated run preview artifacts that show 720 planned runs.
- Updated docs for local validation and SSH/Vast launch.
- Post-run CI sufficiency command that can group by `graph_source`.

## Epic 1: Make Selected Causal Maps First-Class Inputs

### Story 1.1: Add selected map files under versioned inputs

**Status:** Complete in micro-step 1. Added tracked selected-map CSVs, a manifest, and a focused graph-source discovery regression test. The implementation also introduced the minimal `load_graph_source()` API needed by this test; Story 1.2 still needs its dedicated default/unknown-source regression tests.

**Requirement:** move the selected no-header CSVs out of `temporary/` and into stable input paths.

**Red test:**

- Add a failing test in `tests/common/test_graph_data.py` that asserts all three selected graph sources are discoverable by source id.
- Assert each source has non-empty `subgraph_1`, `subgraph_2`, `subgraph_3`, and `mother_graph`.
- Assert the known deterministic counts:
  - `babs_johnson`: 90 nodes, 113 edges, cluster sizes 33/32/25.
  - `clarice_starling`: 53 nodes, 64 edges, cluster sizes 11/22/20.
  - `philip_marlowe`: 29 nodes, 38 edges, cluster sizes 8/7/14.

**Implementation:**

- Create `data/inputs/open_weight_map_extension/`.
- Copy and normalize the three selected category and edge CSV pairs from `temporary/`.
- Add `manifest.yaml` with stable source ids, display names, paths, and cluster labels `1`, `2`, `3`.
- Do not alter the temporary files.

**Green test:**

```bash
uv run pytest tests/common/test_graph_data.py -q
uv run ruff check data/inputs/open_weight_map_extension src/llm_conceptual_modeling/common/graph_data.py tests/common/test_graph_data.py
```

### Story 1.2: Generalize graph loading without breaking the default graph

**Status:** Complete in micro-step 2. Added default-loader compatibility, unknown-source rejection, and root-sensitive graph-source discovery tests. The manifest cache now keys off `LCM_INPUTS_ROOT` so local overrides cannot reuse stale graph-source metadata.

**Requirement:** support both the legacy default graph and named map-extension graph sources through a single graph-loading API.

**Red test:**

- In `tests/common/test_graph_data.py`, add tests for:
  - `load_graph_source("default") == load_default_graph()`.
  - `load_graph_source("babs_johnson")` returns the expected three subgraphs.
  - Unknown graph source raises `ValueError` with the source id.
  - `LCM_INPUTS_ROOT` still overrides input files for the default graph.

**Implementation:**

- Add a small immutable type in `common.graph_data`, for example `GraphSourceSpec`.
- Add `available_graph_sources()` and `load_graph_source(source_id: str)`.
- Keep `load_default_graph()` as the compatibility wrapper for `load_graph_source("default")`.
- For the default graph, preserve the existing category groupings.
- For map-extension graphs, partition edges by numeric cluster membership.
- Fail loudly if a graph source has missing nodes, missing files, fewer than three clusters, or no intra-cluster edges.

**Green test:**

```bash
uv run pytest tests/common/test_graph_data.py -q
uv run ruff check src/llm_conceptual_modeling/common/graph_data.py tests/common/test_graph_data.py
```

## Epic 2: Make Graph Source Part Of Run Configuration And Identity

### Story 2.1: Add multi-graph-source config support

**Status:** Complete in micro-step 3 for parsing and validation. `HFRunConfig` now accepts legacy `graph_source` and new `graph_sources`, rejects conflicting/unknown sources, preserves legacy `to_dict()` output for `default`, and includes `graph_sources` in preview metadata. Planned run totals by graph source are deferred to the planning integration stories where specs become graph-aware.

**Requirement:** allow configs to specify either the legacy single `graph_source` or the new `graph_sources` list.

**Red test:**

- In `tests/core/test_hf_run_config.py`, add tests that:
  - Load `inputs.graph_sources: [babs_johnson, clarice_starling, philip_marlowe]`.
  - Preserve backward compatibility for `inputs.graph_source: default`.
  - Reject configs that set both `graph_source` and `graph_sources`.
  - Reject unknown graph sources.
  - Write `resolved_run_plan.json` with `graph_sources` and the expected total run count.

**Implementation:**

- Extend `HFRunConfig` with `graph_sources: list[str]`.
- Preserve a `graph_source` property or compatibility field if existing callers need it.
- Update `to_dict()` to write `inputs.graph_sources` for multi-source configs and `inputs.graph_source` for legacy single-source configs.
- Update `_validate_top_level_config()` to validate each source through `available_graph_sources()`.
- Update `write_resolved_run_preview()` to include:
  - `graph_sources`
  - `planned_total_runs`
  - `planned_runs_by_algorithm`
  - `planned_runs_by_graph_source`

**Green test:**

```bash
uv run pytest tests/core/test_hf_run_config.py -q
uv run ruff check src/llm_conceptual_modeling/hf_config/run_config.py tests/core/test_hf_run_config.py
```

### Story 2.2: Add fixed runtime fields and fixed design columns

**Status:** Complete in micro-step 4 for config parsing and prompt-preview runtime resolution. `AlgorithmPromptConfig` now supports `fixed_runtime_fields` and `fixed_columns`; runtime resolution starts from fixed fields before applying varied factors. Raw-context emission is deferred to the planning story because that is where run contexts are assembled.

**Requirement:** allow a config to vary only `example`, `number_of_words`, and `depth` while fixing `include_counterexample=false` and `Counter-Example=-1`.

**Red test:**

- In `tests/core/test_hf_run_config.py`, add a config fixture with:
  - `fixed_runtime_fields: {include_counterexample: false}`
  - `fixed_columns: {"Counter-Example": -1}`
- Assert condition count is 8, not 16.
- Assert prompt previews still contain a valid Algo 3 runtime field bundle.

**Implementation:**

- Add `fixed_runtime_fields: dict[str, bool | int]` and `fixed_columns: dict[str, bool | int | str]` to `AlgorithmPromptConfig`.
- Update `resolve_runtime_fields()` to start from fixed fields and overlay varied factor fields.
- Update preview condition generation to include fixed runtime fields.
- Update `_build_raw_context()` to write fixed columns before varied factor columns.
- Do not special-case Algo 3 outside this mechanism.

**Green test:**

```bash
uv run pytest tests/core/test_hf_run_config.py -q
uv run ruff check src/llm_conceptual_modeling/hf_config/run_config.py
```

### Story 2.3: Add graph source to `HFRunSpec`

**Status:** Complete in micro-step 5. `HFRunSpec` now carries a defaulted `graph_source`, manifests and smoke identities include it, and worker spec serialization/deserialization round-trips it while preserving legacy payloads through a `default` fallback.

**Requirement:** each planned run must carry its causal map identity.

**Red test:**

- In `tests/core/test_hf_batch_planning.py`, assert planned specs from a multi-graph config contain all three graph sources.
- Assert every `raw_context` includes `graph_source`.
- Assert every manifest includes `graph_source`.

**Implementation:**

- Add `graph_source: str = "default"` to `HFRunSpec`.
- Update `manifest_for_spec()` and `smoke_spec_identity()` to include `graph_source`.
- Update worker serialization/deserialization tests if spec JSON validation is strict.
- Preserve default behavior for all tests that construct `HFRunSpec` without graph source.

**Green test:**

```bash
uv run pytest tests/core/test_hf_batch_planning.py tests/core/test_hf_batch_spec_path.py tests/core/test_hf_worker.py -q
uv run ruff check src/llm_conceptual_modeling/hf_batch tests/core/test_hf_batch_planning.py tests/core/test_hf_batch_spec_path.py tests/core/test_hf_worker.py
```

### Story 2.4: Make graph source part of run directory identity

**Status:** Complete in micro-step 6. Non-default graph sources now appear in spec identity, run-directory paths, run-directory parsing, and shard-manifest filtering. The legacy `default` graph path and six-field identity remain unchanged.

**Requirement:** non-default graph sources must never collide in run directories, resume manifests, or shard manifests.

**Red test:**

- In `tests/core/test_hf_batch_spec_path.py`, add tests that:
  - Default graph specs keep the existing path layout.
  - Non-default graph specs include `graph_source` in the path.
  - `run_dir_identity()` parses both legacy and graph-aware paths.
  - `filter_planned_specs_for_output_root()` filters graph-aware shard manifests correctly.

**Implementation:**

- Extend `spec_identity()` to include graph source only where needed without breaking legacy default identities.
- Prefer a graph-aware path layout for non-default specs:

```text
runs/{algorithm}/{model_slug}/{condition_label}/{graph_source}/{pair_name}/{condition_bits}/rep_XX
```

- Keep the current six-part legacy layout for `graph_source == "default"`.
- Update shard manifest identity writing/reading to include `graph_source` for new specs.

**Green test:**

```bash
uv run pytest tests/core/test_hf_batch_spec_path.py tests/core/test_hf_shard_manifest.py tests/core/test_hf_state_shard_manifest.py -q
uv run ruff check src/llm_conceptual_modeling/hf_batch tests/core/test_hf_batch_spec_path.py
```

### Story 2.5: Include graph source in deterministic seeds for non-default graphs

**Status:** Complete in micro-step 7 for seed derivation. `derive_run_seed()` accepts an optional graph source, preserves the exact legacy default seed, and differentiates non-default map sources. Planner wiring is deferred to Epic 3, where graph-source iteration is introduced.

**Requirement:** the same model/condition/replication on different maps should not reuse the same seed.

**Red test:**

- In a focused test for `derive_run_seed()`, assert:
  - Existing default seed output is unchanged.
  - `babs_johnson` and `clarice_starling` yield different seeds for the same run identity.

**Implementation:**

- Add optional `graph_source: str = "default"` to `derive_run_seed()`.
- Preserve legacy seed material when `graph_source == "default"`.
- Include graph source in seed material only for non-default sources.
- Pass `graph_source` from config planning.

**Green test:**

```bash
uv run pytest tests/core/test_hf_batch_planning.py tests/core/test_hf_experiments.py -q
uv run ruff check src/llm_conceptual_modeling/hf_batch
```

## Epic 3: Plan The 720-Run Batch Deterministically

### Story 3.1: Load graph-specific subgraphs in config-based planning

**Status:** Complete in micro-step 8. Config-based planning now iterates `config.graph_sources`, loads each graph source through `load_graph_source()`, writes `graph_source` and fixed design columns into `raw_context`, sets `HFRunSpec.graph_source`, and passes graph source into deterministic seed derivation.

**Requirement:** planning should iterate graph sources before pairs and use the correct graph data for each source.

**Red test:**

- In `tests/core/test_hf_batch_planning.py`, add a minimal config with three graph sources, one model, one decoding, one replication, and Algo 3 only.
- Assert it produces `3 maps * 1 model * 1 decoding * 3 pairs * 8 conditions * 1 replication = 72` specs.
- Assert input payload edge counts differ by map.
- Assert condition bits are length 3 because only three factors vary.
- Assert `Counter-Example` is always `-1`.

**Implementation:**

- Replace config-planning calls to `load_default_graph()` with `load_graph_source(graph_source)`.
- Add a graph-source loop in `_plan_paper_batch_from_config()`.
- Pass `graph_source` through algorithm planners and `_build_configured_specs_for_pairs()`.
- Keep non-config planning on the legacy default graph.
- Keep pair names unchanged:
  - `subgraph_1_to_subgraph_3`
  - `subgraph_2_to_subgraph_1`
  - `subgraph_2_to_subgraph_3`

**Green test:**

```bash
uv run pytest tests/core/test_hf_batch_planning.py -q
uv run ruff check src/llm_conceptual_modeling/hf_batch/planning.py tests/core/test_hf_batch_planning.py
```

### Story 3.2: Add the production config

**Status:** Complete in micro-step 9. Added `configs/hf_transformers_open_weight_map_extension.yaml` as the canonical 720-run map-extension config, verified it loads and plans exactly 720 Algo 3 beam runs across the three selected graph sources, and upgraded `write_resolved_run_preview()` so `resolved_run_plan.json` now exposes `planned_total_runs` for local preflight validation.

**Requirement:** add a single canonical config for the map-extension batch.

**Red test:**

- Add a test that loads `configs/hf_transformers_open_weight_map_extension.yaml`.
- Assert:
  - `replications == 5`
  - `graph_sources == ["babs_johnson", "clarice_starling", "philip_marlowe"]`
  - only `algo3` is configured
  - only one decoding condition exists and it is `DecodingConfig(algorithm="beam", num_beams=6)`
  - planned spec count is 720
  - all specs have `include_counterexample is False`

**Implementation:**

- Create `configs/hf_transformers_open_weight_map_extension.yaml`.
- Reuse model ids and runtime policy from the existing open-weight HF config.
- Set output root to a clear remote-safe path, for example `/workspace/results/hf-open-weight-map-extension`.
- Use `inputs.graph_sources`, not `graph_source`.
- Configure Algo 3 only.
- Keep `max_new_tokens_by_schema` values consistent with the prior open-weight config unless a current test proves they are insufficient.

**Green test:**

```bash
uv run pytest tests/core/test_hf_run_config.py tests/core/test_hf_batch_planning.py -q
uv run lcm run validate-config --config configs/hf_transformers_open_weight_map_extension.yaml --output-dir /tmp/lcm-map-extension-preview
python - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path("/tmp/lcm-map-extension-preview/resolved_run_plan.json").read_text())
assert plan["planned_total_runs"] == 720
PY
uv run ruff check configs/hf_transformers_open_weight_map_extension.yaml src/llm_conceptual_modeling/hf_config src/llm_conceptual_modeling/hf_batch
```

## Epic 4: Keep Resume, Smoke, Ledger, And Status Graph-Aware

### Story 4.1: Update resume and ledger code paths for graph identity

**Status:** Complete in micro-step 10. Ledger identity normalization, candidate run-directory reconstruction, shard-manifest identity emission, and manifest identity key matching now preserve non-default `graph_source` values. Regression tests prove that two runs differing only by map stay distinct, and finishing one map no longer marks the sibling map run as finished.

**Requirement:** completed/pending/failed accounting must distinguish maps.

**Red test:**

- In existing resume/ledger tests, add two specs that differ only by `graph_source`.
- Assert both are tracked as separate planned runs.
- Assert finishing one graph source does not mark the other as finished.
- Assert `ledger.json` records graph source inside identity.

**Implementation:**

- Update identity construction wherever `spec_identity()`, run manifests, or ledger records are consumed.
- Prefer centralizing logic in `hf_batch.spec_path` and `hf_batch.utils` rather than duplicating tuple shapes.
- Do not special-case the new batch in resume code.

**Green test:**

```bash
uv run pytest tests/core/test_hf_ledger.py tests/core/test_hf_resume_state.py tests/core/test_hf_resume_preflight.py tests/core/test_hf_experiments.py -q
uv run ruff check src/llm_conceptual_modeling/hf_experiments.py src/llm_conceptual_modeling/hf_resume src/llm_conceptual_modeling/hf_batch
```

### Story 4.2: Make smoke selection graph-aware

**Status:** Complete in micro-step 11. `select_run_spec()` now accepts an explicit `graph_source`, rejects ambiguous omission for multi-graph configs, the CLI exposes `--graph-source` for `lcm run smoke`, and the Vast wrapper forwards an optional `SMOKE_GRAPH_SOURCE` flag when configured.

**Requirement:** `lcm run smoke` must be able to target one selected map explicitly.

**Red test:**

- Add CLI smoke-selection tests that use `--graph-source babs_johnson`.
- Assert no configured run is found when the wrong graph source is requested.
- Assert smoke verdict identity includes graph source.

**Implementation:**

- Add optional `--graph-source`, defaulting to `default`, to the smoke CLI.
- Add a `graph_source` parameter to `select_run_spec()`.
- For single-graph legacy configs, allow omitting `--graph-source`.
- Update `scripts/vast/prepare_and_resume_hf_batch.sh` to accept optional `SMOKE_GRAPH_SOURCE` and pass it to the smoke command only when set.

**Green test:**

```bash
uv run pytest tests/cli/test_cli.py tests/core/test_hf_batch_planning.py -k "smoke or select_run_spec" -q
uv run ruff check src/llm_conceptual_modeling/commands src/llm_conceptual_modeling/hf_batch scripts/vast/prepare_and_resume_hf_batch.sh
```

### Story 4.3: Verify dry-run and resume-preflight surfaces

**Status:** Complete in micro-step 12. Resume-preflight now reports both `total_runs` and `total_planned_runs`, the checked-in map-extension config preflights to exactly 720 pending runs on a fresh root, and status payload helpers now carry `graph_source` so active/last-completed run metadata can distinguish maps.

**Requirement:** a GPU rental should start from a deterministic preview, a preflight report, and a dry-run or smoke check.

**Red test:**

- Add tests for `resume-preflight` on the new config with an empty output root.
- Assert the report has `total_planned_runs == 720` and `can_resume == true`.
- Assert status/ledger summaries include graph source where row-level identity is shown.

**Implementation:**

- Update preflight report fields only if current output omits graph identity or total count.
- Keep JSON output stable for existing configs.

**Green test:**

```bash
uv run pytest tests/core/test_hf_resume_preflight.py tests/cli/test_cli.py -k "resume_preflight or status" -q
uv run ruff check src/llm_conceptual_modeling/hf_resume src/llm_conceptual_modeling/commands
```

## Epic 5: Make Post Hoc CI Sufficiency Analysis Map-Aware

### Story 5.1: Carry graph source into replication-budget observations

**Requirement:** post-run CI sufficiency summaries should identify whether underpowered conditions concentrate by map.

**Red test:**

- In `tests/analysis/test_replication_budget_summary.py`, add ledger records for two graph sources with different variability.
- Assert detailed output includes a `graph_source` column.
- Assert groupings include:
  - `graph_source`
  - `algorithm_model_graph_source`
  - `algorithm_model_graph_source_decoding`
  - `algorithm_model_graph_source_decoding_metric`

**Implementation:**

- Update `_ledger_metric_observations()` to read `identity.get("graph_source", "default")`.
- Add graph-source-aware groupings in `_GROUPINGS`.
- Include graph source in `_run_key()`.
- Preserve existing CSV columns and rows for legacy ledgers by using `default`.

**Green test:**

```bash
uv run pytest tests/analysis/test_replication_budget_summary.py tests/cli/test_cli.py -k "replication_budget_sufficiency" -q
uv run ruff check src/llm_conceptual_modeling/analysis/replication_budget_summary.py tests/analysis/test_replication_budget_summary.py tests/cli/test_cli.py
```

### Story 5.2: Add compact map-extension sufficiency output

**Requirement:** produce a concise CSV after the batch that is easy to inspect at a glance.

**Red test:**

- Add tests for a compact output that groups by algorithm, graph source, decoding, and model.
- Assert expected columns include:
  - `algorithm`
  - `graph_source`
  - `decoding`
  - `qwen_runs`
  - `qwen_ci90_needing_more`
  - `qwen_ci95_needing_more`
  - `mistral_runs`
  - `mistral_ci90_needing_more`
  - `mistral_ci95_needing_more`
- Assert legacy compact output still works for the previous one-map table.

**Implementation:**

- Prefer adding an optional argument such as `--compact-group-by graph_source` or `--include-graph-source` to the existing command rather than a new command.
- Keep the default compact table unchanged for existing users.

**Green test:**

```bash
uv run pytest tests/analysis/test_replication_budget_summary.py tests/cli/test_cli.py -k "replication_budget_sufficiency" -q
uv run ruff check src/llm_conceptual_modeling/analysis/replication_budget_summary.py src/llm_conceptual_modeling/commands
```

## Epic 6: SSH GPU Readiness And Documentation

### Story 6.1: Document exact local validation commands

**Requirement:** before renting a GPU, a user should be able to verify the batch plan locally without loading models.

**Red test:**

- Add or update a docs verification test if the repository has one for command snippets. If not, manually verify all commands in this story and record outputs in the final report.

**Implementation:**

- Update `docs/open-weight-map-extension-decision.md` or add `docs/open-weight-map-extension-runbook.md`.
- Include exact commands:

```bash
uv run pytest tests/common/test_graph_data.py tests/core/test_hf_run_config.py tests/core/test_hf_batch_planning.py tests/core/test_hf_batch_spec_path.py -q
uv run lcm run validate-config --config configs/hf_transformers_open_weight_map_extension.yaml --output-dir /tmp/lcm-map-extension-preview
uv run lcm run resume-preflight --config configs/hf_transformers_open_weight_map_extension.yaml --repo-root . --allow-empty --json
```

- Include how to inspect `/tmp/lcm-map-extension-preview/resolved_run_plan.json`.

**Green test:**

```bash
git diff --check -- docs/open-weight-map-extension-decision.md docs/open-weight-map-extension-runbook.md
```

### Story 6.2: Document exact SSH/Vast launch commands

**Requirement:** launching the batch should be copy-pasteable once an SSH target and port exist.

**Implementation:**

- Update `docs/vast-ai-transformers.md` or the new runbook with:

```bash
export SMOKE_ALGORITHM=algo3
export SMOKE_MODEL='Qwen/Qwen3.5-9B'
export SMOKE_GRAPH_SOURCE=babs_johnson
export SMOKE_PAIR_NAME=subgraph_1_to_subgraph_3
export SMOKE_CONDITION_BITS=000
export SMOKE_DECODING=beam
export SMOKE_REPLICATION=0

scripts/vast/prepare_and_resume_hf_batch.sh \
  root@HOST \
  SSH_PORT \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /workspace/llm-conceptual-modeling \
  configs/hf_transformers_open_weight_map_extension.yaml \
  /workspace/results/hf-open-weight-map-extension \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/hf-open-weight-map-extension
```

- Document resume behavior:
  - Re-run the same wrapper command after interruption.
  - Local result sync can seed remote state.
  - `shard_manifest.json` is rebuilt from unfinished local results when local results are provided.

**Verification:**

```bash
shellcheck scripts/vast/prepare_and_resume_hf_batch.sh
git diff --check -- docs/vast-ai-transformers.md docs/open-weight-map-extension-runbook.md
```

### Story 6.3: Document post-run CI analysis commands

**Requirement:** after runs finish, the CI sufficiency analysis should be one command.

**Implementation:**

- Add commands:

```bash
uv run lcm analyze replication-budget-sufficiency \
  --results-root results/hf-open-weight-map-extension \
  --output results/hf-open-weight-map-extension/replication_sufficiency_detailed.csv \
  --compact-output results/hf-open-weight-map-extension/replication_sufficiency_compact.csv \
  --expected-replications 5 \
  --include-graph-source
```

- If the final CLI flag name differs, update this plan and docs in the same commit that introduces it.

**Verification:**

```bash
uv run lcm analyze replication-budget-sufficiency --help
git diff --check -- docs/open-weight-map-extension-runbook.md
```

## Epic 7: Final Integration Gate

Run these after all stories are green:

```bash
uv run pytest \
  tests/common/test_graph_data.py \
  tests/core/test_hf_run_config.py \
  tests/core/test_hf_batch_planning.py \
  tests/core/test_hf_batch_spec_path.py \
  tests/core/test_hf_resume_preflight.py \
  tests/core/test_hf_ledger.py \
  tests/core/test_hf_resume_state.py \
  tests/core/test_hf_experiments.py \
  tests/analysis/test_replication_budget_summary.py \
  tests/cli/test_cli.py \
  -q

uv run ruff check \
  src/llm_conceptual_modeling/common/graph_data.py \
  src/llm_conceptual_modeling/hf_config \
  src/llm_conceptual_modeling/hf_batch \
  src/llm_conceptual_modeling/hf_resume \
  src/llm_conceptual_modeling/hf_experiments.py \
  src/llm_conceptual_modeling/analysis/replication_budget_summary.py \
  src/llm_conceptual_modeling/commands \
  tests/common/test_graph_data.py \
  tests/core \
  tests/analysis/test_replication_budget_summary.py \
  tests/cli/test_cli.py

uv run lcm run validate-config \
  --config configs/hf_transformers_open_weight_map_extension.yaml \
  --output-dir /tmp/lcm-map-extension-preview

uv run lcm run resume-preflight \
  --config configs/hf_transformers_open_weight_map_extension.yaml \
  --repo-root . \
  --allow-empty \
  --json
```

Inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path("/tmp/lcm-map-extension-preview/resolved_run_plan.json").read_text())
assert plan["planned_total_runs"] == 720
assert plan["planned_runs_by_graph_source"] == {
    "babs_johnson": 240,
    "clarice_starling": 240,
    "philip_marlowe": 240,
}
print(json.dumps(plan, indent=2, sort_keys=True))
PY
```

## Implementation Order

1. Epic 1: graph files and graph loader.
2. Epic 2: config, fixed fields, spec identity, seeds.
3. Epic 3: deterministic 720-run planning and production config.
4. Epic 4: resume, smoke, ledger, and status graph awareness.
5. Epic 5: CI sufficiency analysis grouped by map.
6. Epic 6: runbook and SSH launch documentation.
7. Epic 7: full integration gate.

Do not advance to the next epic with failing tests from the current epic unless the failure is a pre-existing unrelated failure and is recorded with evidence.

## Open Questions Before Implementation

1. Confirm the remote output root should be `/workspace/results/hf-open-weight-map-extension`.
2. Confirm whether smoke should default to Qwen on `babs_johnson`, or whether a Mistral smoke gate is also required before launch.
3. Confirm whether the new map CSVs should keep anonymized display names only, or whether source metadata should include the original topic labels in `manifest.yaml`.

If no answer is available, use the defaults in this plan: `/workspace/results/hf-open-weight-map-extension`, one Qwen smoke gate, and anonymized display names with topic labels omitted from run identity.
