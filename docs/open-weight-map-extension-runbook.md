# Open-Weight Map-Extension Runbook

This runbook covers the scoped open-weight map-extension batch defined by
[docs/open-weight-map-extension-decision.md](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/docs/open-weight-map-extension-decision.md).
The batch is fixed to Algo 3, `beam_num_beams_6`, counterexamples off, and
three graph sources:

- `babs_johnson`
- `clarice_starling`
- `philip_marlowe`

The checked-in config is:

- [configs/hf_transformers_open_weight_map_extension.yaml](/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/configs/hf_transformers_open_weight_map_extension.yaml)

The planned workload is exactly 720 runs:

```text
3 maps × 2 models × 1 decoding × 3 pairs × 8 prompt conditions × 5 replications
```

## 1. Local Validation Before Renting

Run the focused verification set from the repository root:

```bash
uv run pytest \
  tests/common/test_graph_data.py \
  tests/core/test_hf_run_config.py \
  tests/core/test_hf_batch_planning.py \
  tests/core/test_hf_batch_spec_path.py \
  tests/core/test_hf_ledger.py \
  tests/core/test_hf_state_shard_manifest.py \
  tests/core/test_hf_resume_preflight.py \
  tests/analysis/test_replication_budget_summary.py \
  tests/cli/test_cli.py \
  -q
```

Validate the source-of-truth config and write the resolved preview:

```bash
rm -rf /tmp/lcm-map-extension-preview
uv run lcm run validate-config \
  --config configs/hf_transformers_open_weight_map_extension.yaml \
  --output-dir /tmp/lcm-map-extension-preview
```

Check that the resolved preview still plans 720 runs:

```bash
python - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path("/tmp/lcm-map-extension-preview/resolved_run_plan.json").read_text())
assert plan["planned_total_runs"] == 720
assert plan["graph_sources"] == [
    "babs_johnson",
    "clarice_starling",
    "philip_marlowe",
]
print(json.dumps(plan, indent=2, sort_keys=True))
PY
```

Run resume preflight against an empty local root:

```bash
uv run lcm run resume-preflight \
  --config configs/hf_transformers_open_weight_map_extension.yaml \
  --repo-root . \
  --allow-empty \
  --json
```

The expected fresh-root signals are:

- `"total_runs": 720`
- `"total_planned_runs": 720`
- `"pending_count": 720`
- `"can_resume": true`
- `"resume_mode": "fresh-root"`

Run one local dry-run smoke selection to prove the exact graph-aware condition
can be selected before any GPU time is spent:

```bash
uv run lcm run smoke \
  --config configs/hf_transformers_open_weight_map_extension.yaml \
  --algorithm algo3 \
  --model Qwen/Qwen3.5-9B \
  --graph-source babs_johnson \
  --pair-name subgraph_1_to_subgraph_3 \
  --condition-bits 000 \
  --decoding beam \
  --num-beams 6 \
  --replication 0 \
  --dry-run \
  --output-root /tmp/lcm-map-extension-smoke
```

## 2. SSH Launch

Set the smoke gate to one explicit map-aware condition:

```bash
export SMOKE_ALGORITHM=algo3
export SMOKE_MODEL='Qwen/Qwen3.5-9B'
export SMOKE_GRAPH_SOURCE=babs_johnson
export SMOKE_PAIR_NAME=subgraph_1_to_subgraph_3
export SMOKE_CONDITION_BITS=000
export SMOKE_DECODING=beam
export SMOKE_REPLICATION=0
export SMOKE_OUTPUT_ROOT=/workspace/results/hf-open-weight-map-extension-smoke
```

If you want automatic local result syncing during the rented run, also set:

```bash
export LOCAL_RESULTS_SYNC_INTERVAL_SECONDS=60
```

Launch the batch with the standard wrapper:

```bash
scripts/vast/prepare_and_resume_hf_batch.sh \
  root@HOST \
  SSH_PORT \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /workspace/llm-conceptual-modeling \
  configs/hf_transformers_open_weight_map_extension.yaml \
  /workspace/results/hf-open-weight-map-extension \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/hf-open-weight-map-extension
```

The wrapper will:

- sync the repo
- bootstrap or enter the configured runtime
- run remote doctor and config validation
- run the explicit smoke gate above
- start `paper-batch --resume`
- optionally start local result syncing if `LOCAL_RESULTS_DIR` is provided

## 3. Resume After Interruption

Resume is intentionally the same operator command. Re-run the exact same wrapper
after an SSH disconnect or host interruption:

```bash
scripts/vast/prepare_and_resume_hf_batch.sh \
  root@HOST \
  SSH_PORT \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /workspace/llm-conceptual-modeling \
  configs/hf_transformers_open_weight_map_extension.yaml \
  /workspace/results/hf-open-weight-map-extension \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/hf-open-weight-map-extension
```

The local results root can reseed the remote root, and unfinished identities are
rebuilt from the local ledger/shard-manifest state rather than guessed from the
physical run tree alone.

## 4. Post-Run CI Sufficiency Analysis

Once the batch is finished locally, produce both the detailed and compact
graph-aware sufficiency summaries with one command:

```bash
uv run lcm analyze replication-budget-sufficiency \
  --results-root results/hf-open-weight-map-extension \
  --output results/hf-open-weight-map-extension/replication_sufficiency_detailed.csv \
  --compact-output results/hf-open-weight-map-extension/replication_sufficiency_compact.csv \
  --expected-replications 5 \
  --include-graph-source
```

This yields:

- `replication_sufficiency_detailed.csv`
  - includes `graph_source`
  - includes graph-aware groupings such as
    `algorithm_model_graph_source_decoding_metric`
- `replication_sufficiency_compact.csv`
  - one row per algorithm / graph source / decoding condition
  - separate Qwen and Mistral sufficiency columns for 90% and 95% CI targets

## 5. Post-Run Variance Decomposition

After the full batch is finished and the ledger has been refreshed, generate the
variance-decomposition audit bundle from the same completed results root:

```bash
uv run lcm analyze variance-decomposition-bundle \
  --results-root results/hf-open-weight-map-extension \
  --output-dir results/hf-open-weight-map-extension/variance_decomposition
```

This writes the standard variance-decomposition bundle artifacts:

- `variance_decomposition.csv`
- `variance_decomposition_algo1.csv`
- `variance_decomposition_algo2.csv`
- `variance_decomposition_algo3.csv`
- `variance_decomposition.tex`
- `variance_decomposition_algo1.tex`
- `variance_decomposition_algo2.tex`
- `variance_decomposition_algo3.tex`

## 6. Notes

- `shellcheck` is not required by the repository and may not be installed
  locally. The script behavior is locked by the shell-script tests under
  `tests/core/`.
- The local preview artifact at `/tmp/lcm-map-extension-preview/resolved_run_plan.json`
  is the quickest source-of-truth check before renting.
- For this batch, ambiguous smoke selection is treated as a configuration error:
  if the config contains multiple graph sources, pass `--graph-source`.
