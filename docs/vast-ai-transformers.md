# Vast.ai Transformers Batch

This workflow is for the new local-`transformers` experiment family that runs on a remote NVIDIA GPU over SSH.

## Scope

- provider: `hf-transformers`
- chat models:
  - `mistralai/Ministral-3-8B-Instruct-2512`
  - `Qwen/Qwen3.5-9B`
  - `allenai/Olmo-3-7B-Instruct`
- embedding model:
  - `Qwen/Qwen3-Embedding-8B`
- decoding algorithms:
  - greedy
  - beam search with `num_beams` in `{2, 6}`
  - contrastive search with `penalty_alpha` in `{0.2, 0.8}` and `top_k = 4`
- temperature defaults to `0.0` in the checked-in YAML config
- quantization is disabled
- CPU fallback is disabled

## Remote Bootstrap

Clone the repository on the Vast.ai machine, then run:

```bash
scripts/vast/bootstrap_gpu_host.sh /path/to/llm-conceptual-modeling
```

The script:

- installs `uv` if needed
- runs `uv sync`
- reinstalls `torch` from the CUDA wheel index
- fails immediately if CUDA is unavailable

## Preflight Review

Validate the checked-in source-of-truth config and inspect the resolved preview before starting a
paid run:

```bash
uv run lcm run validate-config \
  --config configs/hf_transformers_paper_batch.yaml \
  --output-dir /workspace/results/hf-paper-batch-preview
```

This writes:

- `resolved_run_config.yaml`
- `resolved_run_plan.json`
- `prompt_preview/...`

The YAML controls the actual run settings, including models, decoding parameters, temperature,
seed, prompt fragments, DOE factor fragments, and the output root. The
`max_new_tokens_by_schema` values are starting budgets, not hard caps: the runtime can grow them
when needed, but it will fail loudly instead of returning a truncated output.

## Run Commands

Full batch:

```bash
uv run lcm run paper-batch \
  --config configs/hf_transformers_paper_batch.yaml \
  --resume
```

Single-algorithm smoke run:

```bash
uv run lcm run algo1 \
  --config configs/hf_transformers_paper_batch.yaml \
  --resume
```

The run layout is resumable. Each condition writes:

- `manifest.json`
- `state.json`
- `runtime.json`
- `raw_response.json`
- `raw_row.json`
- `summary.json`
- `error.json` on failure

Completed runs are not recomputed when `--resume` is used.

## Outputs

For each algorithm / model / decoding condition, the runner writes:

- `aggregated/.../raw.csv`
- `aggregated/.../evaluated.csv`
- `aggregated/.../factorial.csv`
- `aggregated/.../condition_stability.csv`
- `aggregated/.../replication_budget_strict.csv`
- `aggregated/.../replication_budget_relaxed.csv`
- `aggregated/.../output_variability.csv`

For each algorithm / model pair, the runner also writes a combined DOE surface under
`aggregated/<algorithm>/<model>/combined/`, including:

- `raw.csv`
- `evaluated.csv`
- `factorial.csv`
- `condition_stability.csv`
- `replication_budget_strict.csv`
- `replication_budget_relaxed.csv`
- `output_variability.csv`

The combined `factorial.csv` is the one that includes decoding as an explicit factor and an
explicit residual `Error` row.

The strict report uses the 95% CI / 5% relative half-width rule.
The relaxed report uses the 90% CI / 10% relative half-width rule.
Neither report launches additional runs automatically.

## Plot Export

Generate the three reviewer-facing plots from a completed batch:

```bash
uv run lcm analyze plots \
  --results-root /workspace/results/hf-paper-batch \
  --output-dir /workspace/results/hf-paper-batch/plots
```

This writes:

- `distribution_metrics.png`
- `factor_effect_summary.png`
- `raw_output_variability.png`

## Sync

Send the repository to the remote host:

```bash
scripts/vast/sync_repo_to_vast.sh \
  /path/to/llm-conceptual-modeling \
  user@host:/workspace/llm-conceptual-modeling
```

Fetch results back:

```bash
scripts/vast/fetch_results_from_vast.sh \
  user@host:/workspace/results/hf-paper-batch \
  /local/path/hf-paper-batch
```

## Notes

- Prompt truncation is not allowed. The local runtime checks the actual chat-templated tokenized
  input against the tokenizer limit before generation and uses the YAML-configured safety margin.
- The runtime derives the smallest required context window for each prompt rather than reserving a
  larger fixed window than necessary.
- Qwen thinking is explicitly disabled through the chat template path. The other selected models
  do not advertise an equivalent public toggle in the implementation, so manifests/runtime records
  mark them as `not-supported-by-model` rather than pretending they were explicitly disabled.
