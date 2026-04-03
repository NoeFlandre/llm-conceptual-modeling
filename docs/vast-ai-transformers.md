# Vast.ai Transformers Batch

This workflow is for the new local-`transformers` experiment family that runs on a remote NVIDIA GPU over SSH.

## Scope

- provider: `hf-transformers`
- chat models:
  - `mistralai/Ministral-3-8B-Instruct-2512`
  - `Qwen/Qwen3.5-9B`
  - `allenai/Olmo-3-7B-Instruct`
- embedding model:
  - `Qwen/Qwen3-Embedding-0.6B`
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
- short-circuits entirely if the existing `.venv` already matches the validated runtime
- runs `uv sync`
- installs `wheel`
- pins `torch` to the CUDA 12.1 wheel line used by the validated remote snapshot
- pins `triton` without re-resolving `torch`
- disables the Xet/HF transfer path that caused remote download stalls
- fails immediately if CUDA is unavailable
- writes and prints the effective `torch` / CUDA / `transformers` / `triton` snapshot to
  `.bootstrap-runtime.json`

For repeated rentals, the best practice is to freeze this exact validated runtime into a GPU-ready
container image or Vast template. The bootstrap script remains the source of truth for the pinned
stack, but a prebuilt image avoids spending paid time re-downloading and re-resolving dependencies.
The checked-in starting point is:

- `docker/vast-gpu.Dockerfile`
- `scripts/vast/prepare_and_resume_hf_batch.sh`

If you already have a prebuilt image, the fresh-host launcher can skip the host-side `uv sync`
bootstrap entirely by setting:

```bash
export REMOTE_RUNTIME_MODE=docker
export REMOTE_DOCKER_IMAGE=<your-prebuilt-image-tag>
```

In container mode the launcher mounts the synced repo and seeded results into the container, then
reuses the same remote preview and launch helpers. This is the lowest-friction path once the image
exists, because the paid host only needs Docker and the image pull.

## Fresh-Instance One-Command Flow

For a brand-new rented GPU, or when moving the resumable batch onto a larger GPU, prefer the
single local wrapper instead of retyping the bootstrap / validate / launch sequence by hand:

```bash
scripts/vast/prepare_and_resume_hf_batch.sh \
  root@61.228.57.170 \
  31291 \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
  /workspace/llm-conceptual-modeling \
  configs/hf_transformers_paper_batch.yaml \
  /workspace/results/hf-paper-batch-algo1-olmo-current \
  /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/hf-paper-batch-algo1-olmo-current
```

This wrapper:

- syncs the local repository to the remote workspace
- optionally seeds the remote results root from a local copy so `--resume` can continue on a fresh
  machine
- either runs the pinned bootstrap script or starts the prebuilt Docker image, depending on
  `REMOTE_RUNTIME_MODE`
- runs the shared remote preview helper, which rewrites the effective config, then runs
  `lcm doctor` and `lcm run validate-config`
- optionally runs the smoke gate when smoke environment variables are set
- launches `paper-batch --resume` under `nohup` either on the host or inside the running container
- can also start the local periodic result pull loop when `LOCAL_RESULTS_DIR` and
  `LOCAL_RESULTS_SYNC_INTERVAL_SECONDS` are both set

Useful runtime overrides for the wrapper:

```bash
export BATCH_GENERATION_TIMEOUT_SECONDS=40
export BATCH_RESUME_PASS_MODE=throughput
export BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME=false
export BATCH_WORKER_PROCESS_MODE=ephemeral
export BATCH_MAX_REQUESTS_PER_WORKER_PROCESS=32
```

Shell glue layout:

- `scripts/vast/common.sh`: shared SSH/rsync/value-validation helpers
- `scripts/vast/fetch_results_from_vast.sh`: one-shot result pull
- `scripts/vast/watch_results_from_vast.sh`: repeated local sync loop
- `scripts/vast/prepare_and_resume_hf_batch.sh`: top-level orchestration wrapper

Persistent-worker guidance:

- `BATCH_WORKER_PROCESS_MODE=ephemeral`: current default, safest behavior, one worker process per run
- `BATCH_WORKER_PROCESS_MODE=persistent`: experimental throughput mode, one loaded worker reused per
  host/model
- `BATCH_MAX_REQUESTS_PER_WORKER_PROCESS`: optional recycling budget for persistent mode so one
  process does not live forever
- `BATCH_RETRY_OOM_FAILURES_ON_RESUME=false`: optional targeted pass setting when a host has already
  shown deterministic CUDA OOM failures and you want to defer them instead of replaying them on the
  same card

Pass guidance:

- `throughput`: default. Defer prior timeout failures, retry structurally recoverable failures, and
  prioritize lower-risk pending cells first.
- `retry-timeouts`: explicit second pass. Retry prior timeout failures first, then continue with the
  remaining queue.

Optional smoke gate variables for the wrapper:

```bash
export SMOKE_ALGORITHM=algo1
export SMOKE_MODEL=allenai/Olmo-3-7B-Instruct
export SMOKE_PAIR_NAME=sg1_sg2
export SMOKE_CONDITION_BITS=00000
export SMOKE_DECODING=greedy
```

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

Then run the exact-condition smoke gate before the full DOE. This is the command that decides
whether the rented machine is actually ready:

```bash
uv run lcm run smoke \
  --config configs/hf_transformers_paper_batch.yaml \
  --algorithm algo1 \
  --model Qwen/Qwen3.5-9B \
  --pair-name sg2_sg3 \
  --condition-bits 00000 \
  --decoding greedy \
  --replication 0 \
  --output-root /workspace/results/hf-smoke-qwen-sg2-sg3
```

If that smoke run does not complete cleanly, stop and fix the environment before launching the full
batch. Do not spend more GPU time debugging inside `paper-batch`.

Each smoke run now writes one definitive verdict file:

- `smoke_verdict.json`

This file records:

- `status`
- `failure_type`
- `failure_message`
- `worker_loaded_model`
- `runtime_snapshot_path`
- the exact spec identity for the smoke condition

If `smoke_verdict.json` is not `success`, stop immediately and return to local fixes.

The YAML controls the actual run settings, including models, decoding parameters, temperature,
seed, per-model thinking-mode declarations, prompt fragments, DOE factor fragments, and the output
root. The
`max_new_tokens_by_schema` values are starting budgets, not hard caps: the runtime can grow them
when needed, but it will fail loudly instead of returning a truncated output.

Supported thinking control is explicit in the config:

- `Qwen/Qwen3.5-9B` must be declared as `disabled`
- models without a documented public toggle must be declared as `acknowledged-unsupported`

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

Exact single-condition HF smoke run:

```bash
uv run lcm run smoke \
  --config configs/hf_transformers_paper_batch.yaml \
  --algorithm algo1 \
  --model Qwen/Qwen3.5-9B \
  --pair-name sg2_sg3 \
  --condition-bits 00000 \
  --decoding greedy \
  --replication 0 \
  --output-root /workspace/results/hf-smoke-qwen-sg2-sg3
```

This is the recommended paid-run gate for Qwen because `algo1 / sg2_sg3 / 00000 / greedy`
was the first condition that exposed the runtime bottleneck during remote execution.

Do not launch the full DOE until this exact smoke run:

- loads the model successfully
- writes `manifest.json`, `runtime.json`, `raw_response.json`, `raw_row.json`, and
  `summary.json`
- completes without hanging on the initial `edge_list` generation stage
- produces a populated `summary.json`

Config-driven planning preview through the legacy `generate` surface:

```bash
uv run lcm generate algo1 \
  --provider hf-transformers \
  --config configs/hf_transformers_paper_batch.yaml \
  --json
```

The run layout is resumable. Each condition writes:

- `manifest.json`
- `state.json`
- `worker_state.json`
- `runtime.json`
- `raw_response.json`
- `raw_row.json`
- `summary.json`
- `error.json` on failure

For ALGO1 and ALGO2, reusable stage caches are written under `stages/` so retries can resume from
completed earlier stages instead of regenerating the full call chain.

Completed runs are not recomputed when `--resume` is used.

For paid runs, the recommended order is:

1. `bootstrap_gpu_host.sh`
2. `lcm doctor --json --results-root ... --smoke-root ...`
2. `lcm run validate-config`
3. `lcm run smoke` for the exact target model / pair / condition
4. `lcm run paper-batch --resume` only after the smoke run succeeds

Stop conditions:

- bootstrap does not produce `.bootstrap-runtime.json`
- `doctor` shows a missing smoke verdict or a non-success smoke verdict
- smoke does not write `summary.json`
- smoke reports `worker_loaded_model=false` on failure, which means the environment is still not
  ready
- `paper-batch` shows more than one GPU-heavy process for the same run, which indicates parent and
  worker duplication

## Live Monitoring

While the batch is running, the root results directory is updated with:

- `batch_status.json`

This file records:

- `total_runs`
- `finished_count`
- `failed_count`
- `running_count`
- `pending_count`
- `percent_complete`
- `current_run`
- `last_completed_run`
- failure summaries
- active stage details
- worker lifecycle details
- visible GPU compute processes

You can inspect it directly:

```bash
cat /workspace/results/hf-paper-batch/batch_status.json
```

Or use the CLI health view, which reconstructs status from the run tree and does not rely only on
the mutable status file:

```bash
uv run lcm run status \
  --results-root /workspace/results/hf-paper-batch \
  --json
```

For a quick remote preflight before the paid run, use:

```bash
uv run lcm doctor \
  --json \
  --results-root /workspace/results/hf-paper-batch \
  --smoke-root /workspace/results/hf-smoke-qwen-sg2-sg3
```

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

Keep local results synced automatically while a batch is running:

```bash
SSH_PORT=31255 \
SYNC_INTERVAL_SECONDS=60 \
scripts/vast/watch_results_from_vast.sh \
  root@61.228.57.170:/workspace/results/hf-paper-batch \
  /local/path/hf-paper-batch
```

`prepare_and_resume_hf_batch.sh` can also start that local autosync loop automatically when both
`LOCAL_RESULTS_DIR` and `LOCAL_RESULTS_SYNC_INTERVAL_SECONDS` are set.

## Notes

- Prompt truncation is not allowed. The local runtime checks the actual chat-templated tokenized
  input against the tokenizer limit before generation and uses the YAML-configured safety margin.
- The runtime derives the smallest required context window for each prompt rather than reserving a
  larger fixed window than necessary.
- Qwen thinking is explicitly disabled through the chat template path. The other selected models
  do not advertise an equivalent public toggle in the implementation, so manifests/runtime records
  mark them as `not-supported-by-model` rather than pretending they were explicitly disabled.
