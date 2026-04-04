# `scripts/vast/`

This folder contains the operational shell entrypoints for rented Vast.ai GPU hosts.

## Role

- `bootstrap_gpu_host.sh`: prepare a fresh remote GPU host with the validated Python/CUDA stack
  - the bootstrap path now retries transient `uv sync` / `uv pip install` failures so a short-lived
    PyPI timeout does not abort an otherwise healthy resume
  - the current default wheel line is CUDA 12.8 so the same bootstrap works on Blackwell and
    non-Blackwell NVIDIA hosts
- `prepare_and_resume_hf_batch.sh`: one-command local wrapper to sync, bootstrap, validate, smoke, and resume a remote batch
- `quick_resume_from_ssh.sh`: convenience wrapper that accepts a raw pasted SSH command and forwards it to `prepare_and_resume_hf_batch.sh`
- `drain_olmo_batches_from_ssh.sh`: sequentially drains the local OLMO roots across `algo1`, `algo2`, and `algo3` using their seeded result-tree configs
- `drain_qwen_batches_from_ssh.sh`: sequentially drains the local Qwen roots across `algo1`, `algo2`, and `algo3` using their seeded result-tree configs
- `remote_runtime_doctor.sh`: remote guardrail that validates the selected runtime path before result seeding and launch
- `resume-sweep` via `uv run lcm run resume-sweep --repo-root ... --results-root ... --json`: local readiness report across all seeded result roots before renting another host
- `remote_resume_preview.sh`: remote helper that rewrites the effective runtime config and runs `lcm doctor` plus `lcm run validate-config`
- `remote_resume_launch.sh`: remote helper that kills stale workers and starts `paper-batch --resume`

Fresh-host flow now starts with a local preflight:

- `uv run lcm run resume-preflight --config ... --repo-root ... --results-root ... --json`

The wrapper script runs this automatically before any SSH sync or bootstrap so obvious seed problems fail locally instead of wasting rented GPU time.
The fresh-host wrapper accepts config paths that live either under the checked-in repo or under the seeded local results root, which keeps legacy result-root configs like `results/hf-paper-batch-algo1-qwen/runtime_config.yaml` launchable without manual copying.
The repository sync step now excludes the top-level `results/` tree plus local-only caches like `.work-venv/` and `.ruff_cache/`, in addition to `data/results/` and `data/analysis_artifacts/`, so the host only receives source and seeded results.
- The wrapper now supports a container-first runtime mode. If `REMOTE_DOCKER_IMAGE` is set, the wrapper automatically switches to Docker mode unless you explicitly force `REMOTE_RUNTIME_MODE=bootstrap`. The mode choice is centralized in `scripts/vast/common.sh` via `vast_select_remote_runtime_mode()`. In Docker mode, the synced repo and seeded results are mounted into the container and the remote preview/launch helpers run inside it.
- `fetch_results_from_vast.sh`: one-shot pull of a remote results root back to the local machine
- `watch_results_from_vast.sh`: repeated local pull loop for periodic result syncing; accepts an optional SSH port argument for nonstandard Vast instances
- `lcm run prefetch-runtime --config ... --json`: preloads the configured chat and embedding model bundles with the same runtime stack used for generation
- `sync_repo_to_vast.sh`: repository-only sync helper
- `common.sh`: shared shell helpers for SSH, rsync, and small argument-validation utilities

## Design notes

- These scripts are intentionally thin orchestration glue around the Python CLI.
- Shared shell behavior lives in `common.sh` to avoid diverging SSH/rsync conventions.
- Result transfer is intended to be embedded in `prepare_and_resume_hf_batch.sh` whenever a local results root is provided.
- Fresh-host launch now performs an explicit remote runtime doctor before result seeding so broken Docker/bootstrap environments fail fast instead of consuming paid GPU time during setup.
- Fresh-host preview now prefetches the configured models before `paper-batch --resume`, so the slow model-download boundary is moved ahead of the live batch launch.
- Worker state now distinguishes model warmup as `prefetching_model` before it transitions to
  `executing_algorithm`, which makes startup easier to reason about during resumes.
- The local watcher status file now carries a stable `watcher_identity` and explicit states (`starting`, `syncing`, `healthy`, `degraded`, `stopped`) so stale watchers from older hosts are easier to detect.
- The remote runtime is split into small helpers so the preview/config rewrite step and the launch step can be reused in both host-bootstrap and container modes.
- Remote launch no longer treats “GPU-attached worker exists” as sufficient proof of a healthy
  resume. The launcher now waits for productive liveness: either the worker reaches
  `executing_algorithm` or the root records a fresh finished run.
- The sync watcher is no longer meant to be launched manually as the primary path.
- Malformed structured outputs that still reach the worker as retryable errors now include empty
  edge endpoints, empty structured fields, and JSON decode failures; those are retried instead of
  being left as permanent failures when the resume policy allows it.
- The subprocess retry loop now also consumes retry budget on generation-stage
  `MonitoredCommandTimeout` failures and on retryable post-load structural validation failures,
  which matters for OLMO `algo1` runs that return malformed edge lists before eventually
  producing a valid finished result.
- Persistent worker sessions use the same retry predicate as the subprocess worker path, so
  structural, infrastructure, and OOM-style worker failures recycle the worker instead of
  becoming terminal on the first attempt.
- That shared retry predicate now also treats startup drift and timeout failures consistently
  across subprocess and persistent-worker execution. A worker that exits before writing its
  result artifact is classified as infrastructure failure and retried instead of being stranded
  as a terminal `other` failure on resume.
- `resume-sweep` is the quickest way to tell whether a local root is `resume-ready`, `needs-config-fix`, or already `active` before you rent another SSH instance.
- `resume-sweep` now also reports the default runtime mode, safe resume profile, excluded decoding labels, whether the root is actually rent-ready under the conservative default profile, and a single recommended results root to rent next.
- `resume-preflight` now carries forward the local `batch_status.json` running count, so roots that are already active are conservatively blocked instead of being misreported as rentable.
- For OLMO work, `drain_olmo_batches_from_ssh.sh` reuses the seeded result-tree `runtime_config.yaml` files and advances each algorithm root pass by pass, waiting for the current root to finish before moving on.
- For Qwen work, `drain_qwen_batches_from_ssh.sh` follows the same pass-by-pass flow across `algo1`, `algo2`, and `algo3`, while keeping contrastive decoding enabled and using dynamic cache for Qwen contrastive generation.
- For `algo1-olmo`, the drain script now excludes the known OOM-heavy
  `contrastive_penalty_alpha_0.8` branch during remote preview rewriting, so fresh
  resumes keep the rest of the seeded root but stop burning GPU time on that branch.
- The OLMO drain script also forces `retry_oom_failures_on_resume=true` into the remote effective
  config so stale preview artifacts cannot silently disable OOM replay on a fresh host.
- The same drain wrapper now also forces timeout and infrastructure retries on resume, so
  unattended `algo1 -> algo2 -> algo3` passes keep consuming deferred retryable work instead of
  stopping after the first timeout-heavy phase.
- For `algo2-olmo`, the safe fresh-host profile is now greedy plus `beam_num_beams=2`; the
  OOM-prone `beam_num_beams=6` and contrastive branches are intentionally excluded from the
  resume profile.
- For `algo2-mistral`, keep contrastive but only the safe branch (`penalty_alpha=0.2`) and
  beam-2. This keeps contrastive coverage while avoiding the higher-risk resume branches.
- Sync health is written into the local results tree:
  - `results-sync-status.json`
  - `results-sync-last-success.txt`
  - `results-sync.log`
  - `results-sync.pid`
- The watcher retries after SSH or rsync failures instead of exiting after the first failed pull.
- When syncing from a custom-port SSH host, pass the port explicitly as the third watcher argument so rsync does not fall back to port 22.
- Any behavioral change here should be locked by the script tests in `tests/core/`.
