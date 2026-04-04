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
- `watch_results_from_vast.sh`: repeated local pull loop for periodic result syncing
- `sync_repo_to_vast.sh`: repository-only sync helper
- `common.sh`: shared shell helpers for SSH, rsync, and small argument-validation utilities

## Design notes

- These scripts are intentionally thin orchestration glue around the Python CLI.
- Shared shell behavior lives in `common.sh` to avoid diverging SSH/rsync conventions.
- Result transfer is intended to be embedded in `prepare_and_resume_hf_batch.sh` whenever a local results root is provided.
- The remote runtime is split into small helpers so the preview/config rewrite step and the launch step can be reused in both host-bootstrap and container modes.
- The sync watcher is no longer meant to be launched manually as the primary path.
- Malformed structured outputs that still reach the worker as retryable errors now include empty
  edge endpoints, empty structured fields, and JSON decode failures; those are retried instead of
  being left as permanent failures when the resume policy allows it.
- Persistent worker sessions use the same retry predicate as the subprocess worker path, so
  structural, infrastructure, and OOM-style worker failures recycle the worker instead of
  becoming terminal on the first attempt.
- `resume-sweep` is the quickest way to tell whether a local root is `resume-ready`, `needs-config-fix`, or already `active` before you rent another SSH instance.
- For OLMO work, `drain_olmo_batches_from_ssh.sh` reuses the seeded result-tree `runtime_config.yaml` files and advances each algorithm root pass by pass, waiting for the current root to finish before moving on.
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
- Any behavioral change here should be locked by the script tests in `tests/core/`.
