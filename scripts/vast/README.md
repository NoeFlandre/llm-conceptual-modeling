# `scripts/vast/`

This folder contains the operational shell entrypoints for rented Vast.ai GPU hosts.

## Role

- `bootstrap_gpu_host.sh`: prepare a fresh remote GPU host with the validated Python/CUDA stack
- `prepare_and_resume_hf_batch.sh`: one-command local wrapper to sync, bootstrap, validate, smoke, and resume a remote batch
- `quick_resume_from_ssh.sh`: convenience wrapper that accepts a raw pasted SSH command and forwards it to `prepare_and_resume_hf_batch.sh`
- `resume-sweep` via `uv run lcm run resume-sweep --repo-root ... --results-root ... --json`: local readiness report across all seeded result roots before renting another host
- `remote_resume_preview.sh`: remote helper that rewrites the effective runtime config and runs `lcm doctor` plus `lcm run validate-config`
- `remote_resume_launch.sh`: remote helper that kills stale workers and starts `paper-batch --resume`

Fresh-host flow now starts with a local preflight:

- `uv run lcm run resume-preflight --config ... --repo-root ... --results-root ... --json`

The wrapper script runs this automatically before any SSH sync or bootstrap so obvious seed problems fail locally instead of wasting rented GPU time.
The fresh-host wrapper accepts config paths that live either under the checked-in repo or under the seeded local results root, which keeps legacy result-root configs like `results/hf-paper-batch-algo1-qwen/runtime_config.yaml` launchable without manual copying.
The repository sync step now excludes the top-level `results/` tree plus local-only caches like `.work-venv/` and `.ruff_cache/`, in addition to `data/results/` and `data/analysis_artifacts/`, so the host only receives source and seeded results.
- The wrapper now supports a container-first runtime mode when `REMOTE_RUNTIME_MODE=docker` and `REMOTE_DOCKER_IMAGE` points at a prebuilt Vast-compatible GPU image. In that mode, the synced repo and seeded results are mounted into the container and the remote preview/launch helpers run inside it.
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
- `resume-sweep` is the quickest way to tell whether a local root is `resume-ready`, `needs-config-fix`, or already `active` before you rent another SSH instance.
- Sync health is written into the local results tree:
  - `results-sync-status.json`
  - `results-sync-last-success.txt`
  - `results-sync.log`
  - `results-sync.pid`
- The watcher retries after SSH or rsync failures instead of exiting after the first failed pull.
- Any behavioral change here should be locked by the script tests in `tests/core/`.
