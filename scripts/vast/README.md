# `scripts/vast/`

This folder contains the operational shell entrypoints for rented Vast.ai GPU hosts.

## Role

- `bootstrap_gpu_host.sh`: prepare a fresh remote GPU host with the validated Python/CUDA stack
- `prepare_and_resume_hf_batch.sh`: one-command local wrapper to sync, bootstrap, validate, smoke, and resume a remote batch
- `quick_resume_from_ssh.sh`: convenience wrapper that accepts a raw pasted SSH command and forwards it to `prepare_and_resume_hf_batch.sh`

Fresh-host flow now starts with a local preflight:

- `uv run lcm run resume-preflight --config ... --repo-root ... --results-root ... --json`

The wrapper script runs this automatically before any SSH sync or bootstrap so obvious seed problems fail locally instead of wasting rented GPU time.
- `fetch_results_from_vast.sh`: one-shot pull of a remote results root back to the local machine
- `watch_results_from_vast.sh`: repeated local pull loop for periodic result syncing
- `sync_repo_to_vast.sh`: repository-only sync helper
- `common.sh`: shared shell helpers for SSH, rsync, and small argument-validation utilities

## Design notes

- These scripts are intentionally thin orchestration glue around the Python CLI.
- Shared shell behavior lives in `common.sh` to avoid diverging SSH/rsync conventions.
- Result transfer is intended to be embedded in `prepare_and_resume_hf_batch.sh` whenever a local results root is provided.
- The sync watcher is no longer meant to be launched manually as the primary path.
- Sync health is written into the local results tree:
  - `results-sync-status.json`
  - `results-sync-last-success.txt`
  - `results-sync.log`
  - `results-sync.pid`
- The watcher retries after SSH or rsync failures instead of exiting after the first failed pull.
- Any behavioral change here should be locked by the script tests in `tests/core/`.
