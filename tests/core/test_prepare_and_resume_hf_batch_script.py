import os
import subprocess
from pathlib import Path


def test_prepare_and_resume_script_bootstraps_and_launches_resumable_batch() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'uv --directory "$LOCAL_REPO_DIR" run lcm run resume-preflight' in script_text
    assert 'source "$SCRIPT_DIR/common.sh"' in script_text
    assert "bootstrap_gpu_host.sh" in script_text
    assert "REMOTE_EFFECTIVE_CONFIG_PATH" in script_text
    assert "REMOTE_PREVIEW_SCRIPT" in script_text
    assert "REMOTE_LAUNCH_SCRIPT" in script_text
    assert "REMOTE_RUNTIME_DOCTOR_SCRIPT" in script_text
    assert "remote_runtime_doctor.sh" in script_text
    assert "remote_resume_preview.sh" in script_text
    assert "remote_resume_launch.sh" in script_text
    assert "docker run -d" in script_text
    assert "docker exec" in script_text
    assert "REMOTE_RUNTIME_MODE" in script_text


def test_prepare_and_resume_script_can_seed_remote_results_and_run_optional_smoke() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")
    assert 'SSH_CMD=($(vast_ssh_command "$SSH_PORT" "$SSH_KEY_PATH"))' in script_text
    assert 'RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"' in script_text
    assert 'if vast_has_value "$LOCAL_RESULTS_DIR"; then' in script_text
    assert "--exclude '.work-venv'" in script_text
    assert "--exclude '.ruff_cache'" in script_text
    assert "--exclude 'results'" in script_text
    assert "--exclude 'runs'" in script_text
    assert "--exclude 'worker-queues'" in script_text
    assert '"${SSH_CMD[@]}" "$SSH_TARGET" "mkdir -p \'$REMOTE_RESULTS_DIR\'"' in script_text
    assert 'vast_retry_rsync 3 rsync -avz \\' in script_text
    assert "  --delete \\" in script_text
    seed_resume_flags = (
        '$(vast_rsync_resume_flags "$LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS") \\'
    )
    assert seed_resume_flags in script_text
    assert '-e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/ "$SSH_TARGET:$REMOTE_RESULTS_DIR"/' in script_text
    assert 'if vast_has_value "${SMOKE_ALGORITHM:-}"' in script_text
    assert ".venv/bin/lcm run smoke" in script_text
    assert 'if vast_has_value "${SMOKE_GRAPH_SOURCE:-}"; then' in script_text
    assert 'SMOKE_GRAPH_SOURCE_FLAG="--graph-source ${SMOKE_GRAPH_SOURCE}"' in script_text
    assert '$SMOKE_GRAPH_SOURCE_FLAG \\' in script_text
    assert "BATCH_GENERATION_TIMEOUT_SECONDS" in script_text
    assert 'BATCH_STARTUP_TIMEOUT_SECONDS="${BATCH_STARTUP_TIMEOUT_SECONDS:-}"' in script_text
    assert "BATCH_RESUME_PASS_MODE" in script_text
    assert "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME" in script_text
    retry_oom_declaration = (
        'BATCH_RETRY_OOM_FAILURES_ON_RESUME="${BATCH_RETRY_OOM_FAILURES_ON_RESUME:-}"'
    )
    assert retry_oom_declaration in script_text
    assert "BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_OOM_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME" in script_text
    assert "BATCH_WORKER_PROCESS_MODE" in script_text
    assert "BATCH_MAX_REQUESTS_PER_WORKER_PROCESS" in script_text
    assert "BATCH_EXCLUDED_DECODING_LABELS" in script_text
    assert "runtime_config.yaml" in script_text
    assert "write-unfinished-manifest" in script_text
    assert "shard_manifest.json" in script_text
    assert "REMOTE_PREVIEW_ENV_PREFIX" in script_text
    assert "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME" in script_text
    assert "context_policy['startup_timeout_seconds'] = float(timeout_value)" in (
        Path("scripts/vast/remote_resume_preview.sh").read_text(encoding="utf-8")
    )


def test_prepare_and_resume_script_retries_rsync_transfers() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "vast_retry_rsync" in script_text
    assert 'vast_retry_rsync 3 rsync -avz \\' in script_text
    assert "  --delete \\" in script_text
    assert '-e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/ "$SSH_TARGET:$REMOTE_RESULTS_DIR"/' in script_text


def test_prepare_and_resume_script_can_launch_local_results_autosync() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert (
        'LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-60}"'
        in script_text
    )
    assert (
        'LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS="${LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-600}"'
        in script_text
    )
    assert "LOCAL_RESULTS_SYNC_LOG_PATH" in script_text
    assert "LOCAL_RESULTS_SYNC_PID_PATH" in script_text
    assert "LOCAL_RESULTS_SYNC_STATUS_PATH" in script_text
    assert "LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH" in script_text
    assert "scripts/vast/watch_results_from_vast.sh" in script_text
    assert 'SSH_PORT="$SSH_PORT" \\' in script_text
    assert '"$SSH_PORT" \\' in script_text
    assert "nohup bash" in script_text
    assert 'if vast_has_value "$LOCAL_RESULTS_DIR"; then' in script_text
    assert 'RSYNC_TIMEOUT_SECONDS="$LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS" \\' in script_text


def test_prepare_and_resume_script_starts_local_autosync_after_remote_launch() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    watcher_branch = 'nohup bash "$LOCAL_REPO_DIR/scripts/vast/watch_results_from_vast.sh"'
    launch_branch = 'Launch resumable batch'

    assert script_text.index(launch_branch) < script_text.index(watcher_branch)


def test_watch_results_script_passes_only_results_args_to_fetch_script() -> None:
    script_path = Path("scripts/vast/watch_results_from_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")

    fetch_call = 'SSH_PORT="$SSH_PORT" "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"'
    bad_fetch_call = (
        'SSH_PORT="$SSH_PORT" "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" '
        '"$LOCAL_RESULTS_DIR" "$SSH_PORT"'
    )

    assert fetch_call in script_text
    assert bad_fetch_call not in script_text


def test_watch_results_script_tracks_local_ledger_snapshot_after_successful_sync() -> None:
    script_text = Path("scripts/vast/watch_results_from_vast.sh").read_text(encoding="utf-8")

    assert 'LEDGER_PATH_VALUE="$LOCAL_RESULTS_DIR/ledger.json"' in script_text
    assert '"ledger_snapshot"' in script_text
    assert '"generated_at": ledger.get("generated_at")' in script_text
    assert 'lcm run refresh-ledger' in script_text
    assert '--results-root "$LEDGER_RESULTS_DIR"' in script_text
    assert '--ledger-root "$LEDGER_REFRESH_LEDGER_ROOT"' in script_text


def test_prepare_and_resume_script_supports_container_first_runtime_mode() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'REMOTE_RUNTIME_MODE="${REMOTE_RUNTIME_MODE:-auto}"' in script_text
    assert 'REMOTE_DOCKER_IMAGE="${REMOTE_DOCKER_IMAGE:-}"' in script_text
    assert (
        'REMOTE_DOCKER_CONTAINER_NAME="${REMOTE_DOCKER_CONTAINER_NAME:-lcm-$(basename '
        '"$REMOTE_RESULTS_DIR")}"'
        in script_text
    )
    assert 'REMOTE_DOCKER_PULL="${REMOTE_DOCKER_PULL:-1}"' in script_text
    assert (
        'REMOTE_RUNTIME_MODE="$(vast_select_remote_runtime_mode '
        '"$REMOTE_RUNTIME_MODE" "$REMOTE_DOCKER_IMAGE")"'
        in script_text
    )
    assert 'if [ "$REMOTE_RUNTIME_MODE" = "docker" ]; then' in script_text
    assert 'elif [ "$REMOTE_RUNTIME_MODE" = "bootstrap" ]; then' in script_text
    assert "docker image inspect" in script_text
    assert "docker run -d" in script_text
    assert "docker exec" in script_text
    assert "docker rm -f" in script_text
    assert "REMOTE_RUNTIME_DOCTOR_SCRIPT" in script_text
    assert "REMOTE_PREVIEW_SCRIPT" in script_text
    assert "REMOTE_LAUNCH_SCRIPT" in script_text
    assert "REMOTE_PREVIEW_ENV_PREFIX" in script_text
    assert "BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME" in script_text


def test_remote_resume_preview_script_writes_and_validates_effective_config() -> None:
    script_path = Path("scripts/vast/remote_resume_preview.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "REMOTE_CONFIG_PATH" in script_text
    assert "REMOTE_EFFECTIVE_CONFIG_PATH" in script_text
    assert "REMOTE_PREVIEW_DIR" in script_text
    assert 'export PYTHONPATH="$REMOTE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"' in script_text
    assert '.venv/bin/python - "$REMOTE_CONFIG_PATH" "$REMOTE_EFFECTIVE_CONFIG_PATH"' in script_text
    assert "context_policy['startup_timeout_seconds'] = float(timeout_value)" in script_text
    assert "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME" in script_text
    assert "BATCH_EXCLUDED_DECODING_LABELS" in script_text
    assert "exclude_decoding_conditions_from_payload" in script_text
    assert "context_policy['retry_structural_failures_on_resume']" in script_text
    assert ".venv/bin/lcm doctor --json --results-root" in script_text
    assert ".venv/bin/lcm run validate-config --config" in script_text
    assert ".venv/bin/lcm run prefetch-runtime --config" in script_text


def test_remote_runtime_doctor_script_validates_selected_runtime_mode() -> None:
    script_path = Path("scripts/vast/remote_runtime_doctor.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'REMOTE_RUNTIME_MODE="${REMOTE_RUNTIME_MODE:-bootstrap}"' in script_text
    assert 'REMOTE_DOCKER_IMAGE="${REMOTE_DOCKER_IMAGE:-}"' in script_text
    assert 'export PYTHONPATH="$REMOTE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"' in script_text
    assert "command -v docker >/dev/null 2>&1" in script_text
    assert "docker image inspect" in script_text
    assert "docker run --rm" in script_text
    assert "command -v curl >/dev/null 2>&1" in script_text
    assert "command -v bash >/dev/null 2>&1" in script_text
    assert "command -v nvidia-smi >/dev/null 2>&1" in script_text


def test_remote_resume_launch_script_restarts_the_batch_process() -> None:
    script_path = Path("scripts/vast/remote_resume_launch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "REMOTE_RUN_LOG" in script_text
    assert "REMOTE_PID_PATH" in script_text
    assert "REMOTE_GPU_LIVENESS_TIMEOUT_SECONDS" in script_text
    gpu_timeout = (
        'REMOTE_GPU_LIVENESS_TIMEOUT_SECONDS="${REMOTE_GPU_LIVENESS_TIMEOUT_SECONDS:-600}"'
    )
    assert gpu_timeout in script_text
    assert 'export PYTHONPATH="$REMOTE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"' in script_text
    assert "REMOTE_GPU_LIVENESS_POLL_INTERVAL_SECONDS" in script_text
    assert "REMOTE_PRODUCTIVE_LIVENESS_TIMEOUT_SECONDS" in script_text
    assert "REMOTE_PRODUCTIVE_LIVENESS_POLL_INTERVAL_SECONDS" in script_text
    assert 'REMOTE_RELAUNCH_IF_PENDING="${REMOTE_RELAUNCH_IF_PENDING:-1}"' in script_text
    assert 'REMOTE_RELAUNCH_SLEEP_SECONDS="${REMOTE_RELAUNCH_SLEEP_SECONDS:-5}"' in script_text
    assert "export REMOTE_REPO_DIR" in script_text
    assert "vast_pending_count()" in script_text
    assert 'echo "[remote_resume_launch] starting supervised resume loop"' in script_text
    assert "pkill -f " in script_text
    assert "llm_conceptual_modeling.hf_worker --queue-dir" in script_text
    assert 'echo "[remote_resume_launch] cleaned stale workers before resume"' in script_text
    assert 'echo "[remote_resume_launch] pending_count=$pending_count"' in script_text
    assert 'python3 - <<'"'"'PY'"'"'' in script_text
    assert 'while true; do' in script_text
    assert 'pending_count="$(python3 - <<' in script_text
    assert 'if [ "$pending_count" -le 0 ]; then' in script_text
    assert 'sleep "$REMOTE_RELAUNCH_SLEEP_SECONDS"' in script_text
    assert "nohup bash <<'EOF'" in script_text
    assert 'REMOTE_REPO_DIR="/workspace/llm-conceptual-modeling"' not in script_text
    assert 'REMOTE_RESULTS_DIR="/workspace/results/hf-paper-batch-canonical"' not in script_text
    assert "/workspace/results/hf-paper-batch-canonical/worker-queues/" not in script_text
    assert "env -u NVIDIA_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES" in script_text
    assert 'PYTHONPATH="$REMOTE_REPO_DIR/src"' in script_text
    assert '"$REMOTE_REPO_DIR/.venv/bin/lcm" run paper-batch' in script_text
    assert "vast_wait_for_gpu_liveness" in script_text
    assert "worker_state.json" in script_text
    assert "batch_status.json" in script_text
    assert "executing_algorithm" in script_text
    assert "vast_wait_for_productive_liveness" in script_text
    assert "--resume" in script_text
    assert "supervisor_pid=$!" in script_text
    assert "REMOTE_PROCESS_EXIT_TIMEOUT_SECONDS" in script_text
    assert "vast_wait_for_process_exit" in script_text
    assert 'Path(os.environ["REMOTE_RESULTS_DIR"]) / "batch_status.json"' in script_text
    batch_pattern = (
        'pkill -f -- ".venv/bin/lcm run paper-batch --config '
        '$REMOTE_EFFECTIVE_CONFIG_PATH --resume"'
    )
    assert batch_pattern in script_text
    worker_pattern = (
        'pkill -f -- "llm_conceptual_modeling.hf_worker --queue-dir '
        '$REMOTE_RESULTS_DIR/worker-queues/"'
    )
    assert worker_pattern in script_text
    batch_wait = (
        'vast_wait_for_process_exit ".venv/bin/lcm run paper-batch --config '
        '$REMOTE_EFFECTIVE_CONFIG_PATH --resume"'
    )
    assert batch_wait in script_text
    worker_wait = (
        'vast_wait_for_process_exit "llm_conceptual_modeling.hf_worker --queue-dir '
        '$REMOTE_RESULTS_DIR/worker-queues/"'
    )
    assert worker_wait in script_text
    assert "find \"$REMOTE_RESULTS_DIR/worker-queues\" -type f" in script_text
    assert (
        "-name '*.request.json' -o -name '*.claimed.json' -o -name '*.result.json'"
        in script_text
    )


def test_vast_common_script_centralizes_shared_shell_helpers() -> None:
    script_path = Path("scripts/vast/common.sh")
    script_text = script_path.read_text(encoding="utf-8")

    connect_timeout = (
        'VAST_SSH_CONNECT_TIMEOUT_SECONDS="${VAST_SSH_CONNECT_TIMEOUT_SECONDS:-60}"'
    )
    assert connect_timeout in script_text
    assert 'VAST_SSH_CONTROL_PATH="${VAST_SSH_CONTROL_PATH:-$HOME/.ssh/lcm-vast-%C}"' in script_text
    assert (
        'VAST_SSH_CONTROL_PERSIST_SECONDS="${VAST_SSH_CONTROL_PERSIST_SECONDS:-600}"'
        in script_text
    )
    assert "vast_ssh_transport_flags()" in script_text
    assert "StrictHostKeyChecking=accept-new" in script_text
    assert "ControlMaster=auto" in script_text
    assert "ControlPersist=${VAST_SSH_CONTROL_PERSIST_SECONDS}" in script_text
    assert "ControlPath=${VAST_SSH_CONTROL_PATH}" in script_text
    assert "vast_ssh_command()" in script_text
    assert "vast_remote_script_path()" in script_text


def test_vast_remote_script_path_translates_local_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    common_script = repo_root / "scripts/vast/common.sh"
    local_repo_dir = str(repo_root)
    remote_repo_dir = "/workspace/llm-conceptual-modeling"
    local_candidate = f"{local_repo_dir}/scripts/vast/remote_resume_preview.sh"

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f"source {common_script!s} && "
                "vast_remote_script_path "
                f"{local_candidate!s} {local_repo_dir!s} {remote_repo_dir!s}"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == (
        f"{remote_repo_dir}/scripts/vast/remote_resume_preview.sh"
    )


def test_remote_resume_preview_noop_script_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "scripts/vast/remote_resume_preview_noop.sh").exists()


def test_quick_resume_script_can_parse_raw_ssh_command_and_delegate() -> None:
    script_path = Path("scripts/vast/quick_resume_from_ssh.sh")
    script_text = script_path.read_text(encoding="utf-8")
    usage = (
        "usage: quick_resume_from_ssh.sh SSH_COMMAND CONFIG_PATH "
        "REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR"
    )

    assert usage in script_text
    assert 'source "$SCRIPT_DIR/common.sh"' in script_text
    assert 'PARSED_SSH="$(vast_parse_ssh_command "$SSH_COMMAND")"' in script_text
    assert 'exec "$SCRIPT_DIR/prepare_and_resume_hf_batch.sh"' in script_text
    assert 'REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/workspace/llm-conceptual-modeling}"' in script_text


def test_prepare_and_resume_script_can_use_result_root_configs_when_present() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "CONFIG_PATH" in script_text
    assert "LOCAL_CONFIG_SOURCE_PATH" in script_text
    assert 'if [ -f "$LOCAL_REPO_DIR/$config_path" ]; then' in script_text
    assert (
        'if vast_has_value "$LOCAL_RESULTS_DIR" && [ -f "$LOCAL_RESULTS_DIR/$config_path" ]'
        in script_text
    )
    assert "REMOTE_CONFIG_SUFFIX" in script_text
    assert 'REMOTE_CONFIG_PATH="$REMOTE_REPO_DIR/$REMOTE_CONFIG_SUFFIX"' in script_text
    assert 'REMOTE_CONFIG_PATH="$REMOTE_RESULTS_DIR/$REMOTE_CONFIG_SUFFIX"' in script_text


def test_prepare_and_resume_script_prefers_local_results_configs_over_repo_paths() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    results_branch = (
        'if vast_has_value "$LOCAL_RESULTS_DIR" && [ -f "$LOCAL_RESULTS_DIR/$config_path" ]'
    )
    repo_branch = 'if [ -f "$LOCAL_REPO_DIR/$config_path" ]; then'

    assert script_text.index(results_branch) < script_text.index(repo_branch)


def test_drain_remaining_script_delegates_to_unified_cli() -> None:
    script_text = Path("scripts/vast/drain_remaining_from_ssh.sh").read_text(encoding="utf-8")

    assert "drain-remaining" in script_text
    assert "--repo-root" in script_text
    assert "--results-root" in script_text
    assert "--ssh-command" in script_text
    assert "--state-file" in script_text
    assert "--phase" in script_text
    assert "--poll-seconds" in script_text
    assert "--stale-after-seconds" in script_text
    assert "--quick-resume-script" in script_text
    assert "DRAIN_PLAN_ONLY" in script_text
    assert "DRAIN_JSON" in script_text


def test_drain_olmo_wrapper_delegates_to_generic_supervisor() -> None:
    script_text = Path("scripts/vast/drain_olmo_batches_from_ssh.sh").read_text(encoding="utf-8")
    drain_poll = 'DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-${OLMO_DRAIN_POLL_SECONDS:-30}}"'
    quick_resume = (
        'DRAIN_QUICK_RESUME_SCRIPT="${DRAIN_QUICK_RESUME_SCRIPT:-'
        '${OLMO_QUICK_RESUME_SCRIPT:-$SCRIPT_DIR/quick_resume_from_ssh.sh}}"'
    )
    sync_interval = (
        'LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-'
        '${OLMO_LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-30}}"'
    )
    sync_timeout = (
        'LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS="${LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-'
        '${OLMO_LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-300}}"'
    )

    assert 'DRAIN_ROOT_NAME_CONTAINS="${DRAIN_ROOT_NAME_CONTAINS:-olmo}"' in script_text
    assert 'DRAIN_PHASE="${DRAIN_PHASE:-all}"' in script_text
    assert drain_poll in script_text
    assert (
        'DRAIN_STALE_AFTER_SECONDS="${DRAIN_STALE_AFTER_SECONDS:-${OLMO_RUNNING_ROOT_STALE_SECONDS:-3600}}"'
        in script_text
    )
    assert quick_resume in script_text
    assert sync_interval in script_text
    assert sync_timeout in script_text
    assert 'exec bash "$SCRIPT_DIR/drain_remaining_from_ssh.sh"' in script_text


def test_drain_qwen_wrapper_delegates_to_generic_supervisor() -> None:
    script_text = Path("scripts/vast/drain_qwen_batches_from_ssh.sh").read_text(encoding="utf-8")
    drain_poll = 'DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-${QWEN_DRAIN_POLL_SECONDS:-30}}"'
    quick_resume = (
        'DRAIN_QUICK_RESUME_SCRIPT="${DRAIN_QUICK_RESUME_SCRIPT:-'
        '${QWEN_QUICK_RESUME_SCRIPT:-$SCRIPT_DIR/quick_resume_from_ssh.sh}}"'
    )
    sync_interval = (
        'LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-'
        '${QWEN_LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-30}}"'
    )
    sync_timeout = (
        'LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS="${LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-'
        '${QWEN_LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-300}}"'
    )

    assert 'DRAIN_ROOT_NAME_CONTAINS="${DRAIN_ROOT_NAME_CONTAINS:-qwen}"' in script_text
    assert 'DRAIN_PHASE="${DRAIN_PHASE:-all}"' in script_text
    assert drain_poll in script_text
    assert (
        'DRAIN_STALE_AFTER_SECONDS="${DRAIN_STALE_AFTER_SECONDS:-${QWEN_RUNNING_ROOT_STALE_SECONDS:-3600}}"'
        in script_text
    )
    assert quick_resume in script_text
    assert sync_interval in script_text
    assert sync_timeout in script_text
    assert 'exec bash "$SCRIPT_DIR/drain_remaining_from_ssh.sh"' in script_text


def test_drain_remaining_script_passes_expected_cli_arguments(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()
    uv_stub = tmp_path / "uv"
    argv_dump = tmp_path / "argv.txt"
    env_dump = tmp_path / "env.json"
    uv_stub.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "{argv_dump}"
python3 - <<'PY'
from pathlib import Path
import json
import os

Path(r"{env_dump}").write_text(
    json.dumps(
        {{
            "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS": os.environ.get(
                "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS"
            ),
            "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS": os.environ.get(
                "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS"
            ),
        }},
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
""",
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)

    env = os.environ.copy()
    env["UV_BIN"] = str(uv_stub)
    env["DRAIN_PLAN_ONLY"] = "1"
    env["DRAIN_JSON"] = "1"
    env["DRAIN_PHASE"] = "safe"
    env["DRAIN_POLL_SECONDS"] = "5"
    env["DRAIN_STALE_AFTER_SECONDS"] = "120"
    env["DRAIN_STATE_FILE"] = str(tmp_path / "state.json")

    result = subprocess.run(
        [
            "bash",
            "scripts/vast/drain_remaining_from_ssh.sh",
            "ssh -p 12345 root@127.0.0.1",
            str(repo_root),
            str(results_root),
            "olmo",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    argv = argv_dump.read_text(encoding="utf-8")
    assert "run" in argv
    assert "drain-remaining" in argv
    assert str(repo_root) in argv
    assert str(results_root) in argv
    assert "ssh -p 12345 root@127.0.0.1" in argv
    assert "--root-name-contains" in argv
    assert "olmo" in argv
    assert "--plan-only" in argv
    assert "--json" in argv


def test_drain_olmo_wrapper_sets_filter_and_sync_defaults(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()
    uv_stub = tmp_path / "uv"
    argv_dump = tmp_path / "argv.txt"
    env_dump = tmp_path / "env.json"
    uv_stub.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "{argv_dump}"
python3 - <<'PY'
from pathlib import Path
import json
import os

Path(r"{env_dump}").write_text(
    json.dumps(
        {{
            "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS": os.environ.get(
                "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS"
            ),
            "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS": os.environ.get(
                "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS"
            ),
            "DRAIN_QUICK_RESUME_SCRIPT": os.environ.get("DRAIN_QUICK_RESUME_SCRIPT"),
        }},
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
""",
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)

    env = os.environ.copy()
    env["UV_BIN"] = str(uv_stub)
    env["DRAIN_PLAN_ONLY"] = "1"
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(tmp_path / "quick_resume_stub.sh")

    result = subprocess.run(
        [
            "bash",
            "scripts/vast/drain_olmo_batches_from_ssh.sh",
            "ssh -p 12345 root@127.0.0.1",
            str(repo_root),
            str(results_root),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    argv = argv_dump.read_text(encoding="utf-8")
    payload = env_dump.read_text(encoding="utf-8")
    assert "--root-name-contains" in argv
    assert "olmo" in argv
    assert '"LOCAL_RESULTS_SYNC_INTERVAL_SECONDS": "30"' in payload
    assert '"LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS": "300"' in payload


def test_drain_qwen_wrapper_sets_filter_and_sync_defaults(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()
    uv_stub = tmp_path / "uv"
    argv_dump = tmp_path / "argv.txt"
    env_dump = tmp_path / "env.json"
    uv_stub.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "{argv_dump}"
python3 - <<'PY'
from pathlib import Path
import json
import os

Path(r"{env_dump}").write_text(
    json.dumps(
        {{
            "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS": os.environ.get(
                "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS"
            ),
            "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS": os.environ.get(
                "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS"
            ),
            "DRAIN_QUICK_RESUME_SCRIPT": os.environ.get("DRAIN_QUICK_RESUME_SCRIPT"),
        }},
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
""",
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)

    env = os.environ.copy()
    env["UV_BIN"] = str(uv_stub)
    env["DRAIN_PLAN_ONLY"] = "1"
    env["QWEN_QUICK_RESUME_SCRIPT"] = str(tmp_path / "quick_resume_stub.sh")

    result = subprocess.run(
        [
            "bash",
            "scripts/vast/drain_qwen_batches_from_ssh.sh",
            "ssh -p 12345 root@127.0.0.1",
            str(repo_root),
            str(results_root),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    argv = argv_dump.read_text(encoding="utf-8")
    payload = env_dump.read_text(encoding="utf-8")
    assert "--root-name-contains" in argv
    assert "qwen" in argv
    assert '"LOCAL_RESULTS_SYNC_INTERVAL_SECONDS": "30"' in payload
    assert '"LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS": "300"' in payload
