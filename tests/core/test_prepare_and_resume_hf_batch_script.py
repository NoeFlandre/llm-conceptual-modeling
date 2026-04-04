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
    seed_resume_flags = (
        '$(vast_rsync_resume_flags "$LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS") \\'
    )
    assert seed_resume_flags in script_text
    assert '-e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/ "$SSH_TARGET:$REMOTE_RESULTS_DIR"/' in script_text
    assert 'if vast_has_value "${SMOKE_ALGORITHM:-}"' in script_text
    assert ".venv/bin/lcm run smoke" in script_text
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
    assert "runtime_config.yaml" in script_text
    assert "REMOTE_PREVIEW_ENV_PREFIX" in script_text
    assert "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME" in script_text
    assert "context_policy['startup_timeout_seconds'] = float(timeout_value)" in (
        Path("scripts/vast/remote_resume_preview.sh").read_text(encoding="utf-8")
    )


def test_prepare_and_resume_script_retries_rsync_transfers() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "vast_retry_rsync" in script_text
    assert 'vast_retry_rsync 3 rsync -avz -e "$RSYNC_SSH"' in script_text
    assert 'vast_retry_rsync 3 rsync -avz -e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/' in script_text


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
    assert "nohup bash" in script_text
    assert 'if vast_has_value "$LOCAL_RESULTS_DIR"; then' in script_text
    assert 'RSYNC_TIMEOUT_SECONDS="$LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS" \\' in script_text


def test_prepare_and_resume_script_starts_local_autosync_after_remote_launch() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    watcher_branch = 'nohup bash "$LOCAL_REPO_DIR/scripts/vast/watch_results_from_vast.sh"'
    launch_branch = 'echo "[6/6] Launch resumable batch"'

    assert script_text.index(launch_branch) < script_text.index(watcher_branch)


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
    assert '.venv/bin/python - "$REMOTE_CONFIG_PATH" "$REMOTE_EFFECTIVE_CONFIG_PATH"' in script_text
    assert "context_policy['startup_timeout_seconds'] = float(timeout_value)" in script_text
    assert "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME" in script_text
    assert "BATCH_EXCLUDED_DECODING_LABELS" in script_text
    assert "exclude_decoding_conditions_from_payload" in script_text
    assert "context_policy['retry_structural_failures_on_resume']" in script_text
    assert ".venv/bin/lcm doctor --json --results-root" in script_text
    assert ".venv/bin/lcm run validate-config --config" in script_text


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
    assert "REMOTE_GPU_LIVENESS_POLL_INTERVAL_SECONDS" in script_text
    assert "pkill -f 'lcm run paper-batch'" in script_text
    assert "pkill -f 'llm_conceptual_modeling.hf_worker'" in script_text
    assert "nohup env -u NVIDIA_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES" in script_text
    assert "vast_wait_for_gpu_liveness" in script_text
    assert "--resume" in script_text
    assert "pgrep -n -f" in script_text


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
    assert "vast_rsync_ssh_command()" in script_text
    assert "vast_rsync_resume_flags()" in script_text
    assert "vast_has_value()" in script_text
    assert "vast_require_positive_integer()" in script_text
    assert "vast_select_remote_runtime_mode()" in script_text


def test_quick_resume_script_can_parse_raw_ssh_command_and_delegate() -> None:
    script_path = Path("scripts/vast/quick_resume_from_ssh.sh")
    script_text = script_path.read_text(encoding="utf-8")
    usage = (
        "usage: quick_resume_from_ssh.sh SSH_COMMAND CONFIG_PATH "
        "REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR"
    )

    assert usage in script_text
    assert "shlex.split" in script_text
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


def test_drain_olmo_script_sequences_all_olmo_roots_and_uses_result_root_configs() -> None:
    script_path = Path("scripts/vast/drain_olmo_batches_from_ssh.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "hf-paper-batch-algo1-olmo-current" in script_text
    assert "hf-paper-batch-algo2-olmo-current" in script_text
    assert "hf-paper-batch-algo3-olmo-current" in script_text
    assert "runtime_config.yaml" in script_text
    assert 'OLMO_GENERATION_TIMEOUT_SECONDS="${OLMO_GENERATION_TIMEOUT_SECONDS:-60}"' in script_text
    assert 'BATCH_GENERATION_TIMEOUT_SECONDS="$OLMO_GENERATION_TIMEOUT_SECONDS" \\' in script_text
    assert (
        'OLMO_RETRY_STRUCTURAL_FAILURES_ON_RESUME="${OLMO_RETRY_STRUCTURAL_FAILURES_ON_RESUME:-true}"'
        in script_text
    )
    assert (
        'OLMO_RETRY_TIMEOUT_FAILURES_ON_RESUME="${OLMO_RETRY_TIMEOUT_FAILURES_ON_RESUME:-true}"'
        in script_text
    )
    assert (
        'OLMO_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME="${OLMO_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME:-true}"'
        in script_text
    )
    assert (
        'OLMO_RETRY_OOM_FAILURES_ON_RESUME="${OLMO_RETRY_OOM_FAILURES_ON_RESUME:-true}"'
        in script_text
    )
    assert "BATCH_RETRY_OOM_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME" in script_text
    assert "root_excluded_decoding_labels()" in script_text
    assert "BATCH_EXCLUDED_DECODING_LABELS" in script_text
    assert "pending_count" in script_text
    assert "running_count" in script_text
    assert "while true; do" in script_text


def test_drain_olmo_script_retries_transient_launch_failures(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "configs" / "hf_transformers_algo3_olmo.yaml").write_text(
        Path("configs/hf_transformers_algo3_olmo.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    results_root = tmp_path / "results"
    olmo_root = results_root / "hf-paper-batch-algo1-olmo-current"
    olmo_root.mkdir(parents=True)
    (olmo_root / "runtime_config.yaml").write_text(
        (Path("results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    (olmo_root / "batch_status.json").write_text(
        """
        {
          "total_runs": 1,
          "finished_count": 0,
          "failed_count": 1,
          "running_count": 0,
          "pending_count": 0,
          "updated_at": "2026-04-04T00:00:00Z"
        }
        """.strip(),
        encoding="utf-8",
    )

    helper_script = tmp_path / "quick_resume_stub.sh"
    helper_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

state_file="${STUB_STATE_FILE:?}"
invocations_file="${STUB_INVOCATIONS_FILE:?}"
local_results_dir="$4"
count=0
if [ -f "$state_file" ]; then
  count="$(cat "$state_file")"
fi
count=$((count + 1))
printf '%s\n' "$count" > "$state_file"
printf '%s\n' "$count" >> "$invocations_file"
if [ "$count" = "1" ]; then
  exit 42
fi
(
  sleep 1
  cat >"$local_results_dir/batch_status.json" <<'JSON'
{
  "total_runs": 1,
  "finished_count": 1,
  "failed_count": 0,
  "running_count": 0,
  "pending_count": 0,
  "updated_at": "2026-04-04T00:00:05Z"
}
JSON
) &
exit 0
""",
        encoding="utf-8",
    )
    helper_script.chmod(0o755)

    env = os.environ.copy()
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(helper_script)
    env["OLMO_LAUNCH_ATTEMPTS"] = "2"
    env["OLMO_LAUNCH_BACKOFF_SECONDS"] = "0"
    env["OLMO_DRAIN_POLL_SECONDS"] = "0"
    env["OLMO_DRAIN_MAX_WAIT_SECONDS"] = "30"
    env["STUB_STATE_FILE"] = str(tmp_path / "state.txt")
    env["STUB_INVOCATIONS_FILE"] = str(tmp_path / "invocations.txt")

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
    assert (tmp_path / "state.txt").read_text(encoding="utf-8").strip() == "2"
    assert (tmp_path / "invocations.txt").read_text(encoding="utf-8").splitlines() == ["1", "2"]
    status = (olmo_root / "batch_status.json").read_text(encoding="utf-8")
    assert '"finished_count": 1' in status
    assert '"pending_count": 0' in status


def test_drain_olmo_script_forces_frequent_local_sync(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    olmo_root = results_root / "hf-paper-batch-algo1-olmo-current"
    olmo_root.mkdir(parents=True)
    (olmo_root / "runtime_config.yaml").write_text(
        Path("results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (olmo_root / "batch_status.json").write_text(
        """
        {
          "total_runs": 1,
          "finished_count": 0,
          "failed_count": 0,
          "running_count": 0,
          "pending_count": 1,
          "updated_at": "2026-04-04T00:00:00Z"
        }
        """.strip(),
        encoding="utf-8",
    )

    helper_script = tmp_path / "quick_resume_stub.sh"
    env_dump = tmp_path / "env_dump.json"
    helper_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json
import os

payload = {{
    "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS": os.environ.get("LOCAL_RESULTS_SYNC_INTERVAL_SECONDS"),
    "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS": os.environ.get(
        "LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS"
    ),
}}
Path(r"{env_dump}").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
cat >"$4/batch_status.json" <<'JSON'
{{
  "total_runs": 1,
  "finished_count": 1,
  "failed_count": 0,
  "running_count": 0,
  "pending_count": 0,
  "updated_at": "2026-04-04T00:00:05Z"
}}
JSON
""",
        encoding="utf-8",
    )
    helper_script.chmod(0o755)

    env = os.environ.copy()
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(helper_script)
    env["OLMO_LAUNCH_ATTEMPTS"] = "1"
    env["OLMO_DRAIN_POLL_SECONDS"] = "0"
    env["STUB_STATE_FILE"] = str(tmp_path / "unused-state.txt")
    env["STUB_INVOCATIONS_FILE"] = str(tmp_path / "unused-invocations.txt")

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
    payload = env_dump.read_text(encoding="utf-8")
    assert '"LOCAL_RESULTS_SYNC_INTERVAL_SECONDS": "30"' in payload
    assert '"LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS": "300"' in payload


def test_drain_olmo_script_processes_all_olmo_roots_in_order(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    order = [
        "hf-paper-batch-algo1-olmo-current",
        "hf-paper-batch-algo2-olmo-current",
        "hf-paper-batch-algo3-olmo-current",
    ]
    config_text = Path("results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml").read_text(
        encoding="utf-8"
    )
    for root_name in order:
        root_dir = results_root / root_name
        root_dir.mkdir(parents=True)
        (root_dir / "runtime_config.yaml").write_text(config_text, encoding="utf-8")
        (root_dir / "batch_status.json").write_text(
            """
            {
              "total_runs": 1,
              "finished_count": 0,
              "failed_count": 0,
              "running_count": 0,
              "pending_count": 1,
              "updated_at": "2026-04-04T00:00:00Z"
            }
            """.strip(),
            encoding="utf-8",
        )

    helper_script = tmp_path / "quick_resume_stub.sh"
    seen_order = tmp_path / "seen_order.txt"
    helper_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$(basename "$4")" >> "{seen_order}"
cat >"$4/batch_status.json" <<'JSON'
{{
  "total_runs": 1,
  "finished_count": 1,
  "failed_count": 0,
  "running_count": 0,
  "pending_count": 0,
  "updated_at": "2026-04-04T00:00:05Z"
}}
JSON
""",
        encoding="utf-8",
    )
    helper_script.chmod(0o755)

    env = os.environ.copy()
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(helper_script)
    env["OLMO_LAUNCH_ATTEMPTS"] = "1"
    env["OLMO_DRAIN_POLL_SECONDS"] = "0"
    env["OLMO_DRAIN_MAX_PASSES"] = "1"

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
    assert seen_order.read_text(encoding="utf-8").splitlines() == order


def test_drain_olmo_script_uses_safe_decoding_excludes_for_algo1_olmo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    olmo_root = results_root / "hf-paper-batch-algo1-olmo-current"
    olmo_root.mkdir(parents=True)
    (olmo_root / "runtime_config.yaml").write_text(
        Path("results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (olmo_root / "batch_status.json").write_text(
        """
        {
          "total_runs": 1,
          "finished_count": 0,
          "failed_count": 0,
          "running_count": 0,
          "pending_count": 1,
          "updated_at": "2026-04-04T00:00:00Z"
        }
        """.strip(),
        encoding="utf-8",
    )

    helper_script = tmp_path / "quick_resume_stub.sh"
    env_dump = tmp_path / "env_dump.json"
    helper_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json
import os

Path(r"{env_dump}").write_text(
    json.dumps(
        {{
            "BATCH_EXCLUDED_DECODING_LABELS": os.environ.get("BATCH_EXCLUDED_DECODING_LABELS"),
        }},
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
cat >"$4/batch_status.json" <<'JSON'
{{
  "total_runs": 1,
  "finished_count": 1,
  "failed_count": 0,
  "running_count": 0,
  "pending_count": 0,
  "updated_at": "2026-04-04T00:00:05Z"
}}
JSON
""",
        encoding="utf-8",
    )
    helper_script.chmod(0o755)

    env = os.environ.copy()
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(helper_script)
    env["OLMO_LAUNCH_ATTEMPTS"] = "1"
    env["OLMO_DRAIN_POLL_SECONDS"] = "0"
    env["OLMO_DRAIN_MAX_PASSES"] = "1"

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
    payload = env_dump.read_text(encoding="utf-8")
    assert '"BATCH_EXCLUDED_DECODING_LABELS": "contrastive_penalty_alpha_0.8"' in payload


def test_drain_olmo_script_uses_preview_resume_config_when_runtime_config_missing(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    olmo_root = results_root / "hf-paper-batch-algo2-olmo-current"
    olmo_root.mkdir(parents=True)
    preview_resume = olmo_root / "preview_resume"
    preview_resume.mkdir(parents=True)
    preview_resume_config = preview_resume / "resolved_run_config.yaml"
    preview_resume_config.write_text(
        Path("results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (olmo_root / "batch_status.json").write_text(
        """
        {
          "total_runs": 1,
          "finished_count": 0,
          "failed_count": 0,
          "running_count": 0,
          "pending_count": 1,
          "updated_at": "2026-04-04T00:00:00Z"
        }
        """.strip(),
        encoding="utf-8",
    )

    helper_script = tmp_path / "quick_resume_stub.sh"
    seen_config = tmp_path / "seen_config.txt"
    helper_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$2" > "{seen_config}"
cat >"$4/batch_status.json" <<'JSON'
{{
  "total_runs": 1,
  "finished_count": 1,
  "failed_count": 0,
  "running_count": 0,
  "pending_count": 0,
  "updated_at": "2026-04-04T00:00:05Z"
}}
JSON
""",
        encoding="utf-8",
    )
    helper_script.chmod(0o755)

    env = os.environ.copy()
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(helper_script)
    env["OLMO_LAUNCH_ATTEMPTS"] = "1"
    env["OLMO_DRAIN_MAX_PASSES"] = "1"
    env["OLMO_DRAIN_POLL_SECONDS"] = "0"

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
    assert seen_config.read_text(encoding="utf-8").strip().endswith(
        "preview_resume/resolved_run_config.yaml"
    )


def test_drain_olmo_script_falls_back_to_repo_config_when_results_configs_are_missing(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "configs" / "hf_transformers_algo3_olmo.yaml").write_text(
        Path("configs/hf_transformers_algo3_olmo.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    results_root = tmp_path / "results"
    olmo_root = results_root / "hf-paper-batch-algo3-olmo-current"
    olmo_root.mkdir(parents=True)
    (olmo_root / "batch_status.json").write_text(
        """
        {
          "total_runs": 1,
          "finished_count": 0,
          "failed_count": 0,
          "running_count": 0,
          "pending_count": 1,
          "updated_at": "2026-04-04T00:00:00Z"
        }
        """.strip(),
        encoding="utf-8",
    )

    helper_script = tmp_path / "quick_resume_stub.sh"
    seen_config = tmp_path / "seen_repo_config.txt"
    helper_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$2" > "{seen_config}"
cat >"$4/batch_status.json" <<'JSON'
{{
  "total_runs": 1,
  "finished_count": 1,
  "failed_count": 0,
  "running_count": 0,
  "pending_count": 0,
  "updated_at": "2026-04-04T00:00:05Z"
}}
JSON
""",
        encoding="utf-8",
    )
    helper_script.chmod(0o755)

    env = os.environ.copy()
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(helper_script)
    env["OLMO_LAUNCH_ATTEMPTS"] = "1"
    env["OLMO_DRAIN_MAX_PASSES"] = "1"
    env["OLMO_DRAIN_POLL_SECONDS"] = "0"

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
    assert seen_config.read_text(encoding="utf-8").strip().endswith(
        "configs/hf_transformers_algo3_olmo.yaml"
    )


def test_drain_olmo_script_reclaims_stale_running_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    olmo_root = results_root / "hf-paper-batch-algo1-olmo-current"
    olmo_root.mkdir(parents=True)
    (olmo_root / "runtime_config.yaml").write_text(
        Path("results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (olmo_root / "batch_status.json").write_text(
        """
        {
          "total_runs": 1,
          "finished_count": 0,
          "failed_count": 0,
          "running_count": 1,
          "pending_count": 0,
          "updated_at": "2024-01-01T00:00:00Z"
        }
        """.strip(),
        encoding="utf-8",
    )

    helper_script = tmp_path / "quick_resume_stub.sh"
    seen_invocations = tmp_path / "seen_stale.txt"
    helper_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$2" > "{seen_invocations}"
cat >"$4/batch_status.json" <<'JSON'
{{
  "total_runs": 1,
  "finished_count": 1,
  "failed_count": 0,
  "running_count": 0,
  "pending_count": 0,
  "updated_at": "2026-04-04T00:00:05Z"
}}
JSON
""",
        encoding="utf-8",
    )
    helper_script.chmod(0o755)

    env = os.environ.copy()
    env["OLMO_QUICK_RESUME_SCRIPT"] = str(helper_script)
    env["OLMO_LAUNCH_ATTEMPTS"] = "1"
    env["OLMO_DRAIN_POLL_SECONDS"] = "0"
    env["OLMO_RUNNING_ROOT_STALE_SECONDS"] = "1"

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
    assert seen_invocations.read_text(encoding="utf-8").strip().endswith(
        "runtime_config.yaml"
    )
