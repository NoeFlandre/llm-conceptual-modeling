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
    seed_rsync = (
        'rsync -avz -e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/ "$SSH_TARGET:$REMOTE_RESULTS_DIR"/'
    )

    assert 'SSH_CMD=($(vast_ssh_command "$SSH_PORT" "$SSH_KEY_PATH"))' in script_text
    assert 'RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"' in script_text
    assert 'if vast_has_value "$LOCAL_RESULTS_DIR"; then' in script_text
    assert "--exclude '.work-venv'" in script_text
    assert "--exclude '.ruff_cache'" in script_text
    assert "--exclude 'results'" in script_text
    assert '"${SSH_CMD[@]}" "$SSH_TARGET" "mkdir -p \'$REMOTE_RESULTS_DIR\'"' in script_text
    assert seed_rsync in script_text
    assert 'if vast_has_value "${SMOKE_ALGORITHM:-}"' in script_text
    assert ".venv/bin/lcm run smoke" in script_text
    assert "BATCH_GENERATION_TIMEOUT_SECONDS" in script_text
    assert 'BATCH_STARTUP_TIMEOUT_SECONDS="${BATCH_STARTUP_TIMEOUT_SECONDS:-}"' in script_text
    assert "BATCH_RESUME_PASS_MODE" in script_text
    retry_oom_declaration = (
        'BATCH_RETRY_OOM_FAILURES_ON_RESUME="${BATCH_RETRY_OOM_FAILURES_ON_RESUME:-}"'
    )
    assert retry_oom_declaration in script_text
    assert "BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME" in script_text
    assert "BATCH_RETRY_OOM_FAILURES_ON_RESUME" in script_text
    assert "BATCH_WORKER_PROCESS_MODE" in script_text
    assert "BATCH_MAX_REQUESTS_PER_WORKER_PROCESS" in script_text
    assert "runtime_config.yaml" in script_text
    assert "context_policy['startup_timeout_seconds'] = float(timeout_value)" in (
        Path("scripts/vast/remote_resume_preview.sh").read_text(encoding="utf-8")
    )


def test_prepare_and_resume_script_can_launch_local_results_autosync() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert (
        'LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-60}"'
        in script_text
    )
    assert "LOCAL_RESULTS_SYNC_LOG_PATH" in script_text
    assert "LOCAL_RESULTS_SYNC_PID_PATH" in script_text
    assert "LOCAL_RESULTS_SYNC_STATUS_PATH" in script_text
    assert "LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH" in script_text
    assert "scripts/vast/watch_results_from_vast.sh" in script_text
    assert "nohup bash" in script_text
    assert 'if vast_has_value "$LOCAL_RESULTS_DIR"; then' in script_text


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


def test_remote_resume_preview_script_writes_and_validates_effective_config() -> None:
    script_path = Path("scripts/vast/remote_resume_preview.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "REMOTE_CONFIG_PATH" in script_text
    assert "REMOTE_EFFECTIVE_CONFIG_PATH" in script_text
    assert "REMOTE_PREVIEW_DIR" in script_text
    assert "python3 - \"$REMOTE_CONFIG_PATH\" \"$REMOTE_EFFECTIVE_CONFIG_PATH\"" in script_text
    assert "context_policy['startup_timeout_seconds'] = float(timeout_value)" in script_text
    assert ".venv/bin/lcm doctor --json --results-root" in script_text
    assert ".venv/bin/lcm run validate-config --config" in script_text


def test_remote_resume_launch_script_restarts_the_batch_process() -> None:
    script_path = Path("scripts/vast/remote_resume_launch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "REMOTE_RUN_LOG" in script_text
    assert "REMOTE_PID_PATH" in script_text
    assert "pkill -f 'llm_conceptual_modeling.hf_worker'" in script_text
    assert "nohup .venv/bin/lcm run paper-batch --config" in script_text
    assert "--resume" in script_text
    assert "pgrep -n -f" in script_text


def test_vast_common_script_centralizes_shared_shell_helpers() -> None:
    script_path = Path("scripts/vast/common.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "vast_ssh_transport_flags()" in script_text
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
