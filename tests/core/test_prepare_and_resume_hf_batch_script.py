from pathlib import Path


def test_prepare_and_resume_script_bootstraps_and_launches_resumable_batch() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'source "$SCRIPT_DIR/common.sh"' in script_text
    assert "bootstrap_gpu_host.sh" in script_text
    assert "REMOTE_EFFECTIVE_CONFIG_PATH" in script_text
    assert ".venv/bin/lcm doctor --json" in script_text
    assert ".venv/bin/lcm run validate-config" in script_text
    assert ".venv/bin/lcm run paper-batch --config" in script_text
    assert "--resume" in script_text
    assert "nohup .venv/bin/lcm run paper-batch" in script_text


def test_prepare_and_resume_script_can_seed_remote_results_and_run_optional_smoke() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")
    seed_rsync = (
        'rsync -avz -e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/ "$SSH_TARGET:$REMOTE_RESULTS_DIR"/'
    )

    assert 'SSH_CMD=($(vast_ssh_command "$SSH_PORT" "$SSH_KEY_PATH"))' in script_text
    assert 'RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"' in script_text
    assert 'if vast_has_value "$LOCAL_RESULTS_DIR"; then' in script_text
    assert seed_rsync in script_text
    assert 'if vast_has_value "${SMOKE_ALGORITHM:-}"' in script_text
    assert ".venv/bin/lcm run smoke" in script_text
    assert "BATCH_GENERATION_TIMEOUT_SECONDS" in script_text
    assert "BATCH_RESUME_PASS_MODE" in script_text
    retry_oom_declaration = (
        'BATCH_RETRY_OOM_FAILURES_ON_RESUME="${BATCH_RETRY_OOM_FAILURES_ON_RESUME:-}"'
    )
    retry_oom_export = (
        "export BATCH_RETRY_OOM_FAILURES_ON_RESUME='$BATCH_RETRY_OOM_FAILURES_ON_RESUME'"
    )

    assert retry_oom_declaration in script_text
    assert "BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME" in script_text
    assert retry_oom_export in script_text
    assert "BATCH_RETRY_OOM_FAILURES_ON_RESUME" in script_text
    assert "BATCH_WORKER_PROCESS_MODE" in script_text
    assert "BATCH_MAX_REQUESTS_PER_WORKER_PROCESS" in script_text
    assert "runtime_config.yaml" in script_text


def test_prepare_and_resume_script_can_launch_local_results_autosync() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "LOCAL_RESULTS_SYNC_INTERVAL_SECONDS" in script_text
    assert "LOCAL_RESULTS_SYNC_LOG_PATH" in script_text
    assert "LOCAL_RESULTS_SYNC_PID_PATH" in script_text
    assert "scripts/vast/watch_results_from_vast.sh" in script_text
    assert "nohup bash" in script_text


def test_vast_common_script_centralizes_shared_shell_helpers() -> None:
    script_path = Path("scripts/vast/common.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "vast_ssh_command()" in script_text
    assert "vast_rsync_ssh_command()" in script_text
    assert "vast_has_value()" in script_text
    assert "vast_require_positive_integer()" in script_text
