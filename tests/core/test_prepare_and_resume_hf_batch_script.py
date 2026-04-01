from pathlib import Path


def test_prepare_and_resume_script_bootstraps_and_launches_resumable_batch() -> None:
    script_path = Path("scripts/vast/prepare_and_resume_hf_batch.sh")
    script_text = script_path.read_text(encoding="utf-8")

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

    assert 'if [ -n "$LOCAL_RESULTS_DIR" ]; then' in script_text
    assert seed_rsync in script_text
    assert 'if [ -n "${SMOKE_ALGORITHM:-}" ]' in script_text
    assert ".venv/bin/lcm run smoke" in script_text
    assert "BATCH_GENERATION_TIMEOUT_SECONDS" in script_text
    assert "BATCH_RESUME_PASS_MODE" in script_text
    assert "BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME" in script_text
    assert "runtime_config.yaml" in script_text
