from pathlib import Path


def test_prepare_and_resume_qwen_algo1_tail_script_uses_dedicated_tail_commands() -> None:
    script_text = Path("scripts/vast/prepare_and_resume_qwen_algo1_tail.sh").read_text(
        encoding="utf-8"
    )

    assert 'source "$SCRIPT_DIR/common.sh"' in script_text
    assert "prepare-qwen-algo1-tail" in script_text
    assert "qwen-algo1-tail-preflight" in script_text
    assert "--canonical-results-root" in script_text
    assert "--tail-results-root" in script_text
    assert "--remote-output-root" in script_text
    assert "REMOTE_TAIL_RESULTS_ROOT" in script_text
    assert "LOCAL_TAIL_RESULTS_ROOT" in script_text
    assert "watch_results_from_vast.sh" in script_text


def test_prepare_and_resume_qwen_algo1_tail_script_exports_pythonpath_for_remote_batch() -> None:
    script_text = Path("scripts/vast/prepare_and_resume_qwen_algo1_tail.sh").read_text(
        encoding="utf-8"
    )

    assert 'export PYTHONPATH=\\"$REMOTE_REPO_DIR/src\\${PYTHONPATH:+:\\$PYTHONPATH}\\"' in script_text
    assert '\\"$REMOTE_REPO_DIR/.venv/bin/lcm\\" run paper-batch --config \\"$REMOTE_TAIL_RESULTS_ROOT/runtime_config.yaml\\" --resume' in script_text
    assert "remote_resume_preview.sh" in script_text
    assert "remote_runtime_doctor.sh" in script_text
    assert "bootstrap_gpu_host.sh" in script_text


def test_prepare_and_resume_qwen_algo1_tail_script_targets_dedicated_watcher_root() -> None:
    script_text = Path("scripts/vast/prepare_and_resume_qwen_algo1_tail.sh").read_text(
        encoding="utf-8"
    )

    assert 'LOCAL_RESULTS_SYNC_STATUS_PATH="${LOCAL_RESULTS_SYNC_STATUS_PATH:-$LOCAL_TAIL_RESULTS_ROOT/results-sync-status.json}"' in script_text
    assert 'LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="${LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH:-$LOCAL_TAIL_RESULTS_ROOT/results-sync-last-success.txt}"' in script_text
    assert 'nohup bash "$LOCAL_REPO_DIR/scripts/vast/watch_results_from_vast.sh" \\' in script_text
    assert '  "$SSH_TARGET:$REMOTE_TAIL_RESULTS_ROOT" \\' in script_text
    assert '  "$LOCAL_TAIL_RESULTS_ROOT" \\' in script_text
