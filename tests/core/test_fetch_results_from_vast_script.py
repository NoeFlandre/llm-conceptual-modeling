from pathlib import Path


def test_fetch_results_script_supports_ssh_key_and_port_overrides() -> None:
    script_path = Path("scripts/vast/fetch_results_from_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'source "$SCRIPT_DIR/common.sh"' in script_text
    assert 'vast_rsync_ssh_command' in script_text
    assert 'RSYNC_TIMEOUT_SECONDS="${RSYNC_TIMEOUT_SECONDS:-60}"' in script_text
    assert 'vast_require_positive_integer "$RSYNC_TIMEOUT_SECONDS" "RSYNC_TIMEOUT_SECONDS"' in script_text
    assert '--partial --append-verify --timeout "$RSYNC_TIMEOUT_SECONDS"' in script_text
    assert '"$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/' in script_text


def test_watch_results_script_retries_fetch_and_writes_health_artifacts() -> None:
    script_path = Path("scripts/vast/watch_results_from_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")
    interval_guard = (
        'vast_require_positive_integer "$SYNC_INTERVAL_SECONDS" "SYNC_INTERVAL_SECONDS"'
    )

    assert 'FETCH_SCRIPT="$SCRIPT_DIR/fetch_results_from_vast.sh"' in script_text
    assert 'SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"' in script_text
    assert 'SYNC_FAILURE_BACKOFF_SECONDS="${SYNC_FAILURE_BACKOFF_SECONDS:-$SYNC_INTERVAL_SECONDS}"' in script_text
    assert 'LOCAL_RESULTS_SYNC_STATUS_PATH="${LOCAL_RESULTS_SYNC_STATUS_PATH:-$LOCAL_RESULTS_DIR/results-sync-status.json}"' in script_text
    assert 'LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="${LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH:-$LOCAL_RESULTS_DIR/results-sync-last-success.txt}"' in script_text
    assert interval_guard in script_text
    assert 'while true; do' in script_text
    assert 'if "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"; then' in script_text
    assert 'CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))' in script_text
    assert 'results-sync-status.json' in script_text
    assert 'results-sync-last-success.txt' in script_text
    assert 'sleep "$SYNC_FAILURE_BACKOFF_SECONDS"' in script_text
    assert 'sleep "$SYNC_INTERVAL_SECONDS"' in script_text
