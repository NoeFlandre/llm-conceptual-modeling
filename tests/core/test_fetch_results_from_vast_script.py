from pathlib import Path


def test_fetch_results_script_supports_ssh_key_and_port_overrides() -> None:
    script_path = Path("scripts/vast/fetch_results_from_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'source "$SCRIPT_DIR/common.sh"' in script_text
    assert 'if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then' in script_text
    assert 'vast_rsync_ssh_command' in script_text
    assert 'vast_rsync_resume_flags' in script_text
    assert 'RSYNC_TIMEOUT_SECONDS="${RSYNC_TIMEOUT_SECONDS:-60}"' in script_text
    assert 'SSH_PORT="${3:-${SSH_PORT:-22}}"' in script_text
    assert 'LOCAL_RESULTS_SYNC_EXCLUDES="${LOCAL_RESULTS_SYNC_EXCLUDES:-}"' in script_text
    assert (
        'vast_require_positive_integer "$RSYNC_TIMEOUT_SECONDS" "RSYNC_TIMEOUT_SECONDS"'
        in script_text
    )
    assert 'RSYNC_RESUME_FLAGS="$(vast_rsync_resume_flags "$RSYNC_TIMEOUT_SECONDS")"' in script_text
    assert "RSYNC_EXCLUDE_FLAGS=()" in script_text
    assert "for pattern in $LOCAL_RESULTS_SYNC_EXCLUDES; do" in script_text
    assert 'RSYNC_EXCLUDE_FLAGS+=(--exclude "$pattern")' in script_text
    assert 'if [ "${#RSYNC_EXCLUDE_FLAGS[@]}" -gt 0 ]; then' in script_text
    assert '$RSYNC_RESUME_FLAGS \\' in script_text
    assert '"${RSYNC_EXCLUDE_FLAGS[@]}"' in script_text
    assert 'if [ "$rsync_status" -eq 24 ]; then' in script_text
    assert 'file vanished during sync' in script_text
    assert '"$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/' in script_text


def test_watch_results_script_retries_fetch_and_writes_health_artifacts() -> None:
    script_path = Path("scripts/vast/watch_results_from_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")
    interval_guard = (
        'vast_require_positive_integer "$SYNC_INTERVAL_SECONDS" "SYNC_INTERVAL_SECONDS"'
    )

    assert 'FETCH_SCRIPT="$SCRIPT_DIR/fetch_results_from_vast.sh"' in script_text
    assert 'SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"' in script_text
    assert (
        'SYNC_FAILURE_BACKOFF_SECONDS="${SYNC_FAILURE_BACKOFF_SECONDS:-$SYNC_INTERVAL_SECONDS}"'
        in script_text
    )
    assert (
        'LOCAL_RESULTS_SYNC_STATUS_PATH="${LOCAL_RESULTS_SYNC_STATUS_PATH:-$LOCAL_RESULTS_DIR/results-sync-status.json}"'
        in script_text
    )
    assert 'LOCAL_RESULTS_SYNC_IDENTITY="${LOCAL_RESULTS_SYNC_IDENTITY:-}"' in script_text
    assert (
        'LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="${LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH:-$LOCAL_RESULTS_DIR/results-sync-last-success.txt}"'
        in script_text
    )
    assert interval_guard in script_text
    assert 'while true; do' in script_text
    assert (
        'if SSH_PORT="$SSH_PORT" "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" '
        '"$LOCAL_RESULTS_DIR" "$SSH_PORT"; then'
        in script_text
    )
    assert 'CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))' in script_text
    assert 'results-sync-status.json' in script_text
    assert 'results-sync-last-success.txt' in script_text
    assert 'SYNC_STATE="$sync_state"' in script_text
    assert 'SYNC_MESSAGE="$message"' in script_text
    assert 'STATUS_PATH="$LOCAL_RESULTS_SYNC_STATUS_PATH"' in script_text
    assert 'WATCHER_IDENTITY_VALUE="$LOCAL_RESULTS_SYNC_IDENTITY"' in script_text
    assert 'environ["STATUS_PATH"]' in script_text
    assert '"watcher_identity": environ["WATCHER_IDENTITY_VALUE"]' in script_text
    assert 'write_sync_status "syncing" "Running result sync."' in script_text
    assert 'trap ' in script_text
    assert 'write_sync_status "stopped" "Watcher stopped."' in script_text
    assert 'sleep "$SYNC_FAILURE_BACKOFF_SECONDS"' in script_text
    assert 'sleep "$SYNC_INTERVAL_SECONDS"' in script_text
