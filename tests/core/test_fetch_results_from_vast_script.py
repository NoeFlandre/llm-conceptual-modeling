from pathlib import Path


def test_fetch_results_script_supports_ssh_key_and_port_overrides() -> None:
    script_path = Path("scripts/vast/fetch_results_from_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'source "$SCRIPT_DIR/common.sh"' in script_text
    assert 'vast_rsync_ssh_command' in script_text
    assert 'rsync -avz -e "$RSYNC_SSH" "$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/' in script_text


def test_watch_results_script_repeats_fetch_on_interval() -> None:
    script_path = Path("scripts/vast/watch_results_from_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'FETCH_SCRIPT="$SCRIPT_DIR/fetch_results_from_vast.sh"' in script_text
    assert 'SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"' in script_text
    assert 'vast_require_positive_integer "$SYNC_INTERVAL_SECONDS" "SYNC_INTERVAL_SECONDS"' in script_text
    assert 'while true; do' in script_text
    assert '"$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"' in script_text
    assert 'sleep "$SYNC_INTERVAL_SECONDS"' in script_text
