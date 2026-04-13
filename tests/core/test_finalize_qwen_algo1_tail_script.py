from pathlib import Path


def test_finalize_qwen_algo1_tail_script_merges_runs_and_refreshes_canonical_ledger() -> None:
    script_text = Path("scripts/vast/finalize_qwen_algo1_tail.sh").read_text(encoding="utf-8")

    assert 'mkdir -p "$CANONICAL_RESULTS_ROOT/runs"' in script_text
    assert 'rsync -av \\' in script_text
    assert '"$LOCAL_TAIL_RESULTS_ROOT/runs/" \\' in script_text
    assert '"$CANONICAL_RESULTS_ROOT/runs/"' in script_text
    assert 'uv --directory "$LOCAL_REPO_DIR" run lcm run refresh-ledger \\' in script_text
    assert '--results-root "$(dirname "$CANONICAL_RESULTS_ROOT")" \\' in script_text
    assert '--ledger-root "$CANONICAL_RESULTS_ROOT" \\' in script_text
