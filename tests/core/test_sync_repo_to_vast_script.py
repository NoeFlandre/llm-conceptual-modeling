from pathlib import Path


def test_sync_repo_script_excludes_local_results_tree() -> None:
    script_path = Path("scripts/vast/sync_repo_to_vast.sh")
    script_text = script_path.read_text(encoding="utf-8")

    assert "rsync -avz \\" in script_text
    assert "--exclude '.work-venv'" in script_text
    assert "--exclude '.ruff_cache'" in script_text
    assert "--exclude 'results'" in script_text
    assert "--exclude 'data/results'" in script_text
    assert "--exclude 'data/analysis_artifacts'" in script_text
