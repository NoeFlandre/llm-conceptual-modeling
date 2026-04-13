import subprocess
from pathlib import Path

from llm_conceptual_modeling.paths import REPO_ROOT


def _results_root() -> Path:
    root = REPO_ROOT / "data" / "results"
    assert root.exists(), "Expected the maintained results tree to live under data/results."
    return root


def test_results_root_is_not_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-v", "data/results"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_results_root_is_flat_except_canonical_and_archive() -> None:
    root = _results_root()

    top_level = {
        path.name
        for path in root.iterdir()
        if path.name not in {".DS_Store", ".metadata_never_index", "drain-remaining-state.json"}
    }

    assert top_level == {"README.md", "archives", "frontier", "open_weights"}


def test_open_weights_root_keeps_paper_outputs_and_drops_operational_clutter() -> None:
    root = _results_root()
    open_weights = root / "open_weights"
    canonical = open_weights / "hf-paper-batch-canonical"

    assert canonical.is_dir()
    assert (canonical / "ledger.json").is_file()
    assert (canonical / "batch_status.json").is_file()
    assert (canonical / "runtime_config.yaml").is_file()
    assert (canonical / "runs").is_dir()
    assert (canonical / "variance_decomposition").is_dir()
    assert (canonical / "variance_decomposition" / "variance_decomposition.csv").is_file()
    assert (canonical / "variance_decomposition" / "variance_decomposition_algo1.csv").is_file()
    assert (canonical / "variance_decomposition" / "variance_decomposition_algo2.csv").is_file()
    assert (canonical / "variance_decomposition" / "variance_decomposition_algo3.csv").is_file()

    for name in ("preview", "preview_resume", "worker-queues"):
        assert not (canonical / name).exists()

    assert not list(canonical.glob("*.pid"))


def test_results_archive_has_expected_buckets() -> None:
    root = _results_root()
    archive = root / "archives"

    assert archive.is_dir()
    assert (archive / "olmo").is_dir()
    assert (archive / "operational").is_dir()
    assert (archive / "stale-shards").is_dir()


def test_frontier_root_contains_the_frontier_algorithm_groups() -> None:
    root = _results_root()
    frontier = root / "frontier"

    assert frontier.is_dir()
    assert (frontier / "algo1").is_dir()
    assert (frontier / "algo2").is_dir()
    assert (frontier / "algo3").is_dir()


def test_results_tree_has_no_known_junk_artifacts() -> None:
    root = _results_root()
    junk_paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.endswith(".dzzZms")
            or ".bak-" in path.name
            or path.name.endswith(".pid")
        )
    ]

    assert junk_paths == []
