from pathlib import Path

from llm_conceptual_modeling.paths import REPO_ROOT


def test_repo_tree_has_no_ds_store_files_in_source_buckets() -> None:
    junk_roots = [
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "docs",
        REPO_ROOT / "tests",
        REPO_ROOT / "paper",
        REPO_ROOT / "data",
        REPO_ROOT / "tmp",
    ]

    junk_paths: list[Path] = []
    for root in junk_roots:
        if root.exists():
            junk_paths.extend(root.rglob(".DS_Store"))

    assert junk_paths == []


def test_root_level_one_off_helper_scripts_are_removed() -> None:
    assert not (REPO_ROOT / "aggregate_results.py").exists()
    assert not (REPO_ROOT / "run_missing.bash").exists()


def test_repo_does_not_contain_accidental_ssh_workspace_mirror() -> None:
    assert not (REPO_ROOT / "ssh -p 44598 root@14.186.40.25 ").exists()


def test_tmp_tree_has_no_known_generated_artifacts() -> None:
    tmp_root = REPO_ROOT / "tmp"
    junk_names = [
        "algo2-mistral-config-preview",
        "algo2-mistral-ephemeral-preview",
        "algo2-mistral-generated-preview",
        "baseline_bundle_check",
        "paper_tables",
        "remote-configs",
        "drain_remaining_30435.log",
        "hf-paper-batch-canonical-30435.log",
        "hf-paper-batch-canonical-direct-30435.log",
        "hf-paper-batch-canonical-launch-30435.log",
        "hf-paper-batch-canonical-launch-42280.log",
        "olmo_drain_21446.log",
        "olmo_drain_21446_supervisor.log",
        "qwen_drain_30444.log",
        "qwen_results_watch_30444.log",
        "watch_qwen_30444.log",
    ]

    junk_paths = [tmp_root / name for name in junk_names if (tmp_root / name).exists()]

    assert junk_paths == []


def test_root_result_json_artifact_is_removed() -> None:
    assert not (REPO_ROOT / "--result-json").exists()
