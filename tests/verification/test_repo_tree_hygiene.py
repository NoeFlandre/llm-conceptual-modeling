from pathlib import Path
from subprocess import run

from llm_conceptual_modeling.paths import REPO_ROOT


def test_repo_tree_has_no_tracked_ds_store_files_in_source_buckets() -> None:
    tracked_roots = [
        "src",
        "scripts",
        "docs",
        "tests",
        "paper",
        "data",
        "tmp",
    ]

    completed = run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *tracked_roots],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_paths = [
        Path(line)
        for line in completed.stdout.splitlines()
        if line.endswith(".DS_Store")
    ]

    assert tracked_paths == []


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


def test_source_packages_have_local_readmes() -> None:
    package_dirs = [
        "src/llm_conceptual_modeling",
        "src/llm_conceptual_modeling/analysis",
        "src/llm_conceptual_modeling/algo1",
        "src/llm_conceptual_modeling/algo2",
        "src/llm_conceptual_modeling/algo3",
        "src/llm_conceptual_modeling/commands",
        "src/llm_conceptual_modeling/common",
        "src/llm_conceptual_modeling/hf_batch",
        "src/llm_conceptual_modeling/hf_config",
        "src/llm_conceptual_modeling/hf_drain",
        "src/llm_conceptual_modeling/hf_execution",
        "src/llm_conceptual_modeling/hf_pipeline",
        "src/llm_conceptual_modeling/hf_resume",
        "src/llm_conceptual_modeling/hf_state",
        "src/llm_conceptual_modeling/hf_tail",
        "src/llm_conceptual_modeling/hf_worker",
        "src/llm_conceptual_modeling/verification",
        "docs",
    ]

    missing_readmes = [
        REPO_ROOT / package_dir / "README.md"
        for package_dir in package_dirs
        if not (REPO_ROOT / package_dir / "README.md").exists()
    ]

    assert missing_readmes == []


def test_revision_summary_docs_are_removed() -> None:
    assert not (REPO_ROOT / "docs" / "revision-summary.md").exists()
    assert not (REPO_ROOT / "docs" / "revision-summary-compact.md").exists()
