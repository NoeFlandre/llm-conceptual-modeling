from __future__ import annotations

from pathlib import Path

from llm_conceptual_modeling.hf_resume_sweep import build_resume_sweep_report


def test_build_resume_sweep_report_classifies_resume_ready_failure_only_and_active_roots(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    ready_root = results_root / "hf-paper-batch-algo1-olmo-current"
    needs_fix_root = results_root / "hf-paper-batch-algo1-qwen"
    active_root = results_root / "hf-paper-batch-algo3-qwen-current"

    for root in (ready_root, needs_fix_root, active_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "batch_status.json").write_text("{}", encoding="utf-8")

    (ready_root / "preview_resume").mkdir(parents=True, exist_ok=True)
    ready_config = ready_root / "preview_resume" / "resolved_run_config.yaml"
    ready_config.write_text("ready-config", encoding="utf-8")

    needs_fix_config = needs_fix_root / "runtime_config.yaml"
    needs_fix_config.write_text("needs-fix-config", encoding="utf-8")

    (active_root / "preview_resume").mkdir(parents=True, exist_ok=True)
    active_config = active_root / "preview_resume" / "resolved_run_config.yaml"
    active_config.write_text("active-config", encoding="utf-8")

    seen_paths: list[Path] = []

    def load_config(path: Path) -> dict[str, str]:
        seen_paths.append(path)
        return {"config_path": str(path)}

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> dict:
        _ = repo_root
        _ = allow_empty
        config_path = Path(config["config_path"])  # type: ignore[index]
        if results_root == ready_root:
            assert config_path == ready_config
            return {
                "repo_root": str(repo_root),
                "results_root": str(results_root),
                "total_runs": 10,
                "finished_count": 8,
                "failed_count": 0,
                "pending_count": 2,
                "can_resume": True,
                "resume_mode": "resume",
            }
        if results_root == needs_fix_root:
            assert config_path == needs_fix_config
            return {
                "repo_root": str(repo_root),
                "results_root": str(results_root),
                "total_runs": 10,
                "finished_count": 8,
                "failed_count": 2,
                "pending_count": 0,
                "can_resume": False,
                "resume_mode": "needs-config-fix",
            }
        assert results_root == active_root
        assert config_path == active_config
        return {
            "repo_root": str(repo_root),
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 8,
            "failed_count": 0,
            "pending_count": 0,
            "running_count": 1,
            "can_resume": False,
            "resume_mode": "active",
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    assert seen_paths == [ready_config, needs_fix_config, active_config]
    assert report["root_count"] == 3
    assert report["ready_count"] == 1
    assert report["needs_config_fix_count"] == 1
    assert report["active_count"] == 1
    assert [item["classification"] for item in report["roots"]] == [
        "resume-ready",
        "needs-config-fix",
        "active",
    ]
    assert report["roots"][0]["results_root"] == str(ready_root)
    assert report["roots"][1]["results_root"] == str(needs_fix_root)
    assert report["roots"][2]["results_root"] == str(active_root)
