from __future__ import annotations

from collections.abc import Iterator, Mapping
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


def test_build_resume_sweep_report_marks_invalid_configs_without_aborting(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    bad_root = results_root / "hf-paper-batch-algo1-qwen"
    good_root = results_root / "hf-paper-batch-algo1-olmo-current"

    for root in (bad_root, good_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "batch_status.json").write_text("{}", encoding="utf-8")

    bad_config = bad_root / "runtime_config.yaml"
    bad_config.write_text("broken", encoding="utf-8")
    good_config = good_root / "preview_resume" / "resolved_run_config.yaml"
    good_config.parent.mkdir(parents=True, exist_ok=True)
    good_config.write_text("good-config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
        if path == bad_config:
            raise ValueError("broken config")
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
        return {
            "repo_root": str(repo_root),
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 8,
            "failed_count": 0,
            "pending_count": 2,
            "can_resume": True,
            "resume_mode": "resume",
            "config_path": str(config["config_path"]),  # type: ignore[index]
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    assert report["root_count"] == 2
    assert report["invalid_config_count"] == 1
    assert [item["classification"] for item in report["roots"]] == [
        "resume-ready",
        "invalid-config",
    ]
    assert report["roots"][0]["results_root"] == str(good_root)
    assert report["roots"][1]["results_root"] == str(bad_root)


def test_build_resume_sweep_report_marks_output_root_mismatches_as_invalid(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    matching_root = results_root / "hf-paper-batch-algo2-olmo-current"
    mismatched_root = results_root / "hf-paper-batch-algo1-qwen-current"

    for root in (matching_root, mismatched_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "batch_status.json").write_text("{}", encoding="utf-8")
        (root / "runtime_config.yaml").write_text("config", encoding="utf-8")

    class RunConfig:
        def __init__(self, output_root: str) -> None:
            self.output_root = output_root

    class Config:
        def __init__(self, output_root: str) -> None:
            self.run = RunConfig(output_root)

    def load_config(path: Path) -> object:
        if path.parent.name == "hf-paper-batch-algo1-qwen-current":
            return Config("/workspace/results/hf-paper-batch-algo1-qwen")
        return Config(f"/workspace/results/{path.parent.name}")

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> dict:
        _ = config
        _ = repo_root
        _ = results_root
        _ = allow_empty
        return {
            "can_resume": True,
            "pending_count": 10,
            "failed_count": 0,
            "finished_count": 0,
            "running_count": 0,
            "resume_mode": "resume",
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    classifications = {
        Path(item["results_root"]).name: item["classification"] for item in report["roots"]
    }
    assert classifications["hf-paper-batch-algo2-olmo-current"] == "resume-ready"
    assert classifications["hf-paper-batch-algo1-qwen-current"] == "invalid-config"
    invalid_report = next(
        item
        for item in report["roots"]
        if Path(item["results_root"]).name == "hf-paper-batch-algo1-qwen-current"
    )
    assert "does not match results root" in str(invalid_report["error"])


def test_build_resume_sweep_report_treats_retryable_failed_only_roots_as_ready(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    root = results_root / "hf-paper-batch-algo1-qwen"
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_status.json").write_text("{}", encoding="utf-8")
    (root / "runtime_config.yaml").write_text("retry-all-config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
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
        _ = config
        return {
            "repo_root": str(repo_root),
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 8,
            "failed_count": 2,
            "pending_count": 0,
            "can_resume": True,
            "resume_mode": "resume",
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    assert report["ready_count"] == 1
    assert report["needs_config_fix_count"] == 0
    assert report["roots"][0]["classification"] == "resume-ready"
    assert report["roots"][0]["runtime_mode"] == "docker"
    assert report["roots"][0]["resume_profile"] == "qwen-safe"
    assert report["roots"][0]["rent_ready"] is True


def test_build_resume_sweep_report_does_not_treat_string_can_resume_as_truthy(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    root = results_root / "hf-paper-batch-algo1-qwen"
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_status.json").write_text("{}", encoding="utf-8")
    (root / "runtime_config.yaml").write_text("config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
        return {"config_path": str(path)}

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> dict:
        _ = config
        _ = repo_root
        _ = allow_empty
        return {
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 8,
            "failed_count": 2,
            "pending_count": 0,
            "running_count": 0,
            "can_resume": "false",
            "resume_mode": "resume",
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    assert report["ready_count"] == 0
    assert report["roots"][0]["classification"] == "needs-config-fix"
    assert report["roots"][0]["can_resume"] is False
    assert report["roots"][0]["rent_ready"] is False


def test_build_resume_sweep_report_accepts_mapping_preflight_report(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    root = results_root / "hf-paper-batch-algo1-qwen"
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_status.json").write_text("{}", encoding="utf-8")
    (root / "runtime_config.yaml").write_text("config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
        return {"config_path": str(path)}

    class PreflightReport(Mapping[str, object]):
        def __init__(self) -> None:
            self._payload = {
                "results_root": str(root),
                "total_runs": 10,
                "finished_count": 8,
                "failed_count": 0,
                "pending_count": 2,
                "running_count": 0,
                "can_resume": True,
                "resume_mode": "resume",
            }

        def __getitem__(self, key: str) -> object:
            return self._payload[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._payload)

        def __len__(self) -> int:
            return len(self._payload)

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> Mapping[str, object]:
        _ = config
        _ = repo_root
        _ = results_root
        _ = allow_empty
        return PreflightReport()

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    assert report["ready_count"] == 1
    assert report["roots"][0]["classification"] == "resume-ready"


def test_build_resume_sweep_report_ignores_malformed_running_count(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    root = results_root / "hf-paper-batch-algo1-qwen"
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_status.json").write_text("{}", encoding="utf-8")
    (root / "runtime_config.yaml").write_text("config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
        return {"config_path": str(path)}

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> dict:
        _ = config
        _ = repo_root
        _ = allow_empty
        return {
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 8,
            "failed_count": 2,
            "pending_count": 0,
            "running_count": "false",
            "can_resume": False,
            "resume_mode": "resume",
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    assert report["active_count"] == 0
    assert report["roots"][0]["classification"] == "needs-config-fix"
    assert report["roots"][0]["running_count"] == 0


def test_build_resume_sweep_report_can_filter_to_olmo_roots(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    results_root = tmp_path / "results"
    olmo_root = results_root / "hf-paper-batch-algo1-olmo-current"
    qwen_root = results_root / "hf-paper-batch-algo1-qwen"
    for root in (olmo_root, qwen_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "batch_status.json").write_text("{}", encoding="utf-8")

    (olmo_root / "runtime_config.yaml").write_text("olmo-config", encoding="utf-8")
    (qwen_root / "runtime_config.yaml").write_text("qwen-config", encoding="utf-8")

    seen_roots: list[Path] = []

    def load_config(path: Path) -> dict[str, str]:
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
        seen_roots.append(results_root)
        return {
            "repo_root": str(repo_root),
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 8,
            "failed_count": 0,
            "pending_count": 2,
            "can_resume": True,
            "resume_mode": "resume",
            "config_path": str(config["config_path"]),  # type: ignore[index]
        }

    _report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
        root_name_contains="olmo",
    )

    assert seen_roots == [olmo_root]


def test_build_resume_sweep_report_includes_rent_ready_runtime_and_profile_metadata(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    results_root = tmp_path / "results"
    root = results_root / "hf-paper-batch-algo1-olmo-current"
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_status.json").write_text("{}", encoding="utf-8")
    config_path = root / "preview_resume" / "resolved_run_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
        assert path == config_path
        return {"config_path": str(path)}

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> dict:
        _ = config
        _ = repo_root
        _ = allow_empty
        return {
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 7,
            "failed_count": 0,
            "pending_count": 3,
            "can_resume": True,
            "resume_mode": "resume",
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    root_report = report["roots"][0]
    assert root_report["runtime_mode"] == "docker"
    assert root_report["resume_profile"] == "olmo-safe"
    assert root_report["excluded_decoding_labels"] == ["contrastive_penalty_alpha_0.8"]
    assert root_report["rent_ready"] is True
    assert root_report["rent_ready_reason"] == "resume-ready"
    assert report["root_count"] == 1
    assert report["ready_count"] == 1
    assert report["roots"][0]["results_root"] == str(root)


def test_build_resume_sweep_report_recommends_largest_resume_ready_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    results_root = tmp_path / "results"
    smaller_root = results_root / "hf-paper-batch-algo2-qwen-current"
    larger_root = results_root / "hf-paper-batch-algo1-qwen-current"
    for root in (smaller_root, larger_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "batch_status.json").write_text("{}", encoding="utf-8")
        (root / "runtime_config.yaml").write_text("config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
        return {"config_path": str(path)}

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> dict:
        _ = config
        _ = repo_root
        _ = allow_empty
        if results_root == larger_root:
            return {
                "results_root": str(results_root),
                "total_runs": 100,
                "finished_count": 20,
                "failed_count": 5,
                "pending_count": 75,
                "running_count": 0,
                "can_resume": True,
                "resume_mode": "resume",
            }
        return {
            "results_root": str(results_root),
            "total_runs": 100,
            "finished_count": 90,
            "failed_count": 2,
            "pending_count": 8,
            "running_count": 0,
            "can_resume": True,
            "resume_mode": "resume",
        }

    report = build_resume_sweep_report(
        repo_root=repo_root,
        results_root=results_root,
        load_config_fn=load_config,
        preflight_fn=preflight,
    )

    assert report["recommended_results_root"] == str(larger_root)
    assert report["recommended_reason"] == "largest rent-ready backlog"


def test_build_resume_sweep_report_classifies_real_active_root_from_preflight_running_count(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    results_root = tmp_path / "results"
    root = results_root / "hf-paper-batch-algo1-qwen-current"
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_status.json").write_text("{}", encoding="utf-8")
    config_path = root / "runtime_config.yaml"
    config_path.write_text("config", encoding="utf-8")

    def load_config(path: Path) -> dict[str, str]:
        assert path == config_path
        return {"config_path": str(path)}

    def preflight(
        *,
        config: object,
        repo_root: Path,
        results_root: Path,
        allow_empty: bool,
    ) -> dict:
        _ = config
        _ = repo_root
        _ = allow_empty
        return {
            "results_root": str(results_root),
            "total_runs": 10,
            "finished_count": 6,
            "failed_count": 0,
            "pending_count": 4,
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

    root_report = report["roots"][0]
    assert root_report["classification"] == "active"
    assert root_report["active"] is True
    assert root_report["rent_ready"] is False
