import json
from datetime import UTC, datetime
from pathlib import Path

from llm_conceptual_modeling.hf_drain import (
    build_drain_plan,
    summarize_results_root_failures,
)
from llm_conceptual_modeling.hf_drain.common import (
    _can_continue_adopted_run,
    _infer_algorithm_scope,
    _infer_model_family,
    _parse_timestamp,
    _resolve_ssh_target_and_port,
    _status_is_stale,
)


def test_resolve_ssh_target_and_port_prefers_explicit_values() -> None:
    assert _resolve_ssh_target_and_port(
        ssh_command="ssh -p 2222 parsed@example.com",
        ssh_target="explicit@example.com",
        ssh_port="3333",
    ) == ("explicit@example.com", "3333")


def test_resolve_ssh_target_and_port_returns_partial_values_without_command() -> None:
    assert _resolve_ssh_target_and_port(
        ssh_command=None,
        ssh_target="explicit@example.com",
        ssh_port=None,
    ) == ("explicit@example.com", None)


def test_resolve_ssh_target_and_port_parses_target_and_port_from_command() -> None:
    assert _resolve_ssh_target_and_port(
        ssh_command="ssh -p 2222 parsed@example.com",
        ssh_target=None,
        ssh_port=None,
    ) == ("parsed@example.com", "2222")


def test_resolve_ssh_target_and_port_keeps_port_when_command_has_no_target() -> None:
    assert _resolve_ssh_target_and_port(
        ssh_command="ssh -p 2222",
        ssh_target=None,
        ssh_port=None,
    ) == (None, "2222")


def test_build_drain_plan_adopts_matching_active_root_and_orders_safe_before_risky(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()

    active_root = results_root / "hf-paper-batch-algo2-olmo-current"
    active_root.mkdir()
    waiting_root = results_root / "hf-paper-batch-algo1-qwen"
    waiting_root.mkdir()

    (active_root / "results-sync-status.json").write_text(
        json.dumps(
            {
                "status": "healthy",
                "watcher_identity": (
                    "root@example.com:2222:/workspace/results/"
                    "hf-paper-batch-algo2-olmo-current"
                ),
            }
        ),
        encoding="utf-8",
    )

    plan = build_drain_plan(
        repo_root=repo_root,
        results_root=results_root,
        ssh_target="root@example.com",
        ssh_port="2222",
        sweep_report={
            "roots": [
                {
                    "results_root": str(waiting_root),
                    "config_source": str(waiting_root / "runtime_config.yaml"),
                    "classification": "resume-ready",
                    "can_resume": True,
                    "pending_count": 80,
                    "failed_count": 4,
                    "finished_count": 10,
                    "running_count": 0,
                    "status_updated_at": "2026-04-04T12:00:00Z",
                },
                {
                    "results_root": str(active_root),
                    "config_source": str(active_root / "runtime_config.yaml"),
                    "classification": "active",
                    "can_resume": False,
                    "pending_count": 20,
                    "failed_count": 0,
                    "finished_count": 40,
                    "running_count": 1,
                    "status_updated_at": "2026-04-04T12:05:00Z",
                },
            ]
        },
        failure_summary_fn=lambda _root: {
            "retryable": {"timeout": 0, "oom": 0, "structural": 0, "infrastructure": 0},
            "terminal": {"unsupported": 0, "semantic": 0, "other": 0},
        },
    )

    queue = plan["queue"]

    assert [item["phase"] for item in queue] == ["safe", "safe", "risky", "risky"]
    assert queue[0]["results_root"] == str(active_root)
    assert queue[0]["adopt_active_run"] is True
    assert queue[0]["watcher_identity"] == (
        "root@example.com:2222:/workspace/results/hf-paper-batch-algo2-olmo-current"
    )
    assert queue[1]["results_root"] == str(waiting_root)
    assert queue[1]["profile_name"] == "qwen-safe"
    assert queue[2]["profile_name"] == "olmo-risky"
    assert queue[3]["profile_name"] == "qwen-risky"


def test_summarize_results_root_failures_separates_retryable_and_terminal_failures(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "hf-paper-batch-algo1-qwen"
    retryable_run = (
        results_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )
    retryable_run.mkdir(parents=True)
    (retryable_run / "state.json").write_text('{"status":"failed"}', encoding="utf-8")
    (retryable_run / "error.json").write_text(
        '{"type":"MonitoredCommandTimeout","message":"MonitoredCommandTimeout: startup stalled"}',
        encoding="utf-8",
    )

    terminal_run = (
        results_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "contrastive_penalty_alpha_0.8"
        / "sg1_sg2"
        / "00001"
        / "rep_00"
    )
    terminal_run.mkdir(parents=True)
    (terminal_run / "state.json").write_text('{"status":"failed"}', encoding="utf-8")
    (terminal_run / "error.json").write_text(
        (
            '{"type":"RuntimeError","message":"contrastive search is not supported '
            'with stateful models"}'
        ),
        encoding="utf-8",
    )

    summary = summarize_results_root_failures(results_root)

    assert summary["retryable"]["timeout"] == 1
    assert summary["terminal"]["unsupported"] == 1
    assert summary["retryable_total"] == 1
    assert summary["terminal_total"] == 1


def test_summarize_results_root_failures_ignores_missing_malformed_and_finished_states(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    assert summarize_results_root_failures(results_root)["retryable_total"] == 0

    runs_root = results_root / "runs"
    malformed_run = runs_root / "malformed"
    malformed_run.mkdir(parents=True)
    (malformed_run / "state.json").write_text("not-json", encoding="utf-8")
    finished_run = runs_root / "finished"
    finished_run.mkdir(parents=True)
    (finished_run / "state.json").write_text('{"status":"finished"}', encoding="utf-8")
    generic_run = runs_root / "generic"
    generic_run.mkdir(parents=True)
    (generic_run / "state.json").write_text('{"status":"failed"}', encoding="utf-8")
    (generic_run / "error.json").write_text(
        '{"type":"RuntimeError","message":"unexpected failure"}',
        encoding="utf-8",
    )
    stale_run = runs_root / "stale"
    stale_run.mkdir(parents=True)
    (stale_run / "state.json").write_text('{"status":"failed"}', encoding="utf-8")
    (stale_run / "error.json").write_text(
        '{"type":"StaleRunState","message":"stale"}',
        encoding="utf-8",
    )

    summary = summarize_results_root_failures(results_root)

    assert summary["terminal"]["other"] == 1
    assert summary["terminal"]["semantic"] == 1
    assert summary["terminal_total"] == 2


def test_drain_timestamp_and_adopted_run_checks_cover_missing_invalid_fresh_and_stale() -> None:
    assert _parse_timestamp("2026-04-04T12:00:00").tzinfo == UTC
    assert _status_is_stale({}, stale_after_seconds=60) is True
    assert _status_is_stale({"updated_at": "invalid"}, stale_after_seconds=60) is True
    assert _status_is_stale(
        {"updated_at": "2000-01-01T00:00:00Z"}, stale_after_seconds=60
    ) is True
    fresh_timestamp = datetime.now(UTC).isoformat()
    assert _status_is_stale({"updated_at": fresh_timestamp}, stale_after_seconds=60) is False

    assert _can_continue_adopted_run(
        item={"adopt_active_run": False},
        status={"running_count": 1, "updated_at": fresh_timestamp},
        stale_after_seconds=60,
    ) is False
    assert _can_continue_adopted_run(
        item={"adopt_active_run": True},
        status={"running_count": 0, "updated_at": fresh_timestamp},
        stale_after_seconds=60,
    ) is False
    assert _can_continue_adopted_run(
        item={"adopt_active_run": True},
        status={"running_count": 1, "updated_at": fresh_timestamp},
        stale_after_seconds=60,
    ) is True


def test_drain_root_name_inference_handles_known_and_unknown_names() -> None:
    assert _infer_model_family("hf-paper-batch-algo1-qwen") == "qwen"
    assert _infer_model_family("hf-paper-batch-algo1-mistral") == "mistral"
    assert _infer_model_family("hf-paper-batch-algo1-olmo") == "olmo"
    assert _infer_model_family("other") == "unknown"
    assert _infer_algorithm_scope("hf-paper-batch-algo1-qwen") == "algo1"
    assert _infer_algorithm_scope("hf-paper-batch-algo2-qwen") == "algo2"
    assert _infer_algorithm_scope("hf-paper-batch-algo3-qwen") == "algo3"
    assert _infer_algorithm_scope("other") == "unknown"


def test_build_drain_plan_skips_invalid_config_roots(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()

    valid_root = results_root / "hf-paper-batch-algo2-olmo-current"
    invalid_root = results_root / "hf-paper-batch-algo1-qwen-current"
    valid_root.mkdir()
    invalid_root.mkdir()

    plan = build_drain_plan(
        repo_root=repo_root,
        results_root=results_root,
        ssh_target="root@example.com",
        ssh_port="2222",
        sweep_report={
            "roots": [
                {
                    "results_root": str(valid_root),
                    "config_source": str(valid_root / "runtime_config.yaml"),
                    "classification": "resume-ready",
                    "can_resume": True,
                    "pending_count": 20,
                    "failed_count": 0,
                    "finished_count": 0,
                    "running_count": 0,
                    "status_updated_at": "2026-04-04T12:00:00Z",
                },
                {
                    "results_root": str(invalid_root),
                    "config_source": str(invalid_root / "runtime_config.yaml"),
                    "classification": "invalid-config",
                    "can_resume": False,
                    "pending_count": 100,
                    "failed_count": 5,
                    "finished_count": 0,
                    "running_count": 0,
                    "status_updated_at": "2026-04-04T12:00:00Z",
                },
            ]
        },
        failure_summary_fn=lambda _root: {
            "retryable": {"timeout": 0, "oom": 0, "structural": 0, "infrastructure": 0},
            "terminal": {"unsupported": 0, "semantic": 0, "other": 0},
        },
    )

    assert [Path(item["results_root"]).name for item in plan["queue"]] == [
        "hf-paper-batch-algo2-olmo-current",
        "hf-paper-batch-algo2-olmo-current",
    ]


def test_build_drain_plan_includes_profile_launch_fields(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()

    root = results_root / "hf-paper-batch-algo2-olmo-current"
    root.mkdir()

    plan = build_drain_plan(
        repo_root=repo_root,
        results_root=results_root,
        ssh_target="root@example.com",
        ssh_port="2222",
        sweep_report={
            "roots": [
                {
                    "results_root": str(root),
                    "config_source": str(root / "runtime_config.yaml"),
                    "classification": "resume-ready",
                    "can_resume": True,
                    "pending_count": 20,
                    "failed_count": 0,
                    "finished_count": 0,
                    "running_count": 0,
                    "status_updated_at": "2026-04-04T12:00:00Z",
                },
            ]
        },
        failure_summary_fn=lambda _root: {
            "retryable": {"timeout": 0, "oom": 0, "structural": 0, "infrastructure": 0},
            "terminal": {"unsupported": 0, "semantic": 0, "other": 0},
        },
    )

    safe_item = plan["queue"][0]
    assert safe_item["excluded_decoding_labels"] == ["contrastive_penalty_alpha_0.8"]
    assert safe_item["retry_timeout_failures_on_resume"] is True
    assert safe_item["retry_oom_failures_on_resume"] is True
    assert safe_item["retry_infrastructure_failures_on_resume"] is True
    assert safe_item["retry_structural_failures_on_resume"] is True
    assert safe_item["generation_timeout_seconds"] > 0
    assert safe_item["startup_timeout_seconds"] > 0
    assert safe_item["worker_process_mode"] in {"ephemeral", "persistent"}
    assert safe_item["max_requests_per_worker_process"] > 0


def test_build_drain_plan_ignores_malformed_root_report_counts(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()

    root = results_root / "hf-paper-batch-algo1-qwen"
    root.mkdir()

    plan = build_drain_plan(
        repo_root=repo_root,
        results_root=results_root,
        ssh_target="root@example.com",
        ssh_port="2222",
        sweep_report={
            "roots": [
                {
                    "results_root": str(root),
                    "config_source": str(root / "runtime_config.yaml"),
                    "classification": "needs-config-fix",
                    "can_resume": False,
                    "pending_count": 0,
                    "failed_count": 2,
                    "finished_count": 10,
                    "running_count": "false",
                    "status_updated_at": "2026-04-05T12:00:00Z",
                },
            ]
        },
        failure_summary_fn=lambda _root: {
            "retryable": {"timeout": 0, "oom": 0, "structural": 0, "infrastructure": 0},
            "terminal": {"unsupported": 0, "semantic": 0, "other": 0},
        },
    )

    assert plan["safe_queue_count"] == 1
    assert plan["queue"][0]["running_count"] == 0
    assert plan["queue"][0]["failed_count"] == 2
