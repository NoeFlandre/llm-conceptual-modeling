from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_conceptual_modeling.hf_drain import common
from llm_conceptual_modeling.hf_resume.profile import RISKY_PHASE, ResumeProfile


def _work_item(**overrides: object) -> common.DrainWorkItem:
    values: dict[str, object] = {
        "results_root": "root",
        "config_source": "config",
        "model_family": "qwen",
        "algorithm_scope": "algo1",
        "runtime_mode": "docker",
        "profile_name": "profile",
        "phase": "safe",
        "excluded_decoding_labels": (),
        "retry_timeout_failures_on_resume": True,
        "retry_oom_failures_on_resume": True,
        "retry_infrastructure_failures_on_resume": True,
        "retry_structural_failures_on_resume": True,
        "generation_timeout_seconds": 60,
        "startup_timeout_seconds": 1800,
        "worker_process_mode": "ephemeral",
        "max_requests_per_worker_process": 1,
        "launch_priority": 1,
        "pending_count": 0,
        "failed_count": 0,
        "finished_count": 0,
        "running_count": 0,
        "status_updated_at": None,
        "retryable_failure_counts": {},
        "terminal_failure_counts": {},
        "retryable_failure_total": 0,
        "terminal_failure_total": 0,
        "watcher_identity": "",
        "adopt_active_run": False,
    }
    values.update(overrides)
    return common.DrainWorkItem(**values)


class _PathProbe:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def __truediv__(self, part: str) -> _PathProbe:
        self.parts.append(part)
        return self


class _ReadTextProbe:
    def __init__(self, content: str) -> None:
        self.content = content
        self.encodings: list[str | None] = []
        self.names: list[str] = []

    def exists(self) -> bool:
        return True

    def read_text(self, *, encoding: str | None = None) -> str:
        self.encodings.append(encoding)
        return self.content

    def with_name(self, name: str) -> _ReadTextProbe:
        self.names.append(name)
        return self


def test_summarize_failures_preserves_all_categories_and_runs_path() -> None:
    root = _PathProbe()
    state_paths = [object() for _ in range(14)]
    failure_kinds = [
        "timeout",
        "timeout",
        "oom",
        "oom",
        "infrastructure",
        "infrastructure",
        "structural",
        "structural",
        "unsupported",
        "unsupported",
        "semantic",
        "semantic",
        "other",
        "other",
    ]
    failure_kind_reader = Mock(side_effect=failure_kinds)
    with (
        patch.object(common, "_failed_run_state_paths", return_value=state_paths) as paths,
        patch.object(common, "_failed_run_failure_kind", failure_kind_reader),
    ):
        summary = common.summarize_results_root_failures(root)  # type: ignore[arg-type]

    paths.assert_called_once_with(root)
    assert [call.args[0] for call in failure_kind_reader.call_args_list] == state_paths
    assert root.parts == ["runs"]
    assert summary == {
        "retryable": {
            "timeout": 2,
            "oom": 2,
            "infrastructure": 2,
            "structural": 2,
        },
        "terminal": {"unsupported": 2, "semantic": 2, "other": 2},
        "retryable_total": 8,
        "terminal_total": 6,
    }


def test_summarize_failures_skips_unknown_runs_without_stopping() -> None:
    with (
        patch.object(common, "_failed_run_state_paths", return_value=["unknown", "known"]),
        patch.object(common, "_failed_run_failure_kind", side_effect=[None, "timeout"]),
    ):
        summary = common.summarize_results_root_failures(Path("root"))

    assert summary["retryable"] == {
        "timeout": 1,
        "oom": 0,
        "infrastructure": 0,
        "structural": 0,
    }


def test_failed_run_state_paths_handles_missing_and_nested_roots(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    assert tuple(common._failed_run_state_paths(missing_root)) == ()

    state_path = tmp_path / "runs" / "nested" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    assert list(common._failed_run_state_paths(state_path.parents[1])) == [state_path]


def test_can_continue_adopted_run_requires_active_and_fresh_status() -> None:
    fresh_status = {
        "running_count": 1,
        "updated_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    }
    assert common._can_continue_adopted_run(
        item={"adopt_active_run": True},
        status=fresh_status,
        stale_after_seconds=60,
    )
    assert not common._can_continue_adopted_run(
        item={"adopt_active_run": False},
        status=fresh_status,
        stale_after_seconds=60,
    )
    assert not common._can_continue_adopted_run(
        item={"adopt_active_run": True},
        status={"running_count": 0, "updated_at": fresh_status["updated_at"]},
        stale_after_seconds=60,
    )
    assert not common._can_continue_adopted_run(
        item={"adopt_active_run": True},
        status={
            "running_count": 1,
            "updated_at": (datetime.now(UTC) - timedelta(seconds=120)).isoformat(),
        },
        stale_after_seconds=60,
    )


def test_failed_run_failure_kind_uses_utf8_error_path_and_defaults() -> None:
    state_path = _ReadTextProbe('{"status": "failed"}')
    classify = Mock(return_value="other")
    with (
        patch.object(common.json, "loads", return_value={"status": "failed"}),
        patch.object(common, "_read_json_file", return_value={}),
        patch.object(common, "classify_failure", classify),
    ):
        result = common._failed_run_failure_kind(state_path)  # type: ignore[arg-type]

    assert result == "other"
    assert state_path.encodings == ["utf-8"]
    assert state_path.names == ["error.json"]
    classify.assert_called_once_with(error_type="RuntimeError", message="")


def test_failed_run_failure_kind_reads_type_and_message_keys_exactly() -> None:
    state_path = _ReadTextProbe('{"status": "failed"}')
    classify = Mock(return_value="semantic")
    with (
        patch.object(common.json, "loads", return_value={"status": "failed"}),
        patch.object(
            common,
            "_read_json_file",
            return_value={"type": "ValueError", "message": "bad input"},
        ),
        patch.object(common, "classify_failure", classify),
    ):
        assert common._failed_run_failure_kind(state_path) == "semantic"  # type: ignore[arg-type]

    classify.assert_called_once_with(error_type="ValueError", message="bad input")


def test_increment_failure_counts_accumulates_each_failure_kind() -> None:
    retryable = {key: 10 for key in ("timeout", "oom", "infrastructure", "structural")}
    terminal = {"unsupported": 10, "semantic": 20, "other": 30}
    for kind in (
        "timeout",
        "oom",
        "infrastructure",
        "structural",
        "timeout",
        "unsupported",
        "unsupported",
        "other",
        "other",
        "semantic",
        "semantic",
        "semantic",
    ):
        common._increment_failure_counts(
            failure_kind=kind,
            retryable_counts=retryable,
            terminal_counts=terminal,
        )

    assert retryable == {"timeout": 12, "oom": 11, "infrastructure": 11, "structural": 11}
    assert terminal == {"unsupported": 12, "semantic": 23, "other": 32}


def test_ssh_resolution_preserves_explicit_and_partial_values() -> None:
    assert common._resolve_ssh_target_and_port(
        ssh_command="ssh parsed.example -p 2200",
        ssh_target="explicit.example",
        ssh_port="2222",
    ) == ("explicit.example", "2222")
    parser = Mock(return_value=("parsed.example", "2200"))
    with patch.object(common, "_parse_ssh_command", parser):
        assert common._resolve_ssh_target_and_port(
            ssh_command="ssh command",
            ssh_target="explicit.example",
            ssh_port=None,
        ) == ("explicit.example", "2200")
        assert common._resolve_ssh_target_and_port(
            ssh_command="ssh command",
            ssh_target=None,
            ssh_port="2222",
        ) == ("parsed.example", "2222")
    assert [entry.args for entry in parser.call_args_list] == [
        ("ssh command",),
        ("ssh command",),
    ]
    assert common._resolve_ssh_target_and_port(
        ssh_command=None,
        ssh_target="explicit.example",
        ssh_port=None,
    ) == ("explicit.example", None)


def test_ssh_parser_and_port_consumer_cover_options_and_boundaries() -> None:
    assert common._parse_ssh_command("ssh -oStrictHostKeyChecking=no -p 2200 user@example") == (
        "user@example",
        "2200",
    )
    assert common._parse_ssh_command("ssh -oStrictHostKeyChecking=no") == (None, None)
    assert common._parse_ssh_command("ssh -p") == (None, None)
    assert common._parse_ssh_command("ssh -p 2200") == (None, "2200")
    assert common._parse_ssh_command("ssh -x host") == ("host", None)
    assert common._parse_ssh_command("wrapper target") == ("wrapper", None)
    assert common._consume_ssh_port(["-p", "2200"], 0) == ("2200", 2)
    assert common._consume_ssh_port(["-p", "2200", "host"], 0) == ("2200", 2)
    assert common._consume_ssh_port(["-p"], 0) == (None, 1)
    assert common._ssh_token_at(["host"], -1) is None
    assert common._ssh_token_at(["host"], 1) is None
    assert common._ssh_token_at(["host"], 0) == "host"
    assert common._is_ssh_option("ssh")
    assert common._is_ssh_option("-o")
    assert not common._is_ssh_option("host")


def test_expected_identity_requires_both_ssh_values() -> None:
    assert (
        common._expected_watcher_identity(
            ssh_target="host", ssh_port="22", results_root_name="batch"
        )
        == "host:22:/workspace/results/batch"
    )
    assert (
        common._expected_watcher_identity(ssh_target=None, ssh_port="22", results_root_name="batch")
        == ""
    )


def test_model_and_algorithm_inference_covers_all_known_and_unknown_names() -> None:
    assert common._infer_model_family("HF-OlMo-batch") == "olmo"
    assert common._infer_model_family("HF-QWEN-batch") == "qwen"
    assert common._infer_model_family("HF-MISTRAL-batch") == "mistral"
    assert common._infer_model_family("HF-unknown-batch") == "unknown"
    assert common._infer_algorithm_scope("batch-ALGO1") == "algo1"
    assert common._infer_algorithm_scope("batch-ALGO2") == "algo2"
    assert common._infer_algorithm_scope("batch-ALGO3") == "algo3"
    assert common._infer_algorithm_scope("batch-unknown") == "unknown"
    assert (
        common._expected_watcher_identity(
            ssh_target="host", ssh_port=None, results_root_name="batch"
        )
        == ""
    )


def test_sort_key_and_phase_helpers_are_exact() -> None:
    item = _work_item(
        launch_priority=3,
        pending_count=7,
        retryable_failure_total=5,
        failed_count=4,
        results_root="sorted-root",
    )
    assert common._work_item_sort_key(item) == (3, -7, -5, -4, "sorted-root")
    assert common._normalize_drain_phase("  SAFE ") == "safe"
    assert common._normalize_drain_phase(" risky ") == RISKY_PHASE
    assert common._normalize_drain_phase(" ALL ") == "all"
    with pytest.raises(ValueError, match="Unsupported drain phase"):
        common._normalize_drain_phase("unknown")
    assert common._optional_str(None) is None
    assert common._optional_str(17) == "17"


def test_order_drain_queue_sorts_by_the_work_item_priority_key() -> None:
    queue = [
        _work_item(results_root="later", launch_priority=1, pending_count=0),
        _work_item(results_root="active", launch_priority=0, pending_count=0),
        _work_item(results_root="pending", launch_priority=1, pending_count=3),
    ]
    assert [item.results_root for item in common._order_drain_queue(queue)] == [
        "active",
        "pending",
        "later",
    ]


def test_timestamp_now_uses_utc_and_the_canonical_wire_format() -> None:
    frozen = datetime(2026, 8, 22, 13, 14, 15, tzinfo=UTC)
    datetime_probe = Mock()
    datetime_probe.now.return_value = frozen
    with patch.object(common, "datetime", datetime_probe):
        assert common._timestamp_now() == "2026-08-22T13:14:15Z"
    datetime_probe.now.assert_called_once_with(UTC)


def test_parse_timestamp_normalizes_z_offsets_and_naive_values() -> None:
    parsed_z = common._parse_timestamp("2026-08-22T12:00:00Z")
    assert parsed_z == datetime(2026, 8, 22, 12, tzinfo=UTC)
    assert parsed_z.tzinfo is UTC
    parsed_offset = common._parse_timestamp("2026-08-22T12:00:00+02:00")
    assert parsed_offset == datetime(2026, 8, 22, 10, tzinfo=UTC)
    assert parsed_offset.tzinfo is UTC
    parsed_naive = common._parse_timestamp("2026-08-22T12:00:00")
    assert parsed_naive == datetime(2026, 8, 22, 12, tzinfo=UTC)
    assert parsed_naive.tzinfo is UTC

    parsed_inputs: list[str] = []

    class DateTimeProbe:
        @staticmethod
        def fromisoformat(value: str) -> datetime:
            parsed_inputs.append(value)
            return datetime(2026, 8, 22, 12, tzinfo=UTC)

    with patch.object(common, "datetime", DateTimeProbe):
        common._parse_timestamp("2026-08-22T12:00:00Z")
    assert parsed_inputs == ["2026-08-22T12:00:00+00:00"]


def test_json_helpers_use_canonical_paths_and_utf8() -> None:
    path = _ReadTextProbe('{"value": 1}')
    with patch.object(common.json, "loads", return_value={"value": 1}):
        assert common._read_json_file(path) == {"value": 1}  # type: ignore[arg-type]
    assert path.encodings == ["utf-8"]

    missing = Mock()
    missing.exists.return_value = False
    assert common._read_json_file(missing) == {}

    root = _PathProbe()
    with patch.object(common, "_read_json_file", return_value={"status": "ok"}) as reader:
        assert common.read_results_sync_status(root) == {"status": "ok"}  # type: ignore[arg-type]
    reader.assert_called_once_with(root)
    assert root.parts == ["results-sync-status.json"]


def test_root_work_detection_requires_positive_counts_or_any_failure_total() -> None:
    empty_failure = {"retryable_total": 0, "terminal_total": 0}
    assert common._root_has_work({}, empty_failure) is False
    assert common._root_has_work({"running_count": 1}, empty_failure) is True
    assert common._root_has_work({"pending_count": 1}, empty_failure) is True
    assert common._root_has_work({"failed_count": 1}, empty_failure) is True
    assert common._root_has_work({}, {"retryable_total": 1, "terminal_total": 0}) is True
    assert common._root_has_work({}, {"retryable_total": 0, "terminal_total": 1}) is True
    assert common._root_has_work({"running_count": 0}, empty_failure) is False

    class GetProbe(dict[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, object]] = []

        def get(self, key: str, default: object = None) -> object:
            self.calls.append((key, default))
            return super().get(key, default)

    report = GetProbe()
    assert common._root_has_work(report, empty_failure) is False
    assert report.calls == [
        ("running_count", 0),
        ("pending_count", 0),
        ("failed_count", 0),
    ]


def test_risky_phase_decision_covers_full_safe_and_failed_cases() -> None:
    safe_profile = ResumeProfile(
        profile_name="safe",
        phase="safe",
        runtime_mode="docker",
        excluded_decoding_labels=(),
        retry_timeout_failures_on_resume=True,
        retry_oom_failures_on_resume=True,
        retry_infrastructure_failures_on_resume=True,
        retry_structural_failures_on_resume=True,
        generation_timeout_seconds=60,
        startup_timeout_seconds=1800,
        worker_process_mode="ephemeral",
        max_requests_per_worker_process=1,
    )
    excluded_profile = ResumeProfile(
        **{
            **safe_profile.__dict__,
            "excluded_decoding_labels": ("beam",),
        }
    )
    assert common._needs_risky_phase({"failed_count": 9}, safe_profile, full_coverage=True) is False
    assert common._needs_risky_phase({}, excluded_profile, full_coverage=False) is True
    assert (
        common._needs_risky_phase({"failed_count": 0}, safe_profile, full_coverage=False) is False
    )
    assert common._needs_risky_phase({"failed_count": 1}, safe_profile, full_coverage=False) is True

    class GetProbe(dict[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, object]] = []

        def get(self, key: str, default: object = None) -> object:
            self.calls.append((key, default))
            return super().get(key, default)

    report = GetProbe()
    assert common._needs_risky_phase(report, safe_profile, full_coverage=False) is False
    assert report.calls == [("failed_count", 0)]


def test_active_root_adoption_checks_identity_and_allowed_states() -> None:
    for state in ("starting", "syncing", "healthy"):
        assert (
            common._should_adopt_active_root(
                root_report={"classification": "active"},
                watcher_status={"watcher_identity": "watcher", "status": state},
                expected_identity="watcher",
            )
            is True
        )
    assert (
        common._should_adopt_active_root(
            root_report={"classification": "complete"},
            watcher_status={"watcher_identity": "watcher", "status": "healthy"},
            expected_identity="watcher",
        )
        is False
    )
    assert (
        common._should_adopt_active_root(
            root_report={"classification": "active"},
            watcher_status={"watcher_identity": "other", "status": "healthy"},
            expected_identity="watcher",
        )
        is False
    )
    assert (
        common._should_adopt_active_root(
            root_report={"classification": "active"},
            watcher_status={"watcher_identity": "watcher", "status": "stopped"},
            expected_identity="watcher",
        )
        is False
    )

    class GetProbe(dict[str, str]):
        def __init__(self) -> None:
            super().__init__(watcher_identity="watcher", status="healthy")
            self.calls: list[tuple[str, object]] = []

        def get(self, key: str, default: object = None) -> object:
            self.calls.append((key, default))
            return super().get(key, default)

    watcher_status = GetProbe()
    assert (
        common._should_adopt_active_root(
            root_report={"classification": "active"},
            watcher_status=watcher_status,
            expected_identity="watcher",
        )
        is True
    )
    assert watcher_status.calls == [("watcher_identity", ""), ("status", "")]


def test_status_stale_boundary_is_inclusive() -> None:
    frozen = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timestamp = (frozen - timedelta(seconds=60)).isoformat()

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FrozenDateTime:
            return cls.fromtimestamp(frozen.timestamp(), tz=tz)  # type: ignore[arg-type]

    with patch.object(common, "datetime", FrozenDateTime):
        assert common._status_is_stale({"updated_at": timestamp}, 60) is True


def test_status_stale_checks_missing_invalid_fresh_and_old_timestamps() -> None:
    frozen = datetime(2026, 8, 22, 12, tzinfo=UTC)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FrozenDateTime:
            return cls.fromtimestamp(frozen.timestamp(), tz=tz)  # type: ignore[arg-type]

    with patch.object(common, "datetime", FrozenDateTime):
        assert common._status_is_stale({}, 60) is True
        assert common._status_is_stale({"updated_at": "not-a-timestamp"}, 60) is True
        assert (
            common._status_is_stale(
                {"updated_at": (frozen - timedelta(seconds=10)).isoformat()}, 60
            )
            is False
        )
        assert (
            common._status_is_stale(
                {"updated_at": (frozen - timedelta(seconds=120)).isoformat()}, 60
            )
            is True
        )

    class GetProbe(dict[str, str]):
        def get(self, key: str, default: object = None) -> object:
            assert key == "updated_at"
            return super().get(key, default)

    assert common._status_is_stale(GetProbe(), 60) is True


def test_build_work_item_maps_every_profile_root_and_failure_field() -> None:
    profile = ResumeProfile(
        profile_name="profile-x",
        phase=RISKY_PHASE,
        runtime_mode="runtime-x",
        excluded_decoding_labels=("label-a", "label-b"),
        retry_timeout_failures_on_resume=False,
        retry_oom_failures_on_resume=True,
        retry_infrastructure_failures_on_resume=False,
        retry_structural_failures_on_resume=True,
        generation_timeout_seconds=17,
        startup_timeout_seconds=19,
        worker_process_mode="worker-x",
        max_requests_per_worker_process=23,
    )
    failure_summary = {
        "retryable": {"timeout": 2, "oom": 3},
        "terminal": {"semantic": 5},
        "retryable_total": 7,
        "terminal_total": 11,
    }
    item = common._build_work_item(
        root_report={
            "results_root": "/tmp/qwen-algo3-batch",
            "classification": "active",
            "pending_count": 2,
            "failed_count": 3,
            "finished_count": 4,
            "running_count": 5,
            "status_updated_at": 29,
        },
        profile=profile,
        config_source="config-x",
        watcher_identity="watcher-x",
        failure_summary=failure_summary,
        adopt_active_run=True,
    )
    assert item == common.DrainWorkItem(
        results_root="/tmp/qwen-algo3-batch",
        config_source="config-x",
        model_family="qwen",
        algorithm_scope="algo3",
        runtime_mode="runtime-x",
        profile_name="profile-x",
        phase=RISKY_PHASE,
        excluded_decoding_labels=("label-a", "label-b"),
        retry_timeout_failures_on_resume=False,
        retry_oom_failures_on_resume=True,
        retry_infrastructure_failures_on_resume=False,
        retry_structural_failures_on_resume=True,
        generation_timeout_seconds=17,
        startup_timeout_seconds=19,
        worker_process_mode="worker-x",
        max_requests_per_worker_process=23,
        launch_priority=0,
        pending_count=2,
        failed_count=3,
        finished_count=4,
        running_count=5,
        status_updated_at="29",
        retryable_failure_counts={"timeout": 2, "oom": 3},
        terminal_failure_counts={"semantic": 5},
        retryable_failure_total=7,
        terminal_failure_total=11,
        watcher_identity="watcher-x",
        adopt_active_run=True,
    )


def test_build_work_item_uses_failure_count_fallbacks_and_inactive_priority() -> None:
    profile = ResumeProfile(
        profile_name="profile",
        phase="safe",
        runtime_mode="docker",
        excluded_decoding_labels=(),
        retry_timeout_failures_on_resume=True,
        retry_oom_failures_on_resume=True,
        retry_infrastructure_failures_on_resume=True,
        retry_structural_failures_on_resume=True,
        generation_timeout_seconds=60,
        startup_timeout_seconds=1800,
        worker_process_mode="ephemeral",
        max_requests_per_worker_process=1,
    )
    item = common._build_work_item(
        root_report={"results_root": "/tmp/unknown"},
        profile=profile,
        config_source="config",
        watcher_identity="",
        failure_summary={
            "retryable": {"timeout": 2},
            "terminal": {"other": 3},
        },
        adopt_active_run=False,
    )
    assert item.retryable_failure_total == 2
    assert item.terminal_failure_total == 3
    assert item.launch_priority == 1
    assert item.pending_count == 0
    assert item.failed_count == 0
    assert item.finished_count == 0
    assert item.running_count == 0
    assert item.status_updated_at is None


def test_build_work_item_uses_explicit_defaults_for_missing_root_fields() -> None:
    profile = ResumeProfile(
        profile_name="profile",
        phase="safe",
        runtime_mode="docker",
        excluded_decoding_labels=(),
        retry_timeout_failures_on_resume=True,
        retry_oom_failures_on_resume=True,
        retry_infrastructure_failures_on_resume=True,
        retry_structural_failures_on_resume=True,
        generation_timeout_seconds=60,
        startup_timeout_seconds=1800,
        worker_process_mode="ephemeral",
        max_requests_per_worker_process=1,
    )

    class GetProbe(dict[str, object]):
        def __init__(self) -> None:
            super().__init__(results_root="/tmp/unknown")
            self.calls: list[tuple[str, object]] = []

        def get(self, key: str, default: object = None) -> object:
            self.calls.append((key, default))
            return super().get(key, default)

    root_report = GetProbe()
    common._build_work_item(
        root_report=root_report,
        profile=profile,
        config_source="config",
        watcher_identity="",
        failure_summary={"retryable": {}, "terminal": {}},
        adopt_active_run=False,
    )
    assert root_report.calls == [
        ("classification", ""),
        ("pending_count", 0),
        ("failed_count", 0),
        ("finished_count", 0),
        ("running_count", 0),
        ("status_updated_at", None),
    ]


def test_work_item_payload_serializes_excluded_labels_as_a_list() -> None:
    item = _work_item(excluded_decoding_labels=("beam", "contrastive"))
    payload = common._work_item_to_payload(item)
    assert payload["excluded_decoding_labels"] == ["beam", "contrastive"]
