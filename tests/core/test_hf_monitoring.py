from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_conceptual_modeling.hf_batch import monitoring


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_current_run_payload_preserves_all_fields_and_default_graph_source() -> None:
    payload = monitoring.current_run_payload(
        algorithm="algo3",
        model="Qwen/Qwen3.5-9B",
        decoding_algorithm="beam",
        graph_source="clarice_starling",
        pair_name="subgraph_1_to_subgraph_3",
        condition_bits="101",
        replication=4,
    )

    assert payload == {
        "algorithm": "algo3",
        "model": "Qwen/Qwen3.5-9B",
        "decoding_algorithm": "beam",
        "graph_source": "clarice_starling",
        "pair_name": "subgraph_1_to_subgraph_3",
        "condition_bits": "101",
        "replication": 4,
    }

    default_payload = monitoring.current_run_payload(
        algorithm="algo1",
        model="model",
        decoding_algorithm="greedy",
        pair_name="sg1_sg2",
        condition_bits="000",
        replication=0,
    )
    assert default_payload["graph_source"] == "default"


@pytest.mark.parametrize(
    ("status", "attribute"),
    [
        ("finished", "finished_count"),
        ("running", "running_count"),
        ("pending", "pending_count"),
    ],
)
def test_update_run_counts_accumulates_each_non_failed_state(
    tmp_path: Path,
    status: str,
    attribute: str,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "state.json", {"status": status})
    counts = monitoring._RunCounts()
    setattr(counts, attribute, 7)

    monitoring._update_run_counts(counts, run_dir)

    assert getattr(counts, attribute) == 8
    if status == "running":
        assert counts.running_run_dirs == [run_dir]


def test_update_run_counts_defaults_missing_status_to_pending(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "state.json", {})
    counts = monitoring._RunCounts(pending_count=7)

    monitoring._update_run_counts(counts, run_dir)

    assert counts.pending_count == 8


def test_update_run_counts_reads_canonical_state_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    observed: list[Path] = []

    def read_json(path: Path) -> dict[str, object]:
        observed.append(path)
        return {"status": "finished"}

    monkeypatch.setattr(monitoring, "_read_json", read_json)
    counts = monitoring._RunCounts()

    monitoring._update_run_counts(counts, run_dir)

    assert observed == [run_dir / "state.json"]
    assert counts.finished_count == 1


def test_update_run_counts_delegates_failed_runs_with_original_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    observed: list[object] = []

    def update_failed(counts: monitoring._RunCounts, failed_run_dir: Path) -> None:
        observed.extend([counts, failed_run_dir])

    monkeypatch.setattr(
        monitoring,
        "_read_json",
        lambda _path: {"status": "failed"},
    )
    monkeypatch.setattr(monitoring, "_update_failed_run_counts", update_failed)
    counts = monitoring._RunCounts()

    monitoring._update_run_counts(counts, run_dir)

    assert observed == [counts, run_dir]


def test_update_failed_run_counts_records_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    observed: dict[str, str] = {}

    def classify_failure(*, error_type: str, message: str) -> str:
        observed.update(error_type=error_type, message=message)
        return "terminal"

    def read_json(path: Path) -> dict[str, object]:
        assert path == run_dir / "error.json"
        return {"type": "ValueError", "message": "invalid output"}

    monkeypatch.setattr(monitoring, "_read_json", read_json)
    monkeypatch.setattr(monitoring, "classify_failure", classify_failure)
    counts = monitoring._RunCounts(failed_count=4)

    monitoring._update_failed_run_counts(counts, run_dir)

    assert observed == {"error_type": "ValueError", "message": "invalid output"}
    assert counts.failed_count == 5
    assert counts.failures == [
        {
            "run_dir": str(run_dir),
            "message": "invalid output",
            "type": "ValueError",
        }
    ]


def test_update_failed_run_counts_requeues_retryable_error_and_normalizes_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "error.json", {})
    observed: dict[str, str] = {}

    def classify_failure(*, error_type: str, message: str) -> str:
        observed.update(error_type=error_type, message=message)
        return "timeout"

    monkeypatch.setattr(monitoring, "classify_failure", classify_failure)
    counts = monitoring._RunCounts(pending_count=4)

    monitoring._update_failed_run_counts(counts, run_dir)

    assert observed == {"error_type": "", "message": ""}
    assert counts.pending_count == 5
    assert counts.failed_count == 0
    assert counts.failures == []


def test_resolve_counts_uses_explicit_values_and_distinct_fallbacks() -> None:
    run_counts = monitoring._RunCounts(
        finished_count=2,
        failed_count=3,
        running_count=4,
        pending_count=5,
    )

    partial = monitoring._resolve_counts(
        status_file={"failed_count": "10"},
        run_counts=run_counts,
        run_count=9,
        planned_total_runs=8,
        manifest_identities=set(),
    )
    assert partial == monitoring._ResolvedCounts(
        total_runs=9,
        finished_count=2,
        failed_count=10,
        running_count=4,
        pending_count=5,
    )

    missing_failed = monitoring._resolve_counts(
        status_file={"finished_count": "10"},
        run_counts=run_counts,
        run_count=9,
        planned_total_runs=8,
        manifest_identities=set(),
    )
    assert missing_failed.failed_count == 3

    explicit = monitoring._resolve_counts(
        status_file={
            "finished_count": 10,
            "failed_count": 11,
            "running_count": 12,
            "pending_count": 13,
            "total_runs": 14,
        },
        run_counts=run_counts,
        run_count=99,
        planned_total_runs=98,
        manifest_identities=set(),
    )
    assert explicit == monitoring._ResolvedCounts(
        total_runs=14,
        finished_count=10,
        failed_count=11,
        running_count=12,
        pending_count=13,
    )


def test_resolve_counts_uses_manifest_identity_count_for_inferred_plan() -> None:
    resolved = monitoring._resolve_counts(
        status_file={},
        run_counts=monitoring._RunCounts(),
        run_count=3,
        planned_total_runs=4,
        manifest_identities={"first", "second"},
    )

    assert resolved == monitoring._ResolvedCounts(
        total_runs=2,
        finished_count=0,
        failed_count=0,
        running_count=0,
        pending_count=2,
    )


def test_resolve_counts_subtracts_all_observed_states_from_inferred_total() -> None:
    resolved = monitoring._resolve_counts(
        status_file={},
        run_counts=monitoring._RunCounts(
            finished_count=2,
            failed_count=3,
            running_count=4,
            pending_count=1,
        ),
        run_count=20,
        planned_total_runs=0,
        manifest_identities=set(),
    )

    assert resolved.pending_count == 11


def test_resolve_final_total_runs_uses_status_then_preview_fallbacks() -> None:
    assert (
        monitoring._resolve_final_total_runs(
            total_runs=0,
            status_file={"total_runs": 6},
            planned_total_runs=7,
        )
        == 6
    )
    assert (
        monitoring._resolve_final_total_runs(
            total_runs=0,
            status_file={},
            planned_total_runs=7,
        )
        == 7
    )
    assert (
        monitoring._resolve_final_total_runs(
            total_runs=0,
            status_file={},
            planned_total_runs=0,
        )
        == 0
    )
    assert (
        monitoring._resolve_final_total_runs(
            total_runs=3,
            status_file={"total_runs": 6},
            planned_total_runs=7,
        )
        == 3
    )


def test_infer_total_runs_uses_preview_when_run_tree_and_status_are_empty() -> None:
    assert (
        monitoring._infer_total_runs(
            run_count=0,
            status_file={},
            planned_total_runs=7,
            manifest_identities=set(),
        )
        == 7
    )


@pytest.mark.parametrize(
    "key", ["finished_count", "failed_count", "running_count", "pending_count"]
)
def test_has_explicit_batch_counts_accepts_each_count_field(key: str) -> None:
    assert monitoring._has_explicit_batch_counts({key: 0}) is True


def test_has_explicit_batch_counts_rejects_unrelated_fields() -> None:
    assert monitoring._has_explicit_batch_counts({"total_runs": 0}) is False


def test_status_payload_preserves_metadata_and_aggregates_failure_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_processes = [{"pid": 123, "used_gpu_memory_mib": 456}]
    monkeypatch.setattr(monitoring, "_query_gpu_processes", lambda: gpu_processes)
    counts = monitoring._ResolvedCounts(
        total_runs=7,
        finished_count=1,
        failed_count=1,
        running_count=1,
        pending_count=4,
    )
    status_file = {
        "current_run": {"id": "current"},
        "last_completed_run": {"id": "previous"},
        "started_at": "2026-08-22T10:00:00+00:00",
        "updated_at": "2026-08-22T10:01:00+00:00",
    }
    active_run = {
        "active_stage": {"schema_name": "edge_list"},
        "active_stage_age_seconds": 1.25,
        "worker_pid": 999,
        "worker_status": "running",
        "worker_loaded_model": True,
    }
    failures = [
        {"type": "ValueError", "message": "bad"},
        {"type": "ValueError", "message": "worse"},
        {"message": "missing type"},
    ]

    payload = monitoring._status_payload(
        status_file=status_file,
        counts=counts,
        failures=failures,
        active_run=active_run,
    )

    assert payload == {
        "total_runs": 7,
        "finished_count": 1,
        "failed_count": 1,
        "running_count": 1,
        "pending_count": 4,
        "failure_count": 3,
        "failures": failures,
        "percent_complete": 14.29,
        "current_run": {"id": "current"},
        "last_completed_run": {"id": "previous"},
        "started_at": "2026-08-22T10:00:00+00:00",
        "updated_at": "2026-08-22T10:01:00+00:00",
        "active_stage": {"schema_name": "edge_list"},
        "active_stage_age_seconds": 1.25,
        "worker_pid": 999,
        "worker_status": "running",
        "worker_loaded_model": True,
        "gpu_processes": gpu_processes,
        "failure_types": {"ValueError": 2},
    }


def test_collect_batch_status_reads_canonical_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    output_root.mkdir()
    observed: dict[str, object] = {}

    def read_json(path: Path) -> dict[str, object]:
        observed["status_path"] = path
        if path == output_root / "batch_status.json":
            return {"current_run": {"id": "current"}}
        return {}

    monkeypatch.setattr(monitoring, "_read_json", read_json)

    monkeypatch.setattr(monitoring, "resolve_active_chat_model_slugs", lambda _path: set())
    manifest_identities: set[object] = {"identity"}

    def load_manifest(path: Path) -> set[object]:
        observed["manifest_path"] = path
        return manifest_identities

    def resolve_counts(**kwargs: object) -> monitoring._ResolvedCounts:
        observed["manifest_for_counts"] = kwargs["manifest_identities"]
        return monitoring._ResolvedCounts(
            total_runs=0,
            finished_count=0,
            failed_count=0,
            running_count=0,
            pending_count=0,
        )

    monkeypatch.setattr(monitoring, "_load_manifest_identity_keys", load_manifest)
    monkeypatch.setattr(monitoring, "_load_planned_total_runs", lambda _path: 0)
    monkeypatch.setattr(monitoring, "_matching_run_directories", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(monitoring, "_resolve_counts", resolve_counts)
    monkeypatch.setattr(monitoring, "_query_gpu_processes", lambda: [])

    status = monitoring.collect_batch_status(output_root)

    assert status["current_run"] == {"id": "current"}
    assert observed["status_path"] == output_root / "batch_status.json"
    assert observed["manifest_path"] == output_root / "shard_manifest.json"
    assert observed["manifest_for_counts"] is manifest_identities


def test_matching_run_directories_uses_runs_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def iter_run_directories(runs_root: Path, **kwargs: object) -> list[Path]:
        observed["runs_root"] = runs_root
        observed["kwargs"] = kwargs
        return [runs_root / "run"]

    monkeypatch.setattr(monitoring, "_iter_run_directories", iter_run_directories)
    active_models = {"model"}
    identities: set[object] = {"identity"}

    result = monitoring._matching_run_directories(
        tmp_path,
        active_model_slugs=active_models,
        manifest_identities=identities,
    )

    assert result == [tmp_path / "runs" / "run"]
    assert observed == {
        "runs_root": tmp_path / "runs",
        "kwargs": {
            "active_model_slugs": active_models,
            "manifest_identities": identities,
        },
    }


def test_iter_run_directories_enumerates_only_rep_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    expected = [runs_root / "model" / "rep_01", runs_root / "rep_00"]
    for path in expected:
        path.mkdir(parents=True)
    (runs_root / "not_a_run").mkdir(parents=True)
    seen: list[Path] = []

    def matches(_runs_root: Path, path: Path, **_kwargs: object) -> bool:
        seen.append(path)
        return True

    monkeypatch.setattr(monitoring, "_run_dir_matches_filters", matches)

    result = monitoring._iter_run_directories(
        runs_root,
        active_model_slugs=set(),
        manifest_identities=set(),
    )

    assert set(result) == set(expected)
    assert set(seen) == set(expected)


def test_run_dir_matches_filters_rejects_unparseable_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(monitoring, "run_dir_identity", lambda **_kwargs: None)

    assert (
        monitoring._run_dir_matches_filters(
            tmp_path,
            tmp_path / "rep_00",
            active_model_slugs=set(),
            manifest_identities=set(),
        )
        is False
    )


def test_load_planned_total_runs_reads_preview_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    observed: list[Path] = []

    def read_json(path: Path) -> dict[str, object]:
        observed.append(path)
        return {"planned_total_runs": 17}

    monkeypatch.setattr(monitoring, "_read_json", read_json)

    assert monitoring._load_planned_total_runs(output_root) == 17
    assert observed == [output_root / "preview" / "resolved_run_plan.json"]


def test_collect_active_run_details_preserves_stage_and_worker_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    active_stage = {"schema_name": "edge_list", "status": "running"}
    worker_state = {
        "worker_pid": 123,
        "pid": 456,
        "status": "running",
        "model_loaded": True,
    }
    payloads = {
        run_dir / "active_stage.json": active_stage,
        run_dir / "worker_state.json": worker_state,
    }
    observed: list[Path] = []

    def read_json(path: Path) -> dict[str, object]:
        observed.append(path)
        return payloads[path]

    monkeypatch.setattr(monitoring, "_read_json", read_json)
    monkeypatch.setattr(
        monitoring,
        "_active_stage_age_seconds",
        lambda _run_dir, _worker_state: 12.345,
    )

    assert monitoring._collect_active_run_details(run_dir) == {
        "active_stage": active_stage,
        "active_stage_age_seconds": 12.345,
        "worker_pid": 123,
        "worker_status": "running",
        "worker_loaded_model": True,
    }
    assert observed == [run_dir / "active_stage.json", run_dir / "worker_state.json"]


def test_collect_active_run_details_falls_back_to_legacy_pid_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "active_stage.json", {"status": "running"})
    _write_json(
        run_dir / "worker_state.json",
        {"pid": 456, "status": "running", "model_loaded": False},
    )
    monkeypatch.setattr(
        monitoring,
        "_active_stage_age_seconds",
        lambda _run_dir, _worker_state: None,
    )

    details = monitoring._collect_active_run_details(run_dir)

    assert details["worker_pid"] == 456


def test_active_stage_age_seconds_uses_stage_then_worker_mtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(monitoring.time, "time", lambda: 1000.0)
    stage_run = tmp_path / "stage"
    stage_path = stage_run / "active_stage.json"
    _write_json(stage_path, {})
    os.utime(stage_path, (990.1234, 990.1234))

    assert (
        monitoring._active_stage_age_seconds(stage_run, {"status": "finished"})
        == 9.877
    )

    worker_run = tmp_path / "worker"
    worker_path = worker_run / "worker_state.json"
    _write_json(worker_path, {})
    os.utime(worker_path, (995.4321, 995.4321))

    assert (
        monitoring._active_stage_age_seconds(worker_run, {"status": "running"})
        == 4.568
    )
    assert (
        monitoring._active_stage_age_seconds(worker_run, {"status": "finished"})
        is None
    )


def test_active_stage_age_seconds_uses_canonical_artifact_paths() -> None:
    seen: list[str] = []

    class RecordingPath:
        def __truediv__(self, child: str) -> RecordingPath:
            seen.append(child)
            return self

        def exists(self) -> bool:
            return False

    assert monitoring._active_stage_age_seconds(RecordingPath(), {}) is None
    assert seen == ["active_stage.json", "worker_state.json"]


def test_query_gpu_processes_passes_strict_nvidia_smi_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="123, 456\n")

    monkeypatch.setattr(monitoring.subprocess, "run", run)

    assert monitoring._query_gpu_processes() == [
        {"pid": 123, "used_gpu_memory_mib": 456}
    ]
    assert calls == [
        (
            (
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
            ),
            {"check": True, "capture_output": True, "text": True},
        )
    ]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("123, 456", {"pid": 123, "used_gpu_memory_mib": 456}),
        ("123,", {"pid": 123, "used_gpu_memory_mib": None}),
        ("123, 456, 789", None),
    ],
)
def test_parse_gpu_process_line_validates_both_columns(
    line: str,
    expected: dict[str, object] | None,
) -> None:
    assert monitoring._parse_gpu_process_line(line) == expected


def test_status_timestamp_now_returns_utc_iso_timestamp() -> None:
    timestamp = monitoring.status_timestamp_now()
    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo == UTC


def test_write_status_snapshot_writes_canonical_json_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    status = {"finished_count": 2, "updated_at": "2026-08-22T10:00:00+00:00"}
    observed: dict[str, object] = {}

    def write_json(path: Path, payload: dict[str, object]) -> None:
        observed["path"] = path
        observed["payload"] = payload

    monkeypatch.setattr(monitoring, "write_json_dict", write_json)
    monitoring.write_status_snapshot(output_root=output_root, status=status)

    assert observed == {
        "path": output_root / "batch_status.json",
        "payload": status,
    }
