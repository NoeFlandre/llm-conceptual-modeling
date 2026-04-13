from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from llm_conceptual_modeling.common.io import coerce_int
from llm_conceptual_modeling.hf_failure_markers import classify_failure
from llm_conceptual_modeling.hf_resume_profile import RISKY_PHASE, SAFE_PHASE, ResumeProfile

JsonDict = dict[str, Any]
FailureSummaryFn = Callable[[Path], JsonDict]


@dataclass(frozen=True)
class DrainWorkItem:
    results_root: str
    config_source: str
    model_family: str
    algorithm_scope: str
    runtime_mode: str
    profile_name: str
    phase: str
    excluded_decoding_labels: tuple[str, ...]
    retry_timeout_failures_on_resume: bool
    retry_oom_failures_on_resume: bool
    retry_infrastructure_failures_on_resume: bool
    retry_structural_failures_on_resume: bool
    generation_timeout_seconds: int
    startup_timeout_seconds: int
    worker_process_mode: str
    max_requests_per_worker_process: int
    launch_priority: int
    pending_count: int
    failed_count: int
    finished_count: int
    running_count: int
    status_updated_at: str | None
    retryable_failure_counts: dict[str, int]
    terminal_failure_counts: dict[str, int]
    retryable_failure_total: int
    terminal_failure_total: int
    watcher_identity: str
    adopt_active_run: bool


def summarize_results_root_failures(results_root: Path) -> JsonDict:
    retryable_counts = {"timeout": 0, "oom": 0, "infrastructure": 0, "structural": 0}
    terminal_counts = {"unsupported": 0, "semantic": 0, "other": 0}
    runs_root = results_root / "runs"
    if not runs_root.exists():
        return _failure_summary_payload(retryable_counts, terminal_counts)
    for state_path in runs_root.rglob("state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("status") != "failed":
            continue
        error_payload = _read_json_file(state_path.with_name("error.json"))
        error_type = str(error_payload.get("type", "RuntimeError"))
        message = str(error_payload.get("message", ""))
        failure_kind = classify_failure(error_type=error_type, message=message)
        if failure_kind in retryable_counts:
            retryable_counts[failure_kind] += 1
            continue
        if failure_kind == "unsupported":
            terminal_counts["unsupported"] += 1
            continue
        if failure_kind == "other":
            terminal_counts["other"] += 1
            continue
        terminal_counts["semantic"] += 1
    return _failure_summary_payload(retryable_counts, terminal_counts)


def _resolve_ssh_target_and_port(
    *,
    ssh_command: str | None,
    ssh_target: str | None,
    ssh_port: str | None,
) -> tuple[str | None, str | None]:
    if ssh_target is not None and ssh_port is not None:
        return ssh_target, ssh_port
    if ssh_command is None:
        return ssh_target, ssh_port
    tokens = shlex.split(ssh_command)
    parsed_target: str | None = None
    parsed_port: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-p" and index + 1 < len(tokens):
            parsed_port = tokens[index + 1]
            index += 2
            continue
        if token == "ssh":
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        parsed_target = token
        break
    return ssh_target or parsed_target, ssh_port or parsed_port


def _expected_watcher_identity(
    *, ssh_target: str | None, ssh_port: str | None, results_root_name: str
) -> str:
    if ssh_target is None or ssh_port is None:
        return ""
    return f"{ssh_target}:{ssh_port}:/workspace/results/{results_root_name}"


def _infer_model_family(results_root_name: str) -> str:
    lowered = results_root_name.lower()
    if "olmo" in lowered:
        return "olmo"
    if "qwen" in lowered:
        return "qwen"
    if "mistral" in lowered:
        return "mistral"
    return "unknown"


def _infer_algorithm_scope(results_root_name: str) -> str:
    lowered = results_root_name.lower()
    if "algo1" in lowered:
        return "algo1"
    if "algo2" in lowered:
        return "algo2"
    if "algo3" in lowered:
        return "algo3"
    return "unknown"


def _work_item_sort_key(item: DrainWorkItem) -> tuple[int, int, int, int, str]:
    return (
        item.launch_priority,
        -item.pending_count,
        -item.retryable_failure_total,
        -item.failed_count,
        item.results_root,
    )


def _order_drain_queue(queue: list[DrainWorkItem]) -> list[DrainWorkItem]:
    return sorted(queue, key=_work_item_sort_key)


def _normalize_drain_phase(phase: str) -> str:
    normalized = phase.strip().lower()
    if normalized not in {SAFE_PHASE, RISKY_PHASE, "all"}:
        raise ValueError(f"Unsupported drain phase: {phase!r}")
    return normalized


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _timestamp_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: str) -> datetime:
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_json_file(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _failure_summary_payload(
    retryable_counts: dict[str, int],
    terminal_counts: dict[str, int],
) -> JsonDict:
    return {
        "retryable": dict(retryable_counts),
        "terminal": dict(terminal_counts),
        "retryable_total": int(sum(retryable_counts.values())),
        "terminal_total": int(sum(terminal_counts.values())),
    }


def _work_item_to_payload(item: DrainWorkItem) -> JsonDict:
    payload = asdict(item)
    payload["excluded_decoding_labels"] = list(item.excluded_decoding_labels)
    return payload


def read_results_sync_status(results_root: Path) -> JsonDict:
    return _read_json_file(results_root / "results-sync-status.json")


def _root_has_work(root_report: JsonDict, failure_summary: JsonDict) -> bool:
    if coerce_int(root_report.get("running_count", 0)) > 0:
        return True
    if coerce_int(root_report.get("pending_count", 0)) > 0:
        return True
    if coerce_int(root_report.get("failed_count", 0)) > 0:
        return True
    return failure_summary["retryable_total"] > 0 or failure_summary["terminal_total"] > 0


def _needs_risky_phase(
    root_report: JsonDict,
    safe_profile: ResumeProfile,
    *,
    full_coverage: bool,
) -> bool:
    if full_coverage:
        return False
    if safe_profile.excluded_decoding_labels:
        return True
    return coerce_int(root_report.get("failed_count", 0)) > 0


def _should_adopt_active_root(
    *,
    root_report: JsonDict,
    watcher_status: JsonDict,
    expected_identity: str,
) -> bool:
    if str(root_report.get("classification")) != "active":
        return False
    watcher_identity = str(watcher_status.get("watcher_identity", ""))
    watcher_state = str(watcher_status.get("status", ""))
    return watcher_identity == expected_identity and watcher_state in {
        "starting",
        "syncing",
        "healthy",
    }


def _can_continue_adopted_run(
    *,
    item: JsonDict,
    status: JsonDict,
    stale_after_seconds: int,
) -> bool:
    if not item.get("adopt_active_run"):
        return False
    if status["running_count"] == 0:
        return False
    return not _status_is_stale(status, stale_after_seconds)


def _status_is_stale(status: JsonDict, stale_after_seconds: int) -> bool:
    updated_at = status.get("updated_at")
    if not updated_at:
        return True
    try:
        parsed = _parse_timestamp(str(updated_at))
    except ValueError:
        return True
    return (datetime.now(UTC) - parsed).total_seconds() >= stale_after_seconds


def _build_work_item(
    *,
    root_report: JsonDict,
    profile: ResumeProfile,
    config_source: str,
    watcher_identity: str,
    failure_summary: JsonDict,
    adopt_active_run: bool,
) -> DrainWorkItem:
    results_root = Path(str(root_report["results_root"]))
    retryable_counts = dict(failure_summary["retryable"])
    terminal_counts = dict(failure_summary["terminal"])
    retryable_total = int(failure_summary.get("retryable_total", sum(retryable_counts.values())))
    terminal_total = int(failure_summary.get("terminal_total", sum(terminal_counts.values())))
    root_classification = str(root_report.get("classification", ""))
    return DrainWorkItem(
        results_root=str(results_root),
        config_source=config_source,
        model_family=_infer_model_family(results_root.name),
        algorithm_scope=_infer_algorithm_scope(results_root.name),
        runtime_mode=profile.runtime_mode,
        profile_name=profile.profile_name,
        phase=profile.phase,
        excluded_decoding_labels=profile.excluded_decoding_labels,
        retry_timeout_failures_on_resume=profile.retry_timeout_failures_on_resume,
        retry_oom_failures_on_resume=profile.retry_oom_failures_on_resume,
        retry_infrastructure_failures_on_resume=profile.retry_infrastructure_failures_on_resume,
        retry_structural_failures_on_resume=profile.retry_structural_failures_on_resume,
        generation_timeout_seconds=profile.generation_timeout_seconds,
        startup_timeout_seconds=profile.startup_timeout_seconds,
        worker_process_mode=profile.worker_process_mode,
        max_requests_per_worker_process=profile.max_requests_per_worker_process,
        launch_priority=0 if root_classification == "active" else 1,
        pending_count=coerce_int(root_report.get("pending_count", 0)),
        failed_count=coerce_int(root_report.get("failed_count", 0)),
        finished_count=coerce_int(root_report.get("finished_count", 0)),
        running_count=coerce_int(root_report.get("running_count", 0)),
        status_updated_at=_optional_str(root_report.get("status_updated_at")),
        retryable_failure_counts=retryable_counts,
        terminal_failure_counts=terminal_counts,
        retryable_failure_total=retryable_total,
        terminal_failure_total=terminal_total,
        watcher_identity=watcher_identity,
        adopt_active_run=adopt_active_run,
    )
