from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from llm_conceptual_modeling.common.coercion import coerce_int
from llm_conceptual_modeling.hf_failure_markers import classify_failure
from llm_conceptual_modeling.hf_resume_profile import (
    RISKY_PHASE,
    SAFE_PHASE,
    ResumeProfile,
    resolve_resume_profile,
)
from llm_conceptual_modeling.hf_resume_sweep import build_resume_sweep_report

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
    retryable_counts = Counter(
        {
            "timeout": 0,
            "oom": 0,
            "infrastructure": 0,
            "structural": 0,
        }
    )
    terminal_counts = Counter(
        {
            "unsupported": 0,
            "semantic": 0,
            "other": 0,
        }
    )

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


def build_drain_plan(
    *,
    repo_root: Path,
    results_root: Path,
    ssh_command: str | None = None,
    ssh_target: str | None = None,
    ssh_port: str | None = None,
    phase: str = "all",
    full_coverage: bool = False,
    root_name_contains: str | None = None,
    sweep_report: JsonDict | None = None,
    failure_summary_fn: FailureSummaryFn = summarize_results_root_failures,
    state_file: Path | None = None,
) -> JsonDict:
    selected_phase = _normalize_drain_phase(phase)
    resolved_target, resolved_port = _resolve_ssh_target_and_port(
        ssh_command=ssh_command,
        ssh_target=ssh_target,
        ssh_port=ssh_port,
    )

    if sweep_report is None:
        sweep_report = build_resume_sweep_report(
            repo_root=repo_root,
            results_root=results_root,
            root_name_contains=root_name_contains,
        )

    safe_queue: list[DrainWorkItem] = []
    risky_queue: list[DrainWorkItem] = []
    adopted_results_root: str | None = None

    for root_report in sweep_report.get("roots", []):
        if str(root_report.get("classification")) in {"invalid-config", "missing-config"}:
            continue
        results_root_path = Path(str(root_report["results_root"]))
        failure_summary = failure_summary_fn(results_root_path)
        if not _root_has_work(root_report, failure_summary):
            continue
        config_source = str(root_report.get("config_source") or "")
        if not config_source:
            continue

        watcher_status = read_results_sync_status(results_root_path)
        watcher_identity = str(
            watcher_status.get("watcher_identity")
            or _expected_watcher_identity(
                ssh_target=resolved_target,
                ssh_port=resolved_port,
                results_root_name=results_root_path.name,
            )
        )
        adopt_active_run = _should_adopt_active_root(
            root_report=root_report,
            watcher_status=watcher_status,
            expected_identity=_expected_watcher_identity(
                ssh_target=resolved_target,
                ssh_port=resolved_port,
                results_root_name=results_root_path.name,
            ),
        )
        if adopt_active_run:
            adopted_results_root = str(results_root_path)

        safe_profile = resolve_resume_profile(
            results_root_path.name,
            phase=SAFE_PHASE,
            full_coverage=full_coverage,
        )
        risky_profile = resolve_resume_profile(
            results_root_path.name,
            phase=RISKY_PHASE,
            full_coverage=full_coverage,
        )

        safe_queue.append(
            _build_work_item(
                root_report=root_report,
                profile=safe_profile,
                config_source=config_source,
                watcher_identity=watcher_identity,
                failure_summary=failure_summary,
                adopt_active_run=adopt_active_run,
            )
        )
        if _needs_risky_phase(root_report, safe_profile, full_coverage=full_coverage):
            risky_queue.append(
                _build_work_item(
                    root_report=root_report,
                    profile=risky_profile,
                    config_source=config_source,
                    watcher_identity=watcher_identity,
                    failure_summary=failure_summary,
                    adopt_active_run=False,
                )
            )

    ordered_safe_queue = sorted(safe_queue, key=_work_item_sort_key)
    ordered_risky_queue = sorted(risky_queue, key=_work_item_sort_key)
    queue = _select_queue_for_phase(
        safe_queue=ordered_safe_queue,
        risky_queue=ordered_risky_queue,
        phase=selected_phase,
    )
    queue_payload = [_work_item_to_payload(item) for item in queue]
    return {
        "repo_root": str(repo_root),
        "results_root": str(results_root),
        "ssh_target": resolved_target,
        "ssh_port": resolved_port,
        "phase": selected_phase,
        "full_coverage": full_coverage,
        "state_file": str(state_file) if state_file is not None else None,
        "adopted_results_root": adopted_results_root,
        "safe_queue_count": len(ordered_safe_queue),
        "risky_queue_count": len(ordered_risky_queue),
        "queue": queue_payload,
    }


def read_drain_state_report(state_file: Path) -> JsonDict:
    return _read_json_file(Path(state_file))


def run_drain_supervisor(
    *,
    repo_root: Path,
    results_root: Path,
    ssh_command: str,
    state_file: Path,
    phase: str = "all",
    full_coverage: bool = False,
    root_name_contains: str | None = None,
    poll_seconds: int = 30,
    stale_after_seconds: int = 3600,
    quick_resume_script: str | Path | None = None,
) -> JsonDict:
    plan = build_drain_plan(
        repo_root=repo_root,
        results_root=results_root,
        ssh_command=ssh_command,
        phase=phase,
        full_coverage=full_coverage,
        root_name_contains=root_name_contains,
        state_file=state_file,
    )
    queue = list(plan["queue"])
    state: JsonDict = {
        "state_file": str(state_file),
        "health": "starting",
        "current_phase": None,
        "current_results_root": None,
        "current_status": None,
        "queue": queue,
        "ssh_target": plan["ssh_target"],
        "ssh_port": plan["ssh_port"],
        "updated_at": _timestamp_now(),
    }
    write_drain_state_report(state_file=state_file, state=state)

    quick_resume_path = Path(quick_resume_script) if quick_resume_script is not None else (
        Path(repo_root) / "scripts" / "vast" / "quick_resume_from_ssh.sh"
    )
    for item in queue:
        current_root = Path(item["results_root"])
        current_status = collect_root_runtime_status(current_root)
        state["current_phase"] = item["phase"]
        state["current_results_root"] = item["results_root"]
        state["current_status"] = current_status
        state["health"] = "running"
        state["updated_at"] = _timestamp_now()
        write_drain_state_report(state_file=state_file, state=state)

        if not _can_continue_adopted_run(
            item=item,
            status=current_status,
            stale_after_seconds=stale_after_seconds,
        ):
            _launch_root_resume(
                ssh_command=ssh_command,
                quick_resume_script=quick_resume_path,
                item=item,
            )

        current_status = wait_for_root_phase_exit(
            results_root=current_root,
            phase=str(item["phase"]),
            poll_seconds=poll_seconds,
            stale_after_seconds=stale_after_seconds,
        )
        state["current_status"] = current_status
        state["updated_at"] = _timestamp_now()
        write_drain_state_report(state_file=state_file, state=state)

    state["health"] = "complete"
    state["updated_at"] = _timestamp_now()
    write_drain_state_report(state_file=state_file, state=state)
    return state


def write_drain_state_report(*, state_file: Path, state: JsonDict) -> None:
    state_path = Path(state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def collect_root_runtime_status(results_root: Path) -> JsonDict:
    batch_status = _read_json_file(results_root / "batch_status.json")
    watcher_status = read_results_sync_status(results_root)
    return {
        "finished_count": coerce_int(batch_status.get("finished_count", 0)),
        "pending_count": coerce_int(batch_status.get("pending_count", 0)),
        "failed_count": coerce_int(batch_status.get("failed_count", 0)),
        "running_count": coerce_int(batch_status.get("running_count", 0)),
        "updated_at": batch_status.get("updated_at"),
        "watcher_status": watcher_status.get("status"),
        "watcher_identity": watcher_status.get("watcher_identity"),
        "watcher_last_success_at": watcher_status.get("last_success_at"),
    }


def read_results_sync_status(results_root: Path) -> JsonDict:
    return _read_json_file(results_root / "results-sync-status.json")


def wait_for_root_phase_exit(
    *,
    results_root: Path,
    phase: str,
    poll_seconds: int,
    stale_after_seconds: int,
) -> JsonDict:
    while True:
        current_status = collect_root_runtime_status(results_root)
        if current_status["running_count"] == 0 and current_status["pending_count"] == 0:
            return current_status
        if _status_is_stale(current_status, stale_after_seconds):
            return current_status
        time.sleep(poll_seconds)


def _launch_root_resume(*, ssh_command: str, quick_resume_script: Path, item: JsonDict) -> None:
    env = os.environ.copy()
    env.update(
        {
            "BATCH_EXCLUDED_DECODING_LABELS": ",".join(item["excluded_decoding_labels"]),
            "BATCH_GENERATION_TIMEOUT_SECONDS": str(item["generation_timeout_seconds"]),
            "BATCH_STARTUP_TIMEOUT_SECONDS": str(item["startup_timeout_seconds"]),
            "BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME": str(
                item["retry_timeout_failures_on_resume"]
            ).lower(),
            "BATCH_RETRY_OOM_FAILURES_ON_RESUME": str(
                item["retry_oom_failures_on_resume"]
            ).lower(),
            "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME": str(
                item["retry_infrastructure_failures_on_resume"]
            ).lower(),
            "BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME": str(
                item["retry_structural_failures_on_resume"]
            ).lower(),
            "BATCH_WORKER_PROCESS_MODE": item["worker_process_mode"],
            "BATCH_MAX_REQUESTS_PER_WORKER_PROCESS": str(
                item["max_requests_per_worker_process"]
            ),
            "REMOTE_RUNTIME_MODE": item["runtime_mode"],
        }
    )
    remote_results_dir = f"/workspace/results/{Path(item['results_root']).name}"
    subprocess.run(
        [
            str(quick_resume_script),
            ssh_command,
            item["config_source"],
            remote_results_dir,
            item["results_root"],
        ],
        check=True,
        env=env,
    )


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


def _select_queue_for_phase(
    *,
    safe_queue: list[DrainWorkItem],
    risky_queue: list[DrainWorkItem],
    phase: str,
) -> list[DrainWorkItem]:
    if phase == SAFE_PHASE:
        return safe_queue
    if phase == RISKY_PHASE:
        return risky_queue
    return [*safe_queue, *risky_queue]


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


def _read_json_file(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _failure_summary_payload(
    retryable_counts: Counter[str],
    terminal_counts: Counter[str],
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
    *,
    ssh_target: str | None,
    ssh_port: str | None,
    results_root_name: str,
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
