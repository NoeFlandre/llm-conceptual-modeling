from __future__ import annotations

from pathlib import Path

from llm_conceptual_modeling.hf_drain.common import (
    DrainWorkItem,
    FailureSummaryFn,
    JsonDict,
    _build_work_item,
    _expected_watcher_identity,
    _needs_risky_phase,
    _normalize_drain_phase,
    _resolve_ssh_target_and_port,
    _root_has_work,
    _should_adopt_active_root,
    _work_item_sort_key,
    _work_item_to_payload,
    read_results_sync_status,
)
from llm_conceptual_modeling.hf_drain.common import (
    summarize_results_root_failures as _summarize_results_root_failures,
)
from llm_conceptual_modeling.hf_resume.profile import (
    RISKY_PHASE,
    SAFE_PHASE,
    resolve_resume_profile,
)
from llm_conceptual_modeling.hf_resume.sweep import build_resume_sweep_report

summarize_results_root_failures = _summarize_results_root_failures


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


def _read_json_file(path: Path) -> JsonDict:
    import json

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
