from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from llm_conceptual_modeling.common.io import coerce_int
from llm_conceptual_modeling.hf_drain.common import (
    JsonDict,
    _can_continue_adopted_run,
    _read_json_file,
    _status_is_stale,
)
from llm_conceptual_modeling.hf_drain.planning import build_drain_plan


def read_drain_state_report(state_file: Path) -> JsonDict:
    return _read_json_file(Path(state_file))


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

    quick_resume_path = (
        Path(quick_resume_script)
        if quick_resume_script is not None
        else (Path(repo_root) / "scripts" / "vast" / "quick_resume_from_ssh.sh")
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
            "BATCH_RETRY_OOM_FAILURES_ON_RESUME": str(item["retry_oom_failures_on_resume"]).lower(),
            "BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME": str(
                item["retry_infrastructure_failures_on_resume"]
            ).lower(),
            "BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME": str(
                item["retry_structural_failures_on_resume"]
            ).lower(),
            "BATCH_WORKER_PROCESS_MODE": item["worker_process_mode"],
            "BATCH_MAX_REQUESTS_PER_WORKER_PROCESS": str(item["max_requests_per_worker_process"]),
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


def _timestamp_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
