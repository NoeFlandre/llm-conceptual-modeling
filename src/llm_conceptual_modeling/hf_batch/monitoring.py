from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def collect_batch_status(output_root: str | Path) -> dict[str, object]:
    output_root_path = Path(output_root)
    runs_root = output_root_path / "runs"
    run_dirs = sorted(_iter_run_directories(runs_root))

    finished_count = 0
    failed_count = 0
    running_count = 0
    pending_count = 0
    failures: list[dict[str, object]] = []

    for run_dir in run_dirs:
        state = _read_json(run_dir / "state.json")
        status = str(state.get("status", "pending"))
        if status == "finished":
            finished_count += 1
            continue
        if status == "failed":
            failed_count += 1
            error = _read_json(run_dir / "error.json")
            failures.append(
                {
                    "run_dir": str(run_dir),
                    "message": error.get("message"),
                    "type": error.get("type"),
                }
            )
            continue
        if status == "running":
            running_count += 1
            continue
        pending_count += 1

    total_runs = len(run_dirs)
    if total_runs == 0:
        status_file = _read_json(output_root_path / "batch_status.json")
        total_runs = int(status_file.get("total_runs", 0))

    status_file = _read_json(output_root_path / "batch_status.json")
    pending_count = max(total_runs - finished_count - failed_count - running_count, pending_count)
    percent_complete = round((finished_count / total_runs) * 100.0, 2) if total_runs else 0.0

    return {
        "total_runs": total_runs,
        "finished_count": finished_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "pending_count": pending_count,
        "failure_count": len(failures),
        "failures": failures,
        "percent_complete": percent_complete,
        "current_run": status_file.get("current_run"),
        "last_completed_run": status_file.get("last_completed_run"),
        "started_at": status_file.get("started_at"),
        "updated_at": status_file.get("updated_at"),
    }


def current_run_payload(
    *,
    algorithm: str,
    model: str,
    decoding_algorithm: str,
    pair_name: str,
    condition_bits: str,
    replication: int,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "model": model,
        "decoding_algorithm": decoding_algorithm,
        "pair_name": pair_name,
        "condition_bits": condition_bits,
        "replication": replication,
    }


def _iter_run_directories(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return [
        path
        for path in runs_root.rglob("rep_*")
        if path.is_dir()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status_timestamp_now() -> str:
    return datetime.now(UTC).isoformat()


def write_status_snapshot(*, output_root: str | Path, status: dict[str, object]) -> None:
    output_root_path = Path(output_root)
    status_path = output_root_path / "batch_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
