from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from llm_conceptual_modeling.common.io import read_json_dict, write_json_dict
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import manifest_for_spec
from llm_conceptual_modeling.hf_pipeline.metrics import summary_from_raw_row

VOLATILE_RUN_FILENAMES = (
    "active_stage.json",
    "error.json",
    "raw_response.json",
    "runtime.json",
    "worker_result.json",
    "worker_spec.json",
    "worker_state.json",
)


def clear_retry_artifacts(run_dir: Path) -> None:
    for name in VOLATILE_RUN_FILENAMES:
        artifact_path = run_dir / name
        if artifact_path.exists():
            artifact_path.unlink()


def normalize_stale_running_run(run_dir: Path) -> dict[str, object] | None:
    state = read_json(run_dir / "state.json")
    if state.get("status") != "running":
        return None
    worker_result_path = run_dir / "worker_result.json"
    if worker_result_path.exists():
        return None
    worker_state = read_json(run_dir / "worker_state.json")
    worker_pid = worker_state.get("worker_pid") or worker_state.get("pid")
    if _is_live_worker_pid(worker_pid):
        return None

    active_stage = read_json(run_dir / "active_stage.json")
    error_payload = {
        "type": "StaleRunState",
        "message": (
            "Found stale running state with no live worker or worker result artifact. "
            "Marking the run as failed so resume can retry safely."
        ),
        "status": "failed",
        "worker_pid": worker_pid,
        "active_stage": active_stage or None,
        "updated_at": _timestamp_now(),
    }
    write_json(run_dir / "error.json", error_payload)
    write_json(run_dir / "state.json", {**state, "status": "failed"})
    return error_payload


def write_smoke_verdict(
    *,
    output_root: Path,
    run_dir: Path,
    spec_identity: dict[str, object],
    status: str,
    failure_type: str | None = None,
    failure_message: str | None = None,
    worker_loaded_model: bool = False,
) -> dict[str, object]:
    verdict: dict[str, object] = {
        "status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "worker_loaded_model": worker_loaded_model,
        "runtime_snapshot_path": _runtime_snapshot_reference(),
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "error_path": str(run_dir / "error.json"),
        "spec": spec_identity,
        "updated_at": _timestamp_now(),
    }
    write_json(output_root / "smoke_verdict.json", verdict)
    return verdict


def read_json(path: Path) -> dict[str, Any]:
    return read_json_dict(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_dict(path, payload)


def _runtime_snapshot_reference() -> str | None:
    snapshot_path = Path.cwd() / ".bootstrap-runtime.json"
    if not snapshot_path.exists():
        return None
    return str(snapshot_path)


def write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _timestamp_now() -> str:
    return datetime.now(UTC).isoformat()


def build_run_summary(
    *,
    spec: HFRunSpec,
    raw_row: dict[str, object],
    runtime_result: RuntimeResult,
    raw_row_path: Path,
) -> dict[str, object]:
    """Build the run summary dict for a finished spec.

    Pure transformation: no I/O, no state mutation.
    """
    return {
        "algorithm": spec.algorithm,
        "model": spec.model,
        "embedding_model": spec.embedding_model,
        "decoding_algorithm": spec.decoding.algorithm,
        "condition_label": spec.condition_label,
        "pair_name": spec.pair_name,
        "condition_bits": spec.condition_bits,
        "replication": spec.replication,
        "status": "finished",
        "thinking_mode_supported": runtime_result["runtime"].get(
            "thinking_mode_supported", False
        ),
        "raw_row_path": str(raw_row_path),
        **summary_from_raw_row(spec.algorithm, raw_row),
        **runtime_result.get("summary", {}),
    }


def write_run_artifacts(
    *,
    run_dir: Path,
    spec: HFRunSpec,
    runtime_result: RuntimeResult,
    raw_row: dict[str, object],
    raw_row_path: Path,
    write_json_fn: Callable[[Path, dict[str, object]], None] | None = None,
    write_text_fn: Callable[[Path, str], None] | None = None,
    manifest_for_spec_fn: Callable[[HFRunSpec], dict[str, object]] | None = None,
) -> None:
    """Write all run artifact files for a finished spec.

    Writes: manifest.json, state.json, raw_row_path, runtime.json, raw_response.json.
    Does NOT write summary.json (caller does that) or smoke_verdict.json
    (requires output_root and spec_identity which differ per caller).
    """
    _wj = write_json_fn if write_json_fn is not None else write_json
    _wt = write_text_fn if write_text_fn is not None else write_text
    _ms = manifest_for_spec_fn if manifest_for_spec_fn is not None else manifest_for_spec
    _wj(run_dir / "manifest.json", _ms(spec))
    _wj(run_dir / "state.json", {"status": "finished"})
    _wj(raw_row_path, raw_row)
    _wj(run_dir / "runtime.json", runtime_result["runtime"])
    _wt(run_dir / "raw_response.json", runtime_result["raw_response"])


def _is_live_worker_pid(raw_pid: object) -> bool:
    if not isinstance(raw_pid, int) or raw_pid <= 0:
        return False
    try:
        os.kill(raw_pid, 0)
    except OSError:
        return False
    return True
