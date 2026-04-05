from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.hf_batch_planning import default_runtime_profile_provider
from llm_conceptual_modeling.hf_experiments import (
    _current_run_payload,
    _run_dir_for_spec,
    _status_timestamp_now,
    _validate_structural_runtime_result,
    plan_paper_batch,
)
from llm_conceptual_modeling.hf_resume_state import (
    build_seeded_resume_snapshot,
    status_int,
)
from llm_conceptual_modeling.hf_run_config import HFRunConfig


def build_resume_preflight_report(
    *,
    config: HFRunConfig,
    repo_root: Path,
    results_root: Path | None,
    allow_empty: bool = False,
) -> dict[str, object]:
    repo_root_path = Path(repo_root)
    if not repo_root_path.exists():
        raise FileNotFoundError(f"Local repo root does not exist: {repo_root_path}")

    inputs_root = repo_root_path / "data" / "inputs"
    if not inputs_root.exists():
        raise FileNotFoundError(
            "Missing local input payloads at "
            f"{inputs_root}; do not rent a fresh host before fixing this."
        )

    effective_results_root = (
        Path(results_root) if results_root is not None else Path(config.run.output_root)
    )
    if results_root is not None and not effective_results_root.exists():
        raise FileNotFoundError(
            "Local results root does not exist: "
            f"{effective_results_root}. Refusing remote resume bootstrap."
        )

    planned_specs = plan_paper_batch(
        models=config.models.chat_models,
        embedding_model=config.models.embedding_model,
        replications=config.run.replications,
        config=config,
        runtime_profile_provider=default_runtime_profile_provider,
    )
    report: dict[str, object] = {
        "repo_root": str(repo_root_path),
        "inputs_root": str(inputs_root),
        "results_root": str(effective_results_root),
        "results_root_exists": effective_results_root.exists(),
        "total_runs": len(planned_specs),
    }

    if not effective_results_root.exists():
        report["finished_count"] = 0
        report["failed_count"] = 0
        report["pending_count"] = len(planned_specs)
        report["running_count"] = 0
        report["can_resume"] = True
        report["resume_mode"] = "fresh-root"
        return report

    batch_status_path = effective_results_root / "batch_status.json"
    current_status: dict[str, object] = {}
    if batch_status_path.exists():
        current_status = json.loads(batch_status_path.read_text(encoding="utf-8"))

    status_snapshot, _summary_rows, _seeded_finished, _seeded_failed = build_seeded_resume_snapshot(
        output_root=effective_results_root,
        planned_specs=planned_specs,
        started_at=_status_timestamp_now(),
        run_dir_for_spec_fn=lambda output_root, spec: _run_dir_for_spec(
            output_root=output_root,
            spec=spec,
        ),
        current_run_payload_fn=_current_run_payload,
        status_timestamp_now_fn=_status_timestamp_now,
        validate_structural_runtime_result_fn=_validate_structural_runtime_result,
    )
    finished_count = status_int(status_snapshot, "finished_count")
    failed_count = status_int(status_snapshot, "failed_count")
    pending_count = status_int(status_snapshot, "pending_count")
    running_count = _status_count(current_status, "running_count")
    report["finished_count"] = finished_count
    report["failed_count"] = failed_count
    report["pending_count"] = pending_count
    report["running_count"] = running_count
    report["status_updated_at"] = current_status.get("updated_at")
    report["can_resume"] = pending_count > 0 and running_count == 0
    report["resume_mode"] = "active" if running_count > 0 else "resume"

    if pending_count == 0 and not allow_empty:
        raise RuntimeError(
            f"No resumable work remains under {effective_results_root}; refusing remote bootstrap."
        )

    return report


def _status_count(status: dict[str, object], key: str) -> int:
    raw_value = status.get(key, 0)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0
