from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llm_conceptual_modeling.common.failure_markers import classify_failure
from llm_conceptual_modeling.common.io import coerce_int, read_json_dict, write_json_dict
from llm_conceptual_modeling.hf_batch.spec_path import SpecIdentity
from llm_conceptual_modeling.hf_batch.utils import slugify_model

RETRYABLE_FAILURE_KINDS = {"timeout", "oom", "infrastructure", "structural"}


def refresh_ledger(
    *,
    results_root: str | Path,
    ledger_root: str | Path | None = None,
) -> dict[str, object]:
    results_root_path = Path(results_root).resolve()
    ledger_root_path = (
        Path(ledger_root).resolve() if ledger_root is not None else results_root_path
    )
    ledger_path = ledger_root_path / "ledger.json"
    existing_ledger = read_json_dict(ledger_path)
    records = _load_reference_records(
        existing_ledger=existing_ledger,
        results_root=results_root_path,
    )
    batch_roots = _discover_batch_roots(results_root_path)
    refreshed_records = [
        _refresh_record(
            identity=_record_identity(record),
            batch_roots=batch_roots,
            results_root=results_root_path,
            ledger_root=ledger_root_path,
        )
        for record in records
    ]

    counts = {
        "finished": 0,
        "pending": 0,
        "retryable_failed": 0,
        "terminal_failed": 0,
    }
    duplicate_logical_run_count = 0
    for record in refreshed_records:
        status = str(record["status"])
        if status not in counts:
            raise ValueError(f"Unexpected ledger status: {status}")
        counts[status] += 1
        candidate_count = len(cast(list[dict[str, object]], record["candidates"]))
        duplicate_logical_run_count += max(candidate_count - 1, 0)

    duplicate_extra_artifact_count = coerce_int(
        existing_ledger.get("duplicate_extra_artifact_count", 0)
    )
    expected_total_runs = len(refreshed_records) or coerce_int(
        existing_ledger.get("expected_total_runs", 0)
    )
    generated_at = datetime.now(UTC).isoformat()
    refreshed_ledger = {
        "duplicate_extra_artifact_count": duplicate_extra_artifact_count,
        "duplicate_logical_run_count": duplicate_logical_run_count,
        "expected_total_runs": expected_total_runs,
        "finished_count": counts["finished"],
        "generated_at": generated_at,
        "pending_count": counts["pending"],
        "records": refreshed_records,
        "retryable_failed_count": counts["retryable_failed"],
        "terminal_failed_count": counts["terminal_failed"],
    }
    write_json_dict(ledger_path, refreshed_ledger)
    return refreshed_ledger


def _load_reference_records(
    *,
    existing_ledger: dict[str, Any],
    results_root: Path,
) -> list[dict[str, object]]:
    records = existing_ledger.get("records")
    if isinstance(records, list) and records:
        normalized_records: list[dict[str, object]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            identity = record.get("identity")
            if not isinstance(identity, dict):
                continue
            normalized_records.append(
                {
                    "identity": _normalize_identity(identity),
                    "status": str(record.get("status", "pending")),
                }
            )
        if normalized_records:
            return normalized_records
    return [
        {
            "identity": identity,
            "status": "pending",
        }
        for identity in _discover_manifest_identities(results_root)
    ]


def _discover_manifest_identities(results_root: Path) -> list[dict[str, object]]:
    identities: dict[SpecIdentity, dict[str, object]] = {}
    for batch_root in _discover_batch_roots(results_root):
        manifest = read_json_dict(batch_root / "shard_manifest.json")
        manifest_identities = manifest.get("identities", [])
        if not isinstance(manifest_identities, list):
            continue
        for identity in manifest_identities:
            if not isinstance(identity, dict):
                continue
            normalized = _normalize_identity(identity)
            identities[_identity_key(normalized)] = normalized
    return [
        identities[key]
        for key in sorted(identities, key=lambda key: key)
    ]


def _discover_batch_roots(results_root: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in results_root.glob("hf-paper-batch-*")
        if candidate.is_dir()
    )


def _refresh_record(
    *,
    identity: dict[str, object],
    batch_roots: list[Path],
    results_root: Path,
    ledger_root: Path,
) -> dict[str, object]:
    candidates = [
        candidate
        for batch_root in batch_roots
        if (candidate := _load_candidate(batch_root, identity)) is not None
    ]
    finished_candidates = [
        candidate for candidate in candidates if candidate["status"] == "finished"
    ]
    if finished_candidates:
        winner = min(
            finished_candidates,
            key=lambda candidate: (
                _root_rank_for_run_dir(
                    Path(str(candidate["source_run_dir"])),
                    results_root=results_root,
                    ledger_root=ledger_root,
                ),
                -float(candidate["artifact_timestamp"]),
                str(candidate["source_run_dir"]),
            ),
        )
        return {
            "identity": identity,
            "status": "finished",
            "candidates": candidates,
            "canonical_unresolved": None,
            "winner": winner,
        }

    unresolved_candidates = [
        candidate for candidate in candidates if candidate["status"] != "finished"
    ]
    canonical_unresolved = (
        max(
            unresolved_candidates,
            key=lambda candidate: (
                float(candidate["artifact_timestamp"]),
                -_root_rank_for_run_dir(
                    Path(str(candidate["source_run_dir"])),
                    results_root=results_root,
                    ledger_root=ledger_root,
                ),
                str(candidate["source_run_dir"]),
            ),
        )
        if unresolved_candidates
        else None
    )
    if canonical_unresolved is None:
        return {
            "identity": identity,
            "status": "pending",
            "candidates": [],
            "canonical_unresolved": None,
            "winner": None,
        }

    failure_kind = str(canonical_unresolved.get("failure_kind", "pending"))
    if failure_kind == "pending":
        status = "pending"
    elif failure_kind in RETRYABLE_FAILURE_KINDS:
        status = "retryable_failed"
    else:
        status = "terminal_failed"
    return {
        "identity": identity,
        "status": status,
        "candidates": candidates,
        "canonical_unresolved": canonical_unresolved,
        "winner": None,
    }


def _load_candidate(batch_root: Path, identity: dict[str, object]) -> dict[str, object] | None:
    run_dir = _candidate_run_dir(batch_root, identity)
    if not run_dir.exists():
        return None
    summary = read_json_dict(run_dir / "summary.json")
    if summary.get("status") == "finished":
        return {
            "artifact_timestamp": _artifact_timestamp(
                run_dir,
                ["summary.json", "state.json", "raw_row.json", "manifest.json", "runtime.json"],
            ),
            "metrics": _candidate_metrics(summary),
            "source_run_dir": str(run_dir.resolve()),
            "status": "finished",
            "validated": True,
        }

    state = read_json_dict(run_dir / "state.json")
    status = str(state.get("status", "pending"))
    if status not in {"failed", "running", "pending"} and not summary:
        return None
    if status == "failed":
        error = read_json_dict(run_dir / "error.json")
        return {
            "artifact_timestamp": _artifact_timestamp(
                run_dir,
                ["error.json", "state.json", "active_stage.json", "worker_state.json"],
            ),
            "failure_kind": _classify_failure_payload(error),
            "source_run_dir": str(run_dir.resolve()),
            "status": "failed",
            "validated": False,
        }
    return {
        "artifact_timestamp": _artifact_timestamp(
            run_dir,
            ["state.json", "active_stage.json", "worker_state.json"],
        ),
        "failure_kind": "pending",
        "source_run_dir": str(run_dir.resolve()),
        "status": "pending",
        "validated": False,
    }


def _candidate_run_dir(batch_root: Path, identity: dict[str, object]) -> Path:
    graph_parts = []
    if str(identity.get("graph_source", "default")) != "default":
        graph_parts.append(str(identity["graph_source"]))
    return (
        batch_root
        / "runs"
        / str(identity["algorithm"])
        / slugify_model(str(identity["model"]))
        / str(identity["condition_label"])
        / Path(*graph_parts)
        / str(identity["pair_name"])
        / str(identity["condition_bits"])
        / f"rep_{int(identity['replication']):02d}"
    )


def _candidate_metrics(summary: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("accuracy", "precision", "recall", "f1"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _artifact_timestamp(run_dir: Path, filenames: list[str]) -> float:
    timestamps: list[float] = []
    for filename in filenames:
        path = run_dir / filename
        if path.exists():
            timestamps.append(path.stat().st_mtime)
    if timestamps:
        return max(timestamps)
    return run_dir.stat().st_mtime


def _classify_failure_payload(error: dict[str, object]) -> str:
    return classify_failure(
        error_type=str(error.get("type", "")),
        message=str(error.get("message", "")),
    )


def _identity_key(identity: dict[str, object]) -> SpecIdentity:
    graph_source = str(identity.get("graph_source", "default"))
    if graph_source != "default":
        return (
            str(identity["algorithm"]),
            str(identity["model"]),
            str(identity["condition_label"]),
            graph_source,
            str(identity["pair_name"]),
            str(identity["condition_bits"]),
            int(identity["replication"]),
        )
    return (
        str(identity["algorithm"]),
        str(identity["model"]),
        str(identity["condition_label"]),
        str(identity["pair_name"]),
        str(identity["condition_bits"]),
        int(identity["replication"]),
    )


def _normalize_identity(identity: dict[str, object]) -> dict[str, object]:
    normalized = {
        "algorithm": str(identity["algorithm"]),
        "condition_bits": str(identity["condition_bits"]),
        "condition_label": str(identity["condition_label"]),
        "model": str(identity["model"]),
        "pair_name": str(identity["pair_name"]),
        "replication": int(identity["replication"]),
    }
    graph_source = str(identity.get("graph_source", "default"))
    if graph_source != "default":
        normalized["graph_source"] = graph_source
    return normalized


def _record_identity(record: dict[str, object]) -> dict[str, object]:
    identity = record.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Ledger record is missing a valid identity")
    return _normalize_identity(identity)


def _root_rank_for_run_dir(run_dir: Path, *, results_root: Path, ledger_root: Path) -> int:
    root_name = _resolve_batch_root_name(run_dir, results_root)
    if root_name == ledger_root.name:
        return 0
    if root_name == "hf-paper-batch-canonical":
        return 1
    if "canonical-unified" in root_name:
        return 2
    if "canonical-unfinished" in root_name:
        return 3
    if "canonical-balanced" in root_name:
        return 4
    if "canonical-redistributed" in root_name:
        return 5
    if root_name.endswith("-current") or "-current" in root_name:
        return 6
    return 7


def _resolve_batch_root_name(run_dir: Path, results_root: Path) -> str:
    try:
        return run_dir.resolve().relative_to(results_root).parts[0]
    except Exception:
        return next((part for part in run_dir.parts if part.startswith("hf-paper-batch-")), "")
