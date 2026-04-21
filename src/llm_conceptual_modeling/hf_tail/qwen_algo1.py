from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml

from llm_conceptual_modeling.common.io import coerce_int
from llm_conceptual_modeling.hf_batch.planning import (
    default_runtime_profile_provider,
    plan_paper_batch,
)
from llm_conceptual_modeling.hf_batch.spec_path import (
    NormalizedSpecIdentityItem,
    normalize_spec_identity_item,
)
from llm_conceptual_modeling.hf_config.run_config import load_hf_run_config
from llm_conceptual_modeling.hf_resume.preflight import ResumePreflightReport
from llm_conceptual_modeling.hf_state.ledger import refresh_ledger

QWEN_ALGO1_TAIL_MODEL = "Qwen/Qwen3.5-9B"
QWEN_ALGO1_TAIL_ALGORITHM = "algo1"
QWEN_ALGO1_TAIL_CONDITION_LABEL = "contrastive_penalty_alpha_0.8"
QWEN_ALGO1_TAIL_PAIR_NAME = "sg1_sg2"
QWEN_ALGO1_TAIL_EXPECTED_COUNT = 10
QWEN_ALGO1_TAIL_EXPECTED_BITS = frozenset({"00101", "10100"})
QWEN_ALGO1_TAIL_EXPECTED_REPLICATIONS = frozenset(range(5))


class QwenAlgo1TailRecord(TypedDict):
    identity: NormalizedSpecIdentityItem
    status: str


class QwenAlgo1TailManifest(TypedDict):
    generated_at: str
    canonical_results_root: str
    results_root: str
    ledger_root: str
    active_chat_models: list[str]
    shard_count: int
    shard_index: int
    identities: list[NormalizedSpecIdentityItem]


class QwenAlgo1TailSeedLedger(TypedDict):
    duplicate_extra_artifact_count: int
    duplicate_logical_run_count: int
    expected_total_runs: int
    finished_count: int
    generated_at: str
    pending_count: int
    records: list[QwenAlgo1TailRecord]
    retryable_failed_count: int
    terminal_failed_count: int


class QwenAlgo1TailBundleReport(TypedDict):
    canonical_results_root: str
    tail_results_root: str
    remote_output_root: str
    manifest_path: str
    config_path: str
    ledger_path: str
    identity_count: int
    copied_run_dir_count: int
    copied_run_dirs: list[str]
    seed_status_counts: dict[str, int]


class QwenAlgo1TailLedgerSnapshot(TypedDict):
    finished_count: int
    pending_count: int
    retryable_failed_count: int
    terminal_failed_count: int
    expected_total_runs: int


class QwenAlgo1TailPreflightReport(TypedDict):
    repo_root: str
    canonical_results_root: str
    tail_results_root: str
    watcher_status: dict[str, object] | None
    canonical_ledger: QwenAlgo1TailLedgerSnapshot
    tail_ledger: QwenAlgo1TailLedgerSnapshot
    resume_preflight: ResumePreflightReport


def prepare_qwen_algo1_tail_bundle(
    *,
    canonical_results_root: str | Path,
    tail_results_root: str | Path,
    remote_output_root: str,
) -> QwenAlgo1TailBundleReport:
    canonical_results_root_path = Path(canonical_results_root).resolve()
    tail_results_root_path = Path(tail_results_root).resolve()
    if not tail_results_root_path.name.startswith("hf-paper-batch-"):
        raise ValueError(
            "tail_results_root basename must start with 'hf-paper-batch-' so ledger refresh can "
            "discover it as an isolated batch root."
        )
    tail_results_root_path.mkdir(parents=True, exist_ok=True)
    records = collect_qwen_algo1_tail_records(canonical_results_root_path)
    manifest = write_qwen_algo1_tail_manifest(
        canonical_results_root=canonical_results_root_path,
        tail_results_root=tail_results_root_path,
        records=records,
    )
    write_qwen_algo1_tail_config(
        canonical_results_root=canonical_results_root_path,
        tail_results_root=tail_results_root_path,
        remote_output_root=remote_output_root,
    )
    copied_run_dirs = seed_qwen_algo1_tail_results(
        canonical_results_root=canonical_results_root_path,
        tail_results_root=tail_results_root_path,
        records=records,
    )
    ledger = write_qwen_algo1_tail_seed_ledger(
        tail_results_root=tail_results_root_path,
        records=records,
    )
    return {
        "canonical_results_root": str(canonical_results_root_path),
        "tail_results_root": str(tail_results_root_path),
        "remote_output_root": remote_output_root,
        "manifest_path": str(tail_results_root_path / "shard_manifest.json"),
        "config_path": str(tail_results_root_path / "runtime_config.yaml"),
        "ledger_path": str(tail_results_root_path / "ledger.json"),
        "identity_count": len(manifest["identities"]),
        "copied_run_dir_count": len(copied_run_dirs),
        "copied_run_dirs": copied_run_dirs,
        "seed_status_counts": {
            "retryable_failed": sum(
                1 for record in ledger["records"] if record["status"] == "retryable_failed"
            ),
        },
    }


def build_qwen_algo1_tail_preflight_report(
    *,
    repo_root: str | Path,
    canonical_results_root: str | Path,
    tail_results_root: str | Path,
    watcher_status_path: str | Path | None = None,
) -> QwenAlgo1TailPreflightReport:
    repo_root_path = Path(repo_root).resolve()
    canonical_results_root_path = Path(canonical_results_root).resolve()
    tail_results_root_path = Path(tail_results_root).resolve()
    watcher_status = read_watcher_status(
        watcher_status_path
        if watcher_status_path is not None
        else tail_results_root_path / "results-sync-status.json"
    )
    if watcher_status.get("status") == "degraded":
        raise RuntimeError("Dedicated tail watcher is degraded; refusing fresh-host prep.")

    canonical_ledger = json.loads(
        (canonical_results_root_path / "ledger.json").read_text(encoding="utf-8")
    )
    if not isinstance(canonical_ledger.get("records"), list):
        raise RuntimeError("Canonical ledger.json is missing records; refusing fresh-host prep.")
    dedicated_ledger = refresh_ledger(
        results_root=tail_results_root_path.parent,
        ledger_root=tail_results_root_path,
    )
    manifest = _read_tail_manifest(tail_results_root_path / "shard_manifest.json")
    manifest_identities = manifest.get("identities", [])
    if len(manifest_identities) != QWEN_ALGO1_TAIL_EXPECTED_COUNT:
        raise RuntimeError("Dedicated tail manifest does not contain the expected 10 identities.")
    config = load_hf_run_config(tail_results_root_path / "runtime_config.yaml")
    planned_specs = plan_paper_batch(
        models=config.models.chat_models,
        embedding_model=config.models.embedding_model,
        replications=config.run.replications,
        config=config,
        runtime_profile_provider=default_runtime_profile_provider,
    )
    allowed_identities = {
        (
            str(identity["algorithm"]),
            str(identity["model"]),
            str(identity["condition_label"]),
            str(identity["pair_name"]),
            str(identity["condition_bits"]),
            int(identity["replication"]),
        )
        for identity in manifest_identities
    }
    filtered_specs = [
        spec
        for spec in planned_specs
        if (
            spec.algorithm,
            spec.model,
            spec.condition_label,
            spec.pair_name,
            spec.condition_bits,
            spec.replication,
        )
        in allowed_identities
    ]
    if len(filtered_specs) != QWEN_ALGO1_TAIL_EXPECTED_COUNT:
        raise RuntimeError(
            "Dedicated tail config planned an unexpected number of manifest-matched runs."
        )
    unfinished_count = (
        coerce_int(dedicated_ledger["pending_count"])
        + coerce_int(dedicated_ledger["retryable_failed_count"])
        + coerce_int(dedicated_ledger["terminal_failed_count"])
    )
    if unfinished_count != QWEN_ALGO1_TAIL_EXPECTED_COUNT:
        raise RuntimeError("Dedicated tail ledger does not resolve to exactly 10 unfinished runs.")
    resume_report: ResumePreflightReport = {
        "results_root": str(tail_results_root_path),
        "total_runs": len(filtered_specs),
        "finished_count": coerce_int(dedicated_ledger["finished_count"]),
        "failed_count": 0,
        "pending_count": unfinished_count,
        "running_count": 0,
        "can_resume": True,
        "resume_mode": "resume",
    }
    return {
        "repo_root": str(repo_root_path),
        "canonical_results_root": str(canonical_results_root_path),
        "tail_results_root": str(tail_results_root_path),
        "watcher_status": watcher_status or None,
        "canonical_ledger": _ledger_snapshot(canonical_ledger),
        "tail_ledger": _ledger_snapshot(dedicated_ledger),
        "resume_preflight": resume_report,
    }


def collect_qwen_algo1_tail_records(ledger_root: str | Path) -> list[QwenAlgo1TailRecord]:
    ledger_root_path = Path(ledger_root).resolve()
    ledger = _read_tail_seed_ledger(ledger_root_path / "ledger.json")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise ValueError("ledger.json does not contain a records list.")
    tail_records: list[QwenAlgo1TailRecord] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        identity = record["identity"]
        if str(identity.get("model")) != QWEN_ALGO1_TAIL_MODEL:
            continue
        if str(identity.get("algorithm")) != QWEN_ALGO1_TAIL_ALGORITHM:
            continue
        if str(identity.get("condition_label")) != QWEN_ALGO1_TAIL_CONDITION_LABEL:
            continue
        if str(record.get("status")) == "finished":
            continue
        if str(identity.get("pair_name")) != QWEN_ALGO1_TAIL_PAIR_NAME:
            raise ValueError("Qwen algo1 tail unexpectedly contains a non-sg1_sg2 pair.")
        if str(identity.get("condition_bits")) not in QWEN_ALGO1_TAIL_EXPECTED_BITS:
            raise ValueError("Qwen algo1 tail unexpectedly contains an unknown condition_bits.")
        if identity["replication"] not in QWEN_ALGO1_TAIL_EXPECTED_REPLICATIONS:
            raise ValueError("Qwen algo1 tail unexpectedly contains an unknown replication.")
        normalized_identity = normalize_spec_identity_item(identity)
        key = (
            normalized_identity["condition_bits"],
            normalized_identity["replication"],
        )
        if key in seen:
            continue
        seen.add(key)
        tail_records.append(
            {
                "identity": normalized_identity,
                "status": str(record.get("status", "pending")),
            }
        )
    if len(tail_records) != QWEN_ALGO1_TAIL_EXPECTED_COUNT:
        raise ValueError(
            "Expected exactly 10 unfinished Qwen algo1 tail records, "
            f"found {len(tail_records)}."
        )
    return sorted(
        tail_records,
        key=lambda record: (
            record["identity"]["condition_bits"],
            record["identity"]["replication"],
        ),
    )


def write_qwen_algo1_tail_manifest(
    *,
    canonical_results_root: str | Path,
    tail_results_root: str | Path,
    records: list[QwenAlgo1TailRecord],
) -> QwenAlgo1TailManifest:
    canonical_results_root_path = Path(canonical_results_root).resolve()
    tail_results_root_path = Path(tail_results_root).resolve()
    manifest: QwenAlgo1TailManifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "canonical_results_root": str(canonical_results_root_path),
        "results_root": str(tail_results_root_path),
        "ledger_root": str(tail_results_root_path),
        "active_chat_models": [QWEN_ALGO1_TAIL_MODEL],
        "shard_count": 1,
        "shard_index": 0,
        "identities": [record["identity"] for record in records],
    }
    (tail_results_root_path / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def write_qwen_algo1_tail_config(
    *,
    canonical_results_root: str | Path,
    tail_results_root: str | Path,
    remote_output_root: str,
) -> dict[str, object]:
    canonical_results_root_path = Path(canonical_results_root).resolve()
    tail_results_root_path = Path(tail_results_root).resolve()
    canonical_config = load_hf_run_config(
        canonical_results_root_path / "runtime_config.yaml",
    )
    tail_config = _build_tail_runtime_config(
        canonical_config=canonical_config,
        remote_output_root=remote_output_root,
    )
    config_path = tail_results_root_path / "runtime_config.yaml"
    config_path.write_text(
        yaml.safe_dump(tail_config, sort_keys=False),
        encoding="utf-8",
    )
    return tail_config


def seed_qwen_algo1_tail_results(
    *,
    canonical_results_root: str | Path,
    tail_results_root: str | Path,
    records: list[QwenAlgo1TailRecord],
) -> list[str]:
    canonical_results_root_path = Path(canonical_results_root).resolve()
    tail_results_root_path = Path(tail_results_root).resolve()
    copied_run_dirs: list[str] = []
    for record in records:
        identity = record["identity"]
        source_run_dir = _tail_run_dir(canonical_results_root_path, identity)
        destination_run_dir = _tail_run_dir(tail_results_root_path, identity)
        destination_run_dir.parent.mkdir(parents=True, exist_ok=True)
        if destination_run_dir.exists():
            shutil.rmtree(destination_run_dir)
        shutil.copytree(source_run_dir, destination_run_dir)
        copied_run_dirs.append(str(destination_run_dir))
    return copied_run_dirs


def write_qwen_algo1_tail_seed_ledger(
    *,
    tail_results_root: str | Path,
    records: list[QwenAlgo1TailRecord],
) -> QwenAlgo1TailSeedLedger:
    tail_results_root_path = Path(tail_results_root).resolve()
    ledger: QwenAlgo1TailSeedLedger = {
        "duplicate_extra_artifact_count": 0,
        "duplicate_logical_run_count": 0,
        "expected_total_runs": len(records),
        "finished_count": 0,
        "generated_at": datetime.now(UTC).isoformat(),
        "pending_count": len(records),
        "records": records,
        "retryable_failed_count": len(records),
        "terminal_failed_count": 0,
    }
    (tail_results_root_path / "ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ledger


def read_watcher_status(path: str | Path) -> dict[str, object]:
    path = Path(path).resolve()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_tail_runtime_config(
    *,
    canonical_config: Any,
    remote_output_root: str,
) -> dict[str, Any]:
    decoding = [
        item
        for item in canonical_config.decoding
        if item.algorithm == "contrastive" and item.penalty_alpha == 0.8
    ]
    if len(decoding) != 1:
        raise RuntimeError(
            "Canonical config does not contain the expected contrastive decoding entry."
        )
    tail_config = replace(
        canonical_config,
        run=replace(canonical_config.run, output_root=remote_output_root, replications=5),
        models=replace(
            canonical_config.models,
            chat_models=[QWEN_ALGO1_TAIL_MODEL],
        ),
        algorithms={
            "algo1": canonical_config.algorithms["algo1"],
        },
        decoding=tuple(decoding),
    )
    payload = tail_config.to_dict()
    payload["runtime"]["context_policy"]["worker_process_mode"] = "persistent"
    payload["runtime"]["context_policy"]["retry_structural_failures_on_resume"] = True
    payload["runtime"]["context_policy"]["max_requests_per_worker_process"] = 64
    payload["runtime"]["max_new_tokens_by_schema"]["edge_list"] = 512
    return payload


def _ledger_snapshot(payload: Mapping[str, object]) -> QwenAlgo1TailLedgerSnapshot:
    return {
        "finished_count": coerce_int(payload.get("finished_count", 0)),
        "pending_count": coerce_int(payload.get("pending_count", 0)),
        "retryable_failed_count": coerce_int(payload.get("retryable_failed_count", 0)),
        "terminal_failed_count": coerce_int(payload.get("terminal_failed_count", 0)),
        "expected_total_runs": coerce_int(payload.get("expected_total_runs", 0)),
    }


def _tail_run_dir(root: Path, identity: NormalizedSpecIdentityItem) -> Path:
    return (
        root
        / "runs"
        / identity["algorithm"]
        / identity["model"].replace("/", "__")
        / identity["condition_label"]
        / identity["pair_name"]
        / identity["condition_bits"]
        / f"rep_{identity['replication']:02d}"
    )


def _read_tail_manifest(path: Path) -> QwenAlgo1TailManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cast(QwenAlgo1TailManifest, payload)


def _read_tail_seed_ledger(path: Path) -> QwenAlgo1TailSeedLedger:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cast(QwenAlgo1TailSeedLedger, payload)
