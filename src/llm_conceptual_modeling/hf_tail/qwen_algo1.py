from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from llm_conceptual_modeling.hf_batch.planning import (
    default_runtime_profile_provider,
    plan_paper_batch,
)
from llm_conceptual_modeling.hf_run_config import HFRunConfig, load_hf_run_config
from llm_conceptual_modeling.hf_state.ledger import refresh_ledger

QWEN_ALGO1_TAIL_MODEL = "Qwen/Qwen3.5-9B"
QWEN_ALGO1_TAIL_ALGORITHM = "algo1"
QWEN_ALGO1_TAIL_CONDITION_LABEL = "contrastive_penalty_alpha_0.8"
QWEN_ALGO1_TAIL_PAIR_NAME = "sg1_sg2"
QWEN_ALGO1_TAIL_EXPECTED_COUNT = 10
QWEN_ALGO1_TAIL_EXPECTED_BITS = frozenset({"00101", "10100"})
QWEN_ALGO1_TAIL_EXPECTED_REPLICATIONS = frozenset(range(5))


def prepare_qwen_algo1_tail_bundle(
    *,
    canonical_results_root: str | Path,
    tail_results_root: str | Path,
    remote_output_root: str,
) -> dict[str, object]:
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
) -> dict[str, object]:
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
    manifest = json.loads(
        (tail_results_root_path / "shard_manifest.json").read_text(encoding="utf-8")
    )
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
    unfinished_count = int(dedicated_ledger["pending_count"]) + int(
        dedicated_ledger["retryable_failed_count"]
    ) + int(dedicated_ledger["terminal_failed_count"])
    if unfinished_count != QWEN_ALGO1_TAIL_EXPECTED_COUNT:
        raise RuntimeError("Dedicated tail ledger does not resolve to exactly 10 unfinished runs.")
    resume_report = {
        "results_root": str(tail_results_root_path),
        "total_runs": len(filtered_specs),
        "finished_count": int(dedicated_ledger["finished_count"]),
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


def collect_qwen_algo1_tail_records(ledger_root: str | Path) -> list[dict[str, object]]:
    ledger_root_path = Path(ledger_root).resolve()
    ledger = json.loads((ledger_root_path / "ledger.json").read_text(encoding="utf-8"))
    records = ledger.get("records")
    if not isinstance(records, list):
        raise ValueError("ledger.json does not contain a records list.")
    tail_records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = record.get("identity")
        if not isinstance(identity, dict):
            continue
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
        if int(identity.get("replication")) not in QWEN_ALGO1_TAIL_EXPECTED_REPLICATIONS:
            raise ValueError("Qwen algo1 tail unexpectedly contains an unknown replication.")
        normalized_identity = {
            "algorithm": QWEN_ALGO1_TAIL_ALGORITHM,
            "condition_bits": str(identity["condition_bits"]),
            "condition_label": QWEN_ALGO1_TAIL_CONDITION_LABEL,
            "model": QWEN_ALGO1_TAIL_MODEL,
            "pair_name": QWEN_ALGO1_TAIL_PAIR_NAME,
            "replication": int(identity["replication"]),
        }
        key = (
            normalized_identity["condition_bits"],
            normalized_identity["condition_label"],
            normalized_identity["pair_name"],
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
            str(record["identity"]["condition_bits"]),
            int(record["identity"]["replication"]),
        ),
    )


def write_qwen_algo1_tail_manifest(
    *,
    canonical_results_root: str | Path,
    tail_results_root: str | Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    canonical_results_root_path = Path(canonical_results_root).resolve()
    tail_results_root_path = Path(tail_results_root).resolve()
    manifest = {
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
    records: list[dict[str, object]],
) -> list[str]:
    canonical_results_root_path = Path(canonical_results_root).resolve()
    tail_results_root_path = Path(tail_results_root).resolve()
    copied_run_dirs: list[str] = []
    for record in records:
        identity = record["identity"]
        source_run_dir = (
            canonical_results_root_path
            / "runs"
            / str(identity["algorithm"])
            / str(identity["model"]).replace("/", "__")
            / str(identity["condition_label"])
            / str(identity["pair_name"])
            / str(identity["condition_bits"])
            / f"rep_{int(identity['replication']):02d}"
        )
        destination_run_dir = (
            tail_results_root_path
            / "runs"
            / str(identity["algorithm"])
            / str(identity["model"]).replace("/", "__")
            / str(identity["condition_label"])
            / str(identity["pair_name"])
            / str(identity["condition_bits"])
            / f"rep_{int(identity['replication']):02d}"
        )
        destination_run_dir.parent.mkdir(parents=True, exist_ok=True)
        if destination_run_dir.exists():
            shutil.rmtree(destination_run_dir)
        shutil.copytree(source_run_dir, destination_run_dir)
        copied_run_dirs.append(str(destination_run_dir))
    return copied_run_dirs


def write_qwen_algo1_tail_seed_ledger(
    *,
    tail_results_root: str | Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    tail_results_root_path = Path(tail_results_root).resolve()
    ledger = {
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
    canonical_config: HFRunConfig,
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


def _ledger_snapshot(payload: dict[str, object]) -> dict[str, object]:
    return {
        "finished_count": int(payload.get("finished_count", 0)),
        "pending_count": int(payload.get("pending_count", 0)),
        "retryable_failed_count": int(payload.get("retryable_failed_count", 0)),
        "terminal_failed_count": int(payload.get("terminal_failed_count", 0)),
        "expected_total_runs": int(payload.get("expected_total_runs", 0)),
    }
