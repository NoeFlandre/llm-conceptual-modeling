from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from llm_conceptual_modeling.hf_batch.planning import default_runtime_profile_provider
from llm_conceptual_modeling.hf_ledger import refresh_ledger
from llm_conceptual_modeling.hf_experiments import plan_paper_batch
from llm_conceptual_modeling.hf_run_config import HFRunConfig, load_hf_run_config

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
    config = write_qwen_algo1_tail_config(
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
    manifest = json.loads((tail_results_root_path / "shard_manifest.json").read_text(encoding="utf-8"))
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
        raise RuntimeError("Dedicated tail config planned an unexpected number of manifest-matched runs.")
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
    canonical_config = load_hf_run_config(canonical_results_root_path / "runtime_config.yaml")
    algo1_config = canonical_config.algorithms[QWEN_ALGO1_TAIL_ALGORITHM]
    restricted_config = replace(
        canonical_config,
        run=replace(canonical_config.run, output_root=remote_output_root),
        models=replace(
            canonical_config.models,
            chat_models=[QWEN_ALGO1_TAIL_MODEL],
        ),
        decoding=[
            decoding
            for decoding in canonical_config.decoding
            if decoding.algorithm == "contrastive" and decoding.penalty_alpha == 0.8
        ],
        runtime=replace(
            canonical_config.runtime,
            context_policy=_restricted_context_policy(canonical_config),
            max_new_tokens_by_schema=_restricted_max_new_tokens(canonical_config),
            thinking_mode_by_model={QWEN_ALGO1_TAIL_MODEL: "disabled"},
        ),
        algorithms={
            QWEN_ALGO1_TAIL_ALGORITHM: replace(
                algo1_config,
                pair_names=[QWEN_ALGO1_TAIL_PAIR_NAME],
            )
        },
    )
    payload = restricted_config.to_dict()
    (tail_results_root_path / "runtime_config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return payload


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
        source_run_dir = run_dir_for_identity(canonical_results_root_path, identity)
        target_run_dir = run_dir_for_identity(tail_results_root_path, identity)
        if target_run_dir.exists():
            shutil.rmtree(target_run_dir)
        target_run_dir.parent.mkdir(parents=True, exist_ok=True)
        if source_run_dir.exists():
            shutil.copytree(source_run_dir, target_run_dir)
            copied_run_dirs.append(str(target_run_dir))
    return copied_run_dirs


def write_qwen_algo1_tail_seed_ledger(
    *,
    tail_results_root: str | Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    tail_results_root_path = Path(tail_results_root).resolve()
    statuses = [str(record["status"]) for record in records]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "expected_total_runs": len(records),
        "finished_count": sum(1 for status in statuses if status == "finished"),
        "pending_count": sum(1 for status in statuses if status == "pending"),
        "retryable_failed_count": sum(1 for status in statuses if status == "retryable_failed"),
        "terminal_failed_count": sum(1 for status in statuses if status == "terminal_failed"),
        "records": records,
    }
    (tail_results_root_path / "ledger.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def read_watcher_status(path: str | Path) -> dict[str, object]:
    status_path = Path(path)
    if not status_path.exists():
        return {}
    return json.loads(status_path.read_text(encoding="utf-8"))


def run_dir_for_identity(results_root: str | Path, identity: dict[str, object]) -> Path:
    results_root_path = Path(results_root).resolve()
    return (
        results_root_path
        / "runs"
        / str(identity["algorithm"])
        / str(identity["model"]).replace("/", "__")
        / str(identity["condition_label"])
        / str(identity["pair_name"])
        / str(identity["condition_bits"])
        / f"rep_{int(identity['replication']):02d}"
    )


def _restricted_context_policy(config: HFRunConfig) -> dict[str, object]:
    context_policy = dict(config.runtime.context_policy)
    context_policy["retry_structural_failures_on_resume"] = True
    context_policy["worker_process_mode"] = "persistent"
    context_policy["max_requests_per_worker_process"] = max(
        int(context_policy.get("max_requests_per_worker_process", 0) or 0),
        64,
    )
    return context_policy


def _restricted_max_new_tokens(config: HFRunConfig) -> dict[str, int]:
    payload = dict(config.runtime.max_new_tokens_by_schema)
    payload["edge_list"] = max(int(payload.get("edge_list", 0) or 0), 512)
    return payload


def _ledger_snapshot(ledger: dict[str, object]) -> dict[str, object]:
    return {
        "generated_at": ledger.get("generated_at"),
        "expected_total_runs": ledger.get("expected_total_runs"),
        "finished_count": ledger.get("finished_count"),
        "pending_count": ledger.get("pending_count"),
        "retryable_failed_count": ledger.get("retryable_failed_count"),
        "terminal_failed_count": ledger.get("terminal_failed_count"),
    }
