from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_conceptual_modeling.common.io import read_json_dict, write_json_dict
from llm_conceptual_modeling.hf_state.active_models import resolve_active_chat_models


def write_unfinished_shard_manifest(
    *,
    results_root: str | Path,
    ledger_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    results_root_path = Path(results_root).resolve()
    ledger_root_path = (
        Path(ledger_root).resolve() if ledger_root is not None else results_root_path
    )
    manifest_output_path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else results_root_path / "shard_manifest.json"
    )
    ledger = read_json_dict(ledger_root_path / "ledger.json")
    active_chat_models = resolve_active_chat_models(results_root_path, ledger_root_path)
    identities = _unfinished_active_identities(
        ledger=ledger,
        active_chat_models=active_chat_models,
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "results_root": str(results_root_path),
        "ledger_root": str(ledger_root_path),
        "active_chat_models": sorted(active_chat_models),
        "shard_count": 1,
        "shard_index": 0,
        "identities": identities,
    }
    write_json_dict(manifest_output_path, manifest)
    return manifest


def _unfinished_active_identities(
    *,
    ledger: dict[str, Any],
    active_chat_models: set[str],
) -> list[dict[str, object]]:
    records = ledger.get("records", [])
    if not isinstance(records, list):
        return []
    identities: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = record.get("identity")
        if not isinstance(identity, dict):
            continue
        model = str(identity.get("model", ""))
        status = str(record.get("status", "pending"))
        if active_chat_models and model not in active_chat_models:
            continue
        if status == "finished":
            continue
        normalized = {
            "algorithm": str(identity["algorithm"]),
            "condition_bits": str(identity["condition_bits"]),
            "condition_label": str(identity["condition_label"]),
            "model": model,
            "pair_name": str(identity["pair_name"]),
            "replication": int(identity["replication"]),
        }
        key = (
            str(normalized["algorithm"]),
            str(normalized["model"]),
            str(normalized["condition_label"]),
            str(normalized["pair_name"]),
            str(normalized["condition_bits"]),
            int(normalized["replication"]),
        )
        if key in seen:
            continue
        seen.add(key)
        identities.append(normalized)
    return identities


def manifest_identity_keys(manifest: dict[str, Any]) -> set[tuple[str, str, str, str, str, int]]:
    identities = manifest.get("identities", [])
    if not isinstance(identities, list):
        return set()
    keys: set[tuple[str, str, str, str, str, int]] = set()
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        keys.add(
            (
                str(identity["algorithm"]),
                str(identity["model"]),
                str(identity["condition_label"]),
                str(identity["pair_name"]),
                str(identity["condition_bits"]),
                int(identity["replication"]),
            )
        )
    return keys
