from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llm_conceptual_modeling.common.io import read_json_dict, write_json_dict
from llm_conceptual_modeling.hf_batch.spec_path import (
    NormalizedSpecIdentityItem,
    SpecIdentity,
    build_spec_identity,
    normalize_spec_identity_item,
)
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
    return cast(dict[str, object], manifest)


def _unfinished_active_identities(
    *,
    ledger: dict[str, Any],
    active_chat_models: set[str],
) -> list[NormalizedSpecIdentityItem]:
    records = ledger.get("records", [])
    if not isinstance(records, list):
        return []
    identities: list[NormalizedSpecIdentityItem] = []
    seen: set[SpecIdentity] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = record.get("identity")
        if not isinstance(identity, dict):
            continue
        normalized = normalize_spec_identity_item(identity)
        model = normalized["model"]
        status = str(record.get("status", "pending"))
        if active_chat_models and model not in active_chat_models:
            continue
        if status == "finished":
            continue
        key = build_spec_identity(
            algorithm=normalized["algorithm"],
            model=normalized["model"],
            condition_label=normalized["condition_label"],
            graph_source=normalized.get("graph_source", "default"),
            pair_name=normalized["pair_name"],
            condition_bits=normalized["condition_bits"],
            replication=normalized["replication"],
        )
        if key in seen:
            continue
        seen.add(key)
        identities.append(normalized)
    return identities


def manifest_identity_keys(manifest: dict[str, Any]) -> set[SpecIdentity]:
    identities = manifest.get("identities", [])
    if not isinstance(identities, list):
        return set()
    keys: set[SpecIdentity] = set()
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        normalized = normalize_spec_identity_item(identity)
        keys.add(
            build_spec_identity(
                algorithm=normalized["algorithm"],
                model=normalized["model"],
                condition_label=normalized["condition_label"],
                graph_source=normalized.get("graph_source", "default"),
                pair_name=normalized["pair_name"],
                condition_bits=normalized["condition_bits"],
                replication=normalized["replication"],
            )
        )
    return keys
