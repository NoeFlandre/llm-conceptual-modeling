"""Spec-identity and run-directory path resolution helpers.

Pure functions that project HFRunSpec data into identity keys and
compute deterministic run-directory paths.  No runtime logic, no
subprocess calls, no state mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, TypedDict

from llm_conceptual_modeling.hf_batch.types import HFRunSpec
from llm_conceptual_modeling.hf_batch.utils import slugify_model

SpecIdentity = tuple[object, ...]


class ShardManifestIdentityItem(TypedDict, total=False):
    algorithm: str
    model: str
    condition_label: str
    graph_source: str
    pair_name: str
    condition_bits: str
    replication: int | str


class NormalizedSpecIdentityItem(TypedDict, total=False):
    algorithm: str
    model: str
    condition_label: str
    graph_source: str
    pair_name: str
    condition_bits: str
    replication: int


def build_spec_identity(
    *,
    algorithm: str,
    model: str,
    condition_label: str,
    graph_source: str = "default",
    pair_name: str,
    condition_bits: str,
    replication: int,
) -> SpecIdentity:
    if graph_source != "default":
        return (
            algorithm,
            model,
            condition_label,
            graph_source,
            pair_name,
            condition_bits,
            replication,
        )
    return (
        algorithm,
        model,
        condition_label,
        pair_name,
        condition_bits,
        replication,
    )


def spec_identity(spec: HFRunSpec) -> SpecIdentity:
    """Return a deterministic tuple key for a run spec."""
    return build_spec_identity(
        algorithm=spec.algorithm,
        model=spec.model,
        condition_label=spec.condition_label,
        graph_source=spec.graph_source,
        pair_name=spec.pair_name,
        condition_bits=spec.condition_bits,
        replication=spec.replication,
    )


def normalize_spec_identity_item(
    identity: Mapping[str, object],
) -> NormalizedSpecIdentityItem:
    normalized: NormalizedSpecIdentityItem = {
        "algorithm": str(identity["algorithm"]),
        "condition_bits": str(identity["condition_bits"]),
        "condition_label": str(identity["condition_label"]),
        "model": str(identity["model"]),
        "pair_name": str(identity["pair_name"]),
        "replication": _coerce_identity_replication(identity["replication"]),
    }
    graph_source = str(identity.get("graph_source", "default"))
    if graph_source != "default":
        normalized["graph_source"] = graph_source
    return normalized


def smoke_spec_identity(spec: HFRunSpec) -> dict[str, object]:
    """Return a dict key suitable for smoke-test verdict files."""
    return {
        "algorithm": spec.algorithm,
        "model": spec.model,
        "embedding_model": spec.embedding_model,
        "graph_source": spec.graph_source,
        "decoding_algorithm": spec.decoding.algorithm,
        "pair_name": spec.pair_name,
        "condition_bits": spec.condition_bits,
        "replication": spec.replication,
    }


def run_dir_for_spec(*, output_root: Path, spec: HFRunSpec) -> Path:
    """Compute the deterministic run directory for a spec."""
    graph_parts = [] if spec.graph_source == "default" else [spec.graph_source]
    return (
        output_root
        / "runs"
        / spec.algorithm
        / slugify_model(spec.model)
        / spec.condition_label
        / Path(*graph_parts)
        / spec.pair_name
        / spec.condition_bits
        / f"rep_{spec.replication:02d}"
    )


def run_dir_identity(
    *,
    runs_root: Path,
    run_dir: Path,
) -> tuple[str, SpecIdentity] | None:
    """Return the model slug and manifest identity for a run directory."""
    try:
        relative_parts = run_dir.resolve().relative_to(runs_root.resolve()).parts
    except ValueError:
        return None
    if len(relative_parts) not in {6, 7}:
        return None
    if len(relative_parts) == 6:
        graph_source = "default"
        algorithm, model_slug, condition_label, pair_name, condition_bits, replication_part = (
            relative_parts
        )
    else:
        (
            algorithm,
            model_slug,
            condition_label,
            graph_source,
            pair_name,
            condition_bits,
            replication_part,
        ) = relative_parts
    if not replication_part.startswith("rep_"):
        return None
    replication_text = replication_part.removeprefix("rep_")
    if not replication_text.isdigit():
        return None
    model = model_slug.replace("__", "/")
    return (
        model_slug,
        build_spec_identity(
            algorithm=algorithm,
            model=model,
            condition_label=condition_label,
            graph_source=graph_source,
            pair_name=pair_name,
            condition_bits=condition_bits,
            replication=int(replication_text),
        ),
    )


def filter_planned_specs_for_output_root(
    *,
    planned_specs: list[HFRunSpec],
    output_root: Path,
) -> list[HFRunSpec]:
    """Filter planned specs to only those listed in the shard manifest."""
    manifest_path = output_root / "shard_manifest.json"
    if not manifest_path.exists():
        return planned_specs
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_identities = {
        _identity_from_manifest_item(item)
        for item in manifest.get("identities", [])
    }
    return [spec for spec in planned_specs if spec_identity(spec) in allowed_identities]


def _identity_from_manifest_item(item: ShardManifestIdentityItem) -> SpecIdentity:
    normalized = normalize_spec_identity_item(item)
    return build_spec_identity(
        algorithm=normalized["algorithm"],
        model=normalized["model"],
        condition_label=normalized["condition_label"],
        graph_source=normalized.get("graph_source", "default"),
        pair_name=normalized["pair_name"],
        condition_bits=normalized["condition_bits"],
        replication=normalized["replication"],
    )

def _coerce_identity_replication(replication: object) -> int:
    if isinstance(replication, bool):
        raise TypeError("Shard manifest replication must be an integer or numeric string.")
    if isinstance(replication, int):
        return replication
    if isinstance(replication, str):
        return int(replication)
    raise TypeError("Shard manifest replication must be an integer or numeric string.")
