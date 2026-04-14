"""Spec-identity and run-directory path resolution helpers.

Pure functions that project HFRunSpec data into identity keys and
compute deterministic run-directory paths.  No runtime logic, no
subprocess calls, no state mutation.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.hf_batch.types import HFRunSpec
from llm_conceptual_modeling.hf_batch.utils import slugify_model


def spec_identity(spec: HFRunSpec) -> tuple[str, str, str, str, str, int]:
    """Return a deterministic tuple key for a run spec."""
    return (
        spec.algorithm,
        spec.model,
        spec.condition_label,
        spec.pair_name,
        spec.condition_bits,
        spec.replication,
    )


def smoke_spec_identity(spec: HFRunSpec) -> dict[str, object]:
    """Return a dict key suitable for smoke-test verdict files."""
    return {
        "algorithm": spec.algorithm,
        "model": spec.model,
        "embedding_model": spec.embedding_model,
        "decoding_algorithm": spec.decoding.algorithm,
        "pair_name": spec.pair_name,
        "condition_bits": spec.condition_bits,
        "replication": spec.replication,
    }


def run_dir_for_spec(*, output_root: Path, spec: HFRunSpec) -> Path:
    """Compute the deterministic run directory for a spec."""
    return (
        output_root
        / "runs"
        / spec.algorithm
        / slugify_model(spec.model)
        / spec.condition_label
        / spec.pair_name
        / spec.condition_bits
        / f"rep_{spec.replication:02d}"
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
        (
            str(item["algorithm"]),
            str(item["model"]),
            str(item["condition_label"]),
            str(item["pair_name"]),
            str(item["condition_bits"]),
            int(item["replication"]),
        )
        for item in manifest.get("identities", [])
    }
    return [spec for spec in planned_specs if spec_identity(spec) in allowed_identities]
