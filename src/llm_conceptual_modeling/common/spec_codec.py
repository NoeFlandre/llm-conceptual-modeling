from __future__ import annotations

from dataclasses import asdict
from typing import Any

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.common.io import coerce_int
from llm_conceptual_modeling.hf_batch.types import HFRunSpec


def serialize_spec(spec: HFRunSpec) -> dict[str, object]:
    return {
        **asdict(spec),
        "decoding": asdict(spec.decoding),
        "runtime_profile": asdict(spec.runtime_profile),
    }


def deserialize_spec(payload: dict[str, Any]) -> HFRunSpec:
    return HFRunSpec(
        algorithm=str(payload["algorithm"]),
        model=str(payload["model"]),
        embedding_model=str(payload["embedding_model"]),
        decoding=DecodingConfig(**payload["decoding"]),
        replication=_required_int(payload, field_name="replication"),
        pair_name=str(payload["pair_name"]),
        condition_bits=str(payload["condition_bits"]),
        condition_label=str(payload["condition_label"]),
        prompt_factors=dict(payload["prompt_factors"]),
        raw_context=dict(payload["raw_context"]),
        input_payload=dict(payload["input_payload"]),
        runtime_profile=RuntimeProfile(**payload["runtime_profile"]),
        prompt_bundle=dict(payload["prompt_bundle"]) if payload.get("prompt_bundle") else None,
        max_new_tokens_by_schema=(
            dict(payload["max_new_tokens_by_schema"])
            if payload.get("max_new_tokens_by_schema")
            else None
        ),
        context_policy=dict(payload["context_policy"]) if payload.get("context_policy") else None,
        base_seed=coerce_int(payload.get("base_seed", 0)),
        seed=coerce_int(payload.get("seed", 0)),
    )


def _required_int(payload: dict[str, Any], *, field_name: str) -> int:
    raw_value = payload[field_name]
    if isinstance(raw_value, bool):
        raise TypeError(f"Spec field {field_name} must be an integer, got bool")
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, str):
        return int(raw_value)
    raise TypeError(
        f"Spec field {field_name} must be an integer, got {type(raw_value).__name__}"
    )
