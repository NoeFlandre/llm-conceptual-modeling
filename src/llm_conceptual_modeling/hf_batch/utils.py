from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from llm_conceptual_modeling.algo1.mistral import Method1PromptConfig
from llm_conceptual_modeling.algo2.mistral import Method2PromptConfig
from llm_conceptual_modeling.algo3.mistral import Method3PromptConfig
from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.common.io import write_json_dict
from llm_conceptual_modeling.hf_batch.types import Edge, HFRunSpec


def manifest_for_spec(spec: HFRunSpec) -> dict[str, object]:
    return {
        "provider": "hf-transformers",
        "algorithm": spec.algorithm,
        "model": spec.model,
        "embedding_model": spec.embedding_model,
        "graph_source": spec.graph_source,
        "temperature": spec.decoding.temperature,
        "base_seed": spec.base_seed,
        "seed": spec.seed,
        "decoding": asdict(spec.decoding),
        "replication": spec.replication,
        "pair_name": spec.pair_name,
        "condition_bits": spec.condition_bits,
        "condition_label": spec.condition_label,
        "prompt_factors": spec.prompt_factors,
        "runtime": runtime_details(spec.runtime_profile),
        "input_payload": stringify_payload(spec.input_payload),
    }


def stringify_payload(payload: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in payload.items():
        serialized[key] = repr(value) if isinstance(value, list) else value
    return serialized


def runtime_details(profile: RuntimeProfile) -> dict[str, object]:
    return {
        "device": profile.device,
        "dtype": profile.dtype,
        "quantization": profile.quantization,
        "thinking_mode_supported": profile.supports_thinking_toggle,
        "thinking_mode_control": (
            "explicit-disable" if profile.supports_thinking_toggle else "not-supported-by-model"
        ),
        "context_limit": profile.context_limit,
    }


def resolve_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def derive_run_seed(
    *,
    base_seed: int,
    algorithm: str,
    model: str,
    pair_name: str,
    condition_bits: str,
    decoding: DecodingConfig,
    replication: int,
) -> int:
    digest = sha256(
        (
            f"{base_seed}|{algorithm}|{model}|{pair_name}|{condition_bits}|"
            f"{condition_label(decoding)}|{replication}"
        ).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def default_runtime_profile_provider(_model: str) -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=False,
        context_limit=None,
    )


def condition_label(decoding: DecodingConfig) -> str:
    if decoding.algorithm == "greedy":
        return "greedy"
    if decoding.algorithm == "beam":
        return f"beam_num_beams_{decoding.num_beams}"
    return f"contrastive_penalty_alpha_{decoding.penalty_alpha}"


def slugify_model(model: str) -> str:
    return model.replace("/", "__")


def coerce_edges(raw_edges: object) -> list[Edge]:
    if not isinstance(raw_edges, list):
        raise ValueError(f"Expected list of edges, got {type(raw_edges)!r}")
    edges: list[Edge] = []
    for edge in raw_edges:
        if not isinstance(edge, tuple | list) or len(edge) != 2:
            raise ValueError(f"Invalid edge payload: {edge!r}")
        edges.append((str(edge[0]), str(edge[1])))
    return edges


def collect_nodes(edges: list[Edge]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for source, target in edges:
        if source not in seen:
            seen.add(source)
            ordered.append(source)
        if target not in seen:
            seen.add(target)
            ordered.append(target)
    return ordered


def algo1_prompt_config(prompt_factors: dict[str, bool | int]) -> Method1PromptConfig:
    return Method1PromptConfig(
        use_adjacency_notation=bool(prompt_factors["use_adjacency_notation"]),
        use_array_representation=bool(prompt_factors["use_array_representation"]),
        include_explanation=bool(prompt_factors["include_explanation"]),
        include_example=bool(prompt_factors["include_example"]),
        include_counterexample=bool(prompt_factors["include_counterexample"]),
    )


def algo2_prompt_config(prompt_factors: dict[str, bool | int]) -> Method2PromptConfig:
    return Method2PromptConfig(
        use_adjacency_notation=bool(prompt_factors["use_adjacency_notation"]),
        use_array_representation=bool(prompt_factors["use_array_representation"]),
        include_explanation=bool(prompt_factors["include_explanation"]),
        include_example=bool(prompt_factors["include_example"]),
        include_counterexample=bool(prompt_factors["include_counterexample"]),
        use_relaxed_convergence=bool(prompt_factors["use_relaxed_convergence"]),
    )


def algo3_prompt_config(prompt_factors: dict[str, bool | int]) -> Method3PromptConfig:
    return Method3PromptConfig(
        include_example=bool(prompt_factors["include_example"]),
        include_counterexample=bool(prompt_factors["include_counterexample"]),
    )


class RecordingChatClient:
    def __init__(
        self,
        inner: Any,
        *,
        persist_path: Path | None = None,
        active_stage_path: Path | None = None,
        active_stage_context: dict[str, object] | None = None,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        self._inner = inner
        self._persist_path = persist_path
        self._active_stage_path = active_stage_path
        self._active_stage_context = active_stage_context or {}
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self.records: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        self._write_active_stage(
            status="running",
            schema_name=schema_name,
            prompt=prompt,
        )
        heartbeat_stop = threading.Event()
        heartbeat_thread = self._start_active_stage_heartbeat(
            heartbeat_stop=heartbeat_stop,
            schema_name=schema_name,
            prompt=prompt,
        )
        try:
            response = self._inner.complete_json(
                prompt=prompt,
                schema_name=schema_name,
                schema=schema,
            )
        except Exception as error:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=self._heartbeat_interval_seconds)
            record = {
                "prompt": prompt,
                "schema_name": schema_name,
                "error": str(error),
                "metrics": getattr(self._inner, "last_call_metrics", None),
            }
            raw_text = getattr(self._inner, "last_failed_response_text", None)
            if raw_text is not None:
                record["raw_text"] = raw_text
            self.records.append(record)
            if self._persist_path is not None:
                write_text(
                    self._persist_path,
                    json.dumps(self.records, indent=2, sort_keys=True),
                )
            self._write_active_stage(
                status="failed",
                schema_name=schema_name,
                prompt=prompt,
                error=str(error),
                raw_text=raw_text,
            )
            raise
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=self._heartbeat_interval_seconds)
        self.records.append(
            {
                "prompt": prompt,
                "schema_name": schema_name,
                "response": response,
                "metrics": getattr(self._inner, "last_call_metrics", None),
            }
        )
        if self._persist_path is not None:
            write_text(
                self._persist_path,
                json.dumps(self.records, indent=2, sort_keys=True),
            )
        self._write_active_stage(
            status="completed",
            schema_name=schema_name,
            prompt=prompt,
            response=response,
        )
        return response

    def _start_active_stage_heartbeat(
        self,
        *,
        heartbeat_stop: threading.Event,
        schema_name: str,
        prompt: str,
    ) -> threading.Thread | None:
        if self._active_stage_path is None:
            return None

        def heartbeat() -> None:
            while not heartbeat_stop.wait(self._heartbeat_interval_seconds):
                self._write_active_stage(
                    status="running",
                    schema_name=schema_name,
                    prompt=prompt,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        return thread

    def _write_active_stage(
        self,
        *,
        status: str,
        schema_name: str,
        prompt: str,
        response: dict[str, object] | None = None,
        error: str | None = None,
        raw_text: str | None = None,
    ) -> None:
        if self._active_stage_path is None:
            return
        payload = {
            **self._active_stage_context,
            "status": status,
            "schema_name": schema_name,
            "prompt": prompt,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if response is not None:
            payload["response"] = response
            metrics = getattr(self._inner, "last_call_metrics", None)
            if metrics is not None:
                payload["metrics"] = metrics
        if error is not None:
            payload["error"] = error
        if raw_text is not None:
            payload["raw_text"] = raw_text
        write_json(self._active_stage_path, payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_dict(path, payload)


def write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def add_decoding_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "decoding_algorithm" not in frame.columns:
        return frame
    augmented = frame.copy()
    augmented["Decoding Algorithm"] = augmented["decoding_algorithm"].astype(str)
    augmented["Beam Width Level"] = augmented.apply(_beam_width_level, axis=1)
    augmented["Contrastive Penalty Level"] = augmented.apply(_contrastive_penalty_level, axis=1)
    return augmented


def beam_width_level(row: pd.Series) -> int:
    return _beam_width_level(row)


def contrastive_penalty_level(row: pd.Series) -> int:
    return _contrastive_penalty_level(row)


def _beam_width_level(row: pd.Series) -> int:
    if str(row["decoding_algorithm"]) != "beam":
        return 0
    condition = str(row.get("decoding_condition", ""))
    return 1 if condition.endswith("_6") else -1


def _contrastive_penalty_level(row: pd.Series) -> int:
    if str(row["decoding_algorithm"]) != "contrastive":
        return 0
    condition = str(row.get("decoding_condition", ""))
    return 1 if condition.endswith("_0.8") else -1
