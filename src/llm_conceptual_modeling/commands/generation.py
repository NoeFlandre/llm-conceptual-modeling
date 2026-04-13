import json
from dataclasses import asdict
from typing import Any, cast

from llm_conceptual_modeling.algo1.generation import (
    build_generation_manifest as build_algo1_manifest,
)
from llm_conceptual_modeling.algo2.generation import (
    build_generation_manifest as build_algo2_manifest,
)
from llm_conceptual_modeling.algo3.generation import (
    build_generation_manifest as build_algo3_manifest,
)
from llm_conceptual_modeling.hf_run_config import load_hf_run_config


def build_generation_stub_payload(
    algorithm: str,
    *,
    fixture_only: bool,
    provider: str = "mistral",
    config_path: str | None = None,
) -> dict[str, object]:
    builders = {
        "algo1": build_algo1_manifest,
        "algo2": build_algo2_manifest,
        "algo3": build_algo3_manifest,
    }
    payload = builders[algorithm](fixture_only=fixture_only)
    if provider == "hf-transformers":
        payload["provider"] = "hf-transformers"
        if config_path is not None:
            config = load_hf_run_config(config_path)
            payload["chat_models"] = config.models.chat_models
            payload["embedding_models"] = [config.models.embedding_model]
            payload["replications"] = config.run.replications
            payload["seed"] = config.runtime.seed
            payload["decoding_conditions"] = [asdict(item) for item in config.decoding]
            payload["supported_decoding_algorithms"] = sorted(
                {item.algorithm for item in config.decoding}
            )
        else:
            payload["chat_models"] = [
                "mistralai/Ministral-3-8B-Instruct-2512",
                "Qwen/Qwen3.5-9B",
                "allenai/Olmo-3-7B-Instruct",
            ]
            payload["embedding_models"] = ["Qwen/Qwen3-Embedding-0.6B"]
            payload["supported_decoding_algorithms"] = ["greedy", "beam", "contrastive"]
        if algorithm == "algo2":
            method_contract = cast(dict[str, Any], payload["method_contract"])
            method_contract["embedding_model"] = payload["embedding_models"][0]
    return payload


def emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload))
