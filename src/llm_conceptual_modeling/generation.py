import json

from llm_conceptual_modeling.algo1.generation import (
    build_generation_manifest as build_algo1_manifest,
)
from llm_conceptual_modeling.algo2.generation import (
    build_generation_manifest as build_algo2_manifest,
)
from llm_conceptual_modeling.algo3.generation import (
    build_generation_manifest as build_algo3_manifest,
)


def build_generation_stub_payload(
    algorithm: str,
    *,
    fixture_only: bool,
) -> dict[str, object]:
    builders = {
        "algo1": build_algo1_manifest,
        "algo2": build_algo2_manifest,
        "algo3": build_algo3_manifest,
    }
    return builders[algorithm](fixture_only=fixture_only)


def emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload))
