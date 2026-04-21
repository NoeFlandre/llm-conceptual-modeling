from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.common.types import Edge as _Edge

Edge = _Edge


class _RuntimeResultOptional(TypedDict, total=False):
    summary: dict[str, object]


class RuntimeResult(_RuntimeResultOptional):
    raw_row: dict[str, object]
    runtime: dict[str, object]
    raw_response: str


class BatchInfrastructureFailure(RuntimeError):
    """Abort a batch when the worker host is unhealthy."""


@dataclass(frozen=True)
class HFRunSpec:
    algorithm: str
    model: str
    embedding_model: str
    decoding: DecodingConfig
    replication: int
    pair_name: str
    condition_bits: str
    condition_label: str
    prompt_factors: dict[str, bool | int]
    raw_context: dict[str, object]
    input_payload: dict[str, object]
    runtime_profile: RuntimeProfile
    prompt_bundle: dict[str, str] | None = None
    max_new_tokens_by_schema: dict[str, int] | None = None
    context_policy: dict[str, object] | None = None
    base_seed: int = 0
    seed: int = 0
    graph_source: str = "default"

    @property
    def run_name(self) -> str:
        return (
            f"{self.algorithm}_{self.pair_name}_rep{self.replication:02d}_cond{self.condition_bits}"
        )


RuntimeFactory = Callable[[HFRunSpec], RuntimeResult]
