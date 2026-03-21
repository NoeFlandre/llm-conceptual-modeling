from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | Path


@dataclass(frozen=True)
class GenerationManifest:
    algorithm: str
    mode: str
    implemented: bool
    requires_live_llm: bool
    fixture_only: bool
    next_step: str
    input_data: dict[str, str]
    condition_count: int
    replications: int
    subgraph_pairs: list[str]
    prompt_preview: str
    method_contract: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    name: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status}


@dataclass(frozen=True)
class MultiMetricFactorialSpec:
    factor_columns: list[str]
    metric_columns: list[str]
    output_columns: list[str]
