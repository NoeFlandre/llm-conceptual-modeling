from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import yaml

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


@dataclass(frozen=True)
class GeneralizedFactorialSpec:
    factor_columns: list[str]
    metric_columns: list[str]
    output_columns: list[str]
    replication_column: str | None = None
    include_pairwise_interactions: bool = True


@dataclass(frozen=True)
class ExperimentManifest:
    """YAML manifest capturing all experiment configuration for reproducibility.

    This manifest is written to ``manifest.yaml`` in each experiment output directory
    and captures every parameter needed to reproduce an experiment exactly.
    """

    experiment_id: str
    algorithm: str
    model: str
    provider: str
    temperature: float
    top_p: float | None
    max_tokens: int | None
    prompt_factors: dict[str, bool | int]
    full_prompt: str
    input_subgraph_pairs: list[dict[str, Any]]
    output_dir: str
    timestamp: str
    repetitions: int
    condition_bits: str
    pair_name: str
    embedding_provider: str | None = None
    embedding_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Path) -> None:
        """Write this manifest to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentManifest":
        """Reconstruct an ExperimentManifest from a dictionary (e.g., parsed YAML)."""
        return cls(
            experiment_id=str(data["experiment_id"]),
            algorithm=str(data["algorithm"]),
            model=str(data["model"]),
            provider=str(data["provider"]),
            temperature=float(data["temperature"]),
            top_p=float(data["top_p"]) if data.get("top_p") is not None else None,
            max_tokens=int(data["max_tokens"]) if data.get("max_tokens") is not None else None,
            prompt_factors=dict(data["prompt_factors"]),
            full_prompt=str(data["full_prompt"]),
            input_subgraph_pairs=list(data["input_subgraph_pairs"]),
            output_dir=str(data["output_dir"]),
            timestamp=str(data["timestamp"]),
            repetitions=int(data["repetitions"]),
            condition_bits=str(data["condition_bits"]),
            pair_name=str(data["pair_name"]),
            embedding_provider=str(data["embedding_provider"])
            if data.get("embedding_provider") is not None
            else None,
            embedding_model=str(data["embedding_model"])
            if data.get("embedding_model") is not None
            else None,
        )

    @staticmethod
    def prompt_config_to_factors(config: Any, algorithm: str) -> dict[str, bool | int]:
        """Convert a prompt config dataclass to a factors dictionary.

        Parameters
        ----------
        config
            A Method1PromptConfig, Method2PromptConfig, or Method3PromptConfig instance.
        algorithm
            One of "algo1", "algo2", "algo3".
        """
        if algorithm in ("algo1", "algo2"):
            factors = {
                "use_adjacency_notation": config.use_adjacency_notation,
                "use_array_representation": config.use_array_representation,
                "include_explanation": config.include_explanation,
                "include_example": config.include_example,
                "include_counterexample": config.include_counterexample,
            }
            if algorithm == "algo2":
                factors["use_relaxed_convergence"] = config.use_relaxed_convergence
            return factors
        if algorithm == "algo3":
            return {
                "include_example": config.include_example,
                "include_counterexample": config.include_counterexample,
            }
        msg = f"Unsupported algorithm: {algorithm}"
        raise ValueError(msg)

    @classmethod
    def from_probe_spec(
        cls,
        spec: Any,
        algorithm: str,
        provider: str,
        temperature: float,
        top_p: float | None,
        max_tokens: int | None,
        full_prompt: str,
        pair_name: str,
        condition_bits: str,
        repetitions: int,
    ) -> "ExperimentManifest":
        """Construct an ExperimentManifest from a probe spec object.

        Parameters
        ----------
        spec
            An Algo1ProbeSpec, Algo2ProbeSpec, or Algo3ProbeSpec instance.
        algorithm
            One of "algo1", "algo2", "algo3".
        provider
            The LLM provider ("mistral" or "anthropic").
        embedding_provider
            The embedding provider, when relevant.
        temperature
            LLM sampling temperature.
        top_p
            LLM top_p sampling parameter, or None.
        max_tokens
            LLM max_tokens parameter, or None.
        full_prompt
            The complete prompt text used for this experiment run.
        pair_name
            Human-readable name for the subgraph pair.
        condition_bits
            Bitstring encoding the condition (e.g., "00000").
        repetitions
            Total number of repetitions in the full experiment.
        """
        # Build input subgraph pairs from the probe spec
        if algorithm in ("algo1", "algo2"):
            input_pairs = [
                {"subgraph_name": "subgraph1", "edges": [list(e) for e in spec.subgraph1]},
                {"subgraph_name": "subgraph2", "edges": [list(e) for e in spec.subgraph2]},
            ]
        elif algorithm == "algo3":
            input_pairs = [
                {"subgraph_name": "source_labels", "labels": list(spec.source_labels)},
                {"subgraph_name": "target_labels", "labels": list(spec.target_labels)},
            ]
        else:
            input_pairs = []

        return cls(
            experiment_id=spec.run_name,
            algorithm=algorithm,
            model=spec.model,
            provider=provider,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_factors=cls.prompt_config_to_factors(spec.prompt_config, algorithm),
            full_prompt=full_prompt,
            input_subgraph_pairs=input_pairs,
            output_dir=str(spec.output_dir),
            timestamp=datetime.now(UTC).isoformat(),
            repetitions=repetitions,
            condition_bits=condition_bits,
            pair_name=pair_name,
            embedding_provider=getattr(spec, "embedding_provider", None),
            embedding_model=getattr(spec, "embedding_model", None),
        )
