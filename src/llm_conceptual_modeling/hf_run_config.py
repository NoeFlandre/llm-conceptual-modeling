from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    supports_explicit_thinking_disable,
)


@dataclass(frozen=True)
class RunConfig:
    provider: str
    output_root: str
    replications: int


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    temperature: float
    quantization: str
    device_policy: str
    context_policy: dict[str, object]
    max_new_tokens_by_schema: dict[str, int]
    thinking_mode_by_model: dict[str, str]


@dataclass(frozen=True)
class ModelsConfig:
    chat_models: list[str]
    embedding_model: str


@dataclass(frozen=True)
class FactorConfig:
    column: str
    levels: list[int]
    low_fragments: list[str]
    high_fragments: list[str]
    runtime_field: str | None = None
    low_runtime_value: bool | int | None = None
    high_runtime_value: bool | int | None = None


@dataclass(frozen=True)
class AlgorithmPromptConfig:
    name: str
    base_fragments: list[str]
    factors: dict[str, FactorConfig]
    fragment_definitions: dict[str, str]
    prompt_templates: dict[str, str]
    pair_names: list[str] | None = None

    def assemble_prompt(
        self,
        active_high_factors: list[str],
        *,
        template_name: str = "body",
        template_values: dict[str, object] | None = None,
    ) -> str:
        fragments = list(self.base_fragments)
        for factor_name, factor in self.factors.items():
            if factor_name in active_high_factors:
                fragments.extend(factor.high_fragments)
            else:
                fragments.extend(factor.low_fragments)

        parts = [self._resolve_fragment(fragment_name) for fragment_name in fragments]
        template_text = self.prompt_templates[template_name]
        resolved_template = template_text.format_map(
            _PromptValueMap(
                {
                    **self.fragment_definitions,
                    **(template_values or {}),
                }
            )
        )
        parts.append(resolved_template)
        return " ".join(part.strip() for part in parts if part.strip())

    def resolve_runtime_fields(
        self,
        active_high_factors: list[str],
    ) -> dict[str, bool | int]:
        resolved: dict[str, bool | int] = {}
        for factor_name, factor in self.factors.items():
            if factor.runtime_field is None:
                continue
            if factor_name in active_high_factors:
                value = factor.high_runtime_value
            else:
                value = factor.low_runtime_value
            if value is None:
                continue
            resolved[factor.runtime_field] = value
        return resolved

    def factor_condition_count(self) -> int:
        return 2 ** len(self.factors)

    def _resolve_fragment(self, fragment_name: str) -> str:
        if fragment_name not in self.fragment_definitions:
            raise ValueError(f"Unknown fragment reference: {fragment_name}")
        return self.fragment_definitions[fragment_name]


@dataclass(frozen=True)
class HFRunConfig:
    run: RunConfig
    runtime: RuntimeConfig
    models: ModelsConfig
    decoding: list[DecodingConfig]
    graph_source: str
    shared_fragments: dict[str, str]
    algorithms: dict[str, AlgorithmPromptConfig]

    def to_dict(self) -> dict[str, object]:
        return {
            "run": asdict(self.run),
            "runtime": asdict(self.runtime),
            "models": asdict(self.models),
            "decoding": [asdict(config) for config in self.decoding],
            "inputs": {"graph_source": self.graph_source},
            "shared_fragments": dict(self.shared_fragments),
            "algorithms": {
                name: {
                    "base_fragments": value.base_fragments,
                    "factors": {
                        factor_name: asdict(factor)
                        for factor_name, factor in value.factors.items()
                    },
                    "fragment_definitions": value.fragment_definitions,
                    "prompt_templates": value.prompt_templates,
                    "pair_names": value.pair_names,
                }
                for name, value in self.algorithms.items()
            },
        }


def load_hf_run_config(path: str | Path) -> HFRunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    shared_fragments = {
        str(name): str(text) for name, text in dict(raw.get("shared_fragments", {})).items()
    }
    algorithms: dict[str, AlgorithmPromptConfig] = {}

    for algorithm_name, algorithm_payload in dict(raw["algorithms"]).items():
        fragment_definitions = dict(shared_fragments)
        fragment_definitions.update(
            {
                str(name): str(text)
                for name, text in dict(algorithm_payload.get("fragment_definitions", {})).items()
            }
        )
        factor_payloads = dict(algorithm_payload.get("factors", {}))
        factors = {
            str(name): _load_factor_config(payload)
            for name, payload in factor_payloads.items()
        }
        algorithm_config = AlgorithmPromptConfig(
            name=str(algorithm_name),
            base_fragments=[
                str(name) for name in list(algorithm_payload.get("base_fragments", []))
            ],
            factors=factors,
            fragment_definitions=fragment_definitions,
            prompt_templates={
                str(name): str(text)
                for name, text in dict(algorithm_payload.get("prompt_templates", {})).items()
            },
            pair_names=[
                str(name) for name in list(algorithm_payload.get("pair_names", []))
            ]
            or None,
        )
        _validate_algorithm_fragments(algorithm_config)
        algorithms[str(algorithm_name)] = algorithm_config

    config = HFRunConfig(
        run=RunConfig(
            provider=str(raw["run"]["provider"]),
            output_root=str(raw["run"]["output_root"]),
            replications=int(raw["run"]["replications"]),
        ),
        runtime=RuntimeConfig(
            seed=int(raw["runtime"]["seed"]),
            temperature=float(raw["runtime"]["temperature"]),
            quantization=str(raw["runtime"]["quantization"]),
            device_policy=str(raw["runtime"]["device_policy"]),
            context_policy=dict(raw["runtime"]["context_policy"]),
            max_new_tokens_by_schema={
                str(name): int(value)
                for name, value in dict(raw["runtime"]["max_new_tokens_by_schema"]).items()
            },
            thinking_mode_by_model={
                str(name): str(value)
                for name, value in dict(raw["runtime"].get("thinking_mode_by_model", {})).items()
            },
        ),
        models=ModelsConfig(
            chat_models=[str(model) for model in list(raw["models"]["chat_models"])],
            embedding_model=str(raw["models"]["embedding_model"]),
        ),
        decoding=_load_decoding_configs(
            raw["decoding"],
            temperature=float(raw["runtime"]["temperature"]),
        ),
        graph_source=str(raw["inputs"]["graph_source"]),
        shared_fragments=shared_fragments,
        algorithms=algorithms,
    )
    _validate_top_level_config(config)
    return config


def write_resolved_run_preview(*, config: HFRunConfig, output_dir: str | Path) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    resolved_yaml = yaml.safe_dump(config.to_dict(), sort_keys=False)
    (output_dir_path / "resolved_run_config.yaml").write_text(resolved_yaml, encoding="utf-8")

    plan_summary = {
        "provider": config.run.provider,
        "output_root": config.run.output_root,
        "replications": config.run.replications,
        "seed": config.runtime.seed,
        "chat_models": config.models.chat_models,
        "embedding_model": config.models.embedding_model,
        "decoding_conditions": [asdict(item) for item in config.decoding],
        "algorithm_condition_counts": {
            name: value.factor_condition_count() for name, value in config.algorithms.items()
        },
    }
    (output_dir_path / "resolved_run_plan.json").write_text(
        json.dumps(plan_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    prompt_root = output_dir_path / "prompt_preview"
    for algorithm_name, algorithm_config in config.algorithms.items():
        algorithm_dir = prompt_root / algorithm_name
        algorithm_dir.mkdir(parents=True, exist_ok=True)
        preview_bundle = _build_preview_bundle(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
        )
        preview_names = list(preview_bundle.keys())
        if preview_names:
            first_preview = preview_bundle[preview_names[0]]
            (algorithm_dir / "base.txt").write_text(first_preview, encoding="utf-8")
        for preview_name, preview_text in preview_bundle.items():
            (algorithm_dir / f"{preview_name}.txt").write_text(preview_text, encoding="utf-8")
        _write_condition_prompt_previews(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            output_dir=algorithm_dir / "conditions",
        )
    _write_condition_matrix(config=config, output_dir=output_dir_path)


def _load_factor_config(payload: object) -> FactorConfig:
    data = _expect_mapping(payload, label="Factor payload")
    levels = _expect_list(data.get("levels"), label="Factor levels")
    low_fragments = _expect_list(data.get("low_fragments", []), label="Factor low_fragments")
    high_fragments = _expect_list(data.get("high_fragments", []), label="Factor high_fragments")
    return FactorConfig(
        column=str(data["column"]),
        levels=[_coerce_int(value, label="Factor level") for value in levels],
        low_fragments=[str(value) for value in low_fragments],
        high_fragments=[str(value) for value in high_fragments],
        runtime_field=str(data["runtime_field"]) if data.get("runtime_field") is not None else None,
        low_runtime_value=_coerce_runtime_value(data.get("low_runtime_value")),
        high_runtime_value=_coerce_runtime_value(data.get("high_runtime_value")),
    )


def _load_decoding_configs(raw: object, *, temperature: float) -> list[DecodingConfig]:
    payload = _expect_mapping(raw, label="Decoding payload")
    configs: list[DecodingConfig] = []
    if _expect_mapping(payload.get("greedy", {}), label="Greedy decoding").get("enabled", False):
        configs.append(DecodingConfig(algorithm="greedy", temperature=temperature))
    beam_payload = _expect_mapping(payload.get("beam", {}), label="Beam decoding")
    if beam_payload.get("enabled", False):
        for num_beams in _expect_list(beam_payload.get("num_beams", []), label="Beam num_beams"):
            configs.append(
                DecodingConfig(
                    algorithm="beam",
                    num_beams=_coerce_int(num_beams, label="Beam num_beams"),
                    temperature=temperature,
                )
            )
    contrastive_payload = _expect_mapping(
        payload.get("contrastive", {}),
        label="Contrastive decoding",
    )
    if contrastive_payload.get("enabled", False):
        top_k = _coerce_int(contrastive_payload["top_k"], label="Contrastive top_k")
        penalty_alphas = _expect_list(
            contrastive_payload.get("penalty_alpha", []),
            label="Contrastive penalty_alpha",
        )
        for penalty_alpha in penalty_alphas:
            configs.append(
                DecodingConfig(
                    algorithm="contrastive",
                    penalty_alpha=_coerce_float(
                        penalty_alpha,
                        label="Contrastive penalty_alpha",
                    ),
                    top_k=top_k,
                    temperature=temperature,
                )
            )
    for config in configs:
        config.validate()
    return configs


def _validate_algorithm_fragments(config: AlgorithmPromptConfig) -> None:
    referenced_fragments = list(config.base_fragments)
    for factor in config.factors.values():
        referenced_fragments.extend(factor.low_fragments)
        referenced_fragments.extend(factor.high_fragments)
    for fragment_name in referenced_fragments:
        if fragment_name not in config.fragment_definitions:
            raise ValueError(f"Unknown fragment reference: {fragment_name}")


class _PromptValueMap(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _expect_mapping(value: object, *, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value)!r}")
    return {str(key): item for key, item in value.items()}


def _expect_list(value: object, *, label: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list, got {type(value)!r}")
    return cast(list[object], value)


def _coerce_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{label} must be int-like, got {type(value)!r}")


def _coerce_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got boolean {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"{label} must be float-like, got {type(value)!r}")


def _coerce_runtime_value(value: object) -> bool | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return _coerce_int(value, label="Runtime value")


def _validate_top_level_config(config: HFRunConfig) -> None:
    if config.run.provider != "hf-transformers":
        raise ValueError("HF run config provider must be hf-transformers.")
    if config.runtime.quantization != "none":
        raise ValueError("HF run config forbids quantization.")
    if config.runtime.device_policy != "cuda-only":
        raise ValueError("HF run config requires device_policy=cuda-only.")
    if config.runtime.context_policy.get("prompt_truncation") != "forbid":
        raise ValueError("HF run config requires prompt_truncation=forbid.")
    if config.graph_source != "default":
        raise ValueError("Only the default graph source is supported in HF run configs.")
    if not config.models.chat_models:
        raise ValueError("HF run config must list at least one chat model.")
    if not config.decoding:
        raise ValueError("HF run config must enable at least one decoding condition.")
    for model in config.models.chat_models:
        declared_mode = config.runtime.thinking_mode_by_model.get(model)
        if declared_mode is None:
            raise ValueError(f"HF run config must declare thinking_mode_by_model for {model}.")
        if supports_explicit_thinking_disable(model):
            if declared_mode != "disabled":
                raise ValueError(
                    f"Model {model} supports explicit thinking disable and must be set to "
                    "'disabled'."
                )
            continue
        if declared_mode != "acknowledged-unsupported":
            raise ValueError(
                f"Model {model} does not support explicit thinking disable and must be "
                "'acknowledged-unsupported'."
            )


def _default_preview_template_name(config: AlgorithmPromptConfig) -> str:
    if "body" in config.prompt_templates:
        return "body"
    return next(iter(config.prompt_templates))


def _build_preview_bundle(
    *,
    algorithm_name: str,
    algorithm_config: AlgorithmPromptConfig,
) -> dict[str, str]:
    prompt_factors: dict[str, bool | int] = {}
    active_high_factors: list[str] = []
    for factor_name, factor in algorithm_config.factors.items():
        if factor.runtime_field is None or factor.low_runtime_value is None:
            continue
        prompt_factors[factor.runtime_field] = factor.low_runtime_value
        if factor.high_runtime_value == factor.low_runtime_value:
            active_high_factors.append(factor_name)

    try:
        from llm_conceptual_modeling.hf_experiments import _build_prompt_bundle

        return _build_prompt_bundle(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            active_high_factors=active_high_factors,
            prompt_factors=prompt_factors,
        )
    except (ImportError, KeyError, ValueError):
        preview_text = algorithm_config.assemble_prompt(
            [],
            template_name=_default_preview_template_name(algorithm_config),
        )
        return {"base": preview_text}


def _write_condition_prompt_previews(
    *,
    algorithm_name: str,
    algorithm_config: AlgorithmPromptConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition_bits, active_high_factors, prompt_factors in _iter_factor_conditions(
        algorithm_config
    ):
        preview_bundle = _build_preview_bundle_for_condition(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            active_high_factors=active_high_factors,
            prompt_factors=prompt_factors,
        )
        condition_dir = output_dir / condition_bits
        condition_dir.mkdir(parents=True, exist_ok=True)
        for preview_name, preview_text in preview_bundle.items():
            (condition_dir / f"{preview_name}.txt").write_text(preview_text, encoding="utf-8")
        (condition_dir / "runtime_fields.json").write_text(
            json.dumps(prompt_factors, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _write_condition_matrix(*, config: HFRunConfig, output_dir: Path) -> None:
    from llm_conceptual_modeling.hf_experiments import plan_paper_batch

    planned_specs = plan_paper_batch(
        models=config.models.chat_models,
        embedding_model=config.models.embedding_model,
        replications=config.run.replications,
        config=config,
    )
    rows = [
        {
            "algorithm": spec.algorithm,
            "model": spec.model,
            "decoding_condition": spec.condition_label,
            "replication": spec.replication,
            "pair_name": spec.pair_name,
            "condition_bits": spec.condition_bits,
        }
        for spec in planned_specs
    ]
    frame = pd.DataFrame.from_records(rows)
    frame.to_csv(output_dir / "condition_matrix.csv", index=False)


def _iter_factor_conditions(
    algorithm_config: AlgorithmPromptConfig,
) -> list[tuple[str, list[str], dict[str, bool | int]]]:
    factor_names = list(algorithm_config.factors.keys())
    if not factor_names:
        return [("base", [], {})]
    conditions: list[tuple[str, list[str], dict[str, bool | int]]] = []
    for levels in product((False, True), repeat=len(factor_names)):
        active_high_factors = [
            factor_name
            for factor_name, is_high in zip(factor_names, levels, strict=True)
            if is_high
        ]
        condition_bits = "".join("1" if is_high else "0" for is_high in levels)
        prompt_factors = algorithm_config.resolve_runtime_fields(active_high_factors)
        conditions.append((condition_bits, active_high_factors, prompt_factors))
    return conditions


def _build_preview_bundle_for_condition(
    *,
    algorithm_name: str,
    algorithm_config: AlgorithmPromptConfig,
    active_high_factors: list[str],
    prompt_factors: dict[str, bool | int],
) -> dict[str, str]:
    try:
        from llm_conceptual_modeling.hf_experiments import _build_prompt_bundle

        return _build_prompt_bundle(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            active_high_factors=active_high_factors,
            prompt_factors=prompt_factors,
        )
    except (ImportError, KeyError, ValueError):
        preview_text = algorithm_config.assemble_prompt(
            active_high_factors,
            template_name=_default_preview_template_name(algorithm_config),
        )
        return {"base": preview_text}
