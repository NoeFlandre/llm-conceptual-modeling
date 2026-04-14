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
    shared_fragments = _load_string_map(raw.get("shared_fragments", {}), label="shared fragments")
    config = HFRunConfig(
        run=_load_run_config(raw),
        runtime=_load_runtime_config(raw),
        models=_load_models_config(raw),
        decoding=_load_decoding_configs(
            raw["decoding"],
            temperature=float(raw["runtime"]["temperature"]),
        ),
        graph_source=str(raw["inputs"]["graph_source"]),
        shared_fragments=shared_fragments,
        algorithms=_load_algorithm_configs(raw, shared_fragments=shared_fragments),
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


def exclude_decoding_conditions_from_payload(
    payload: dict[str, object],
    *,
    excluded_condition_labels: set[str],
) -> None:
    if not excluded_condition_labels:
        return
    runtime_payload = _expect_mapping(payload.get("runtime"), label="Runtime payload")
    temperature = _coerce_float(
        runtime_payload.get("temperature", 0.0),
        label="Runtime temperature",
    )
    decoding_payload = payload.get("decoding")
    decoded_configs = _load_decoding_configs(decoding_payload, temperature=temperature)
    filtered_configs = [
        config
        for config in decoded_configs
        if _decoding_condition_label(config) not in excluded_condition_labels
    ]
    payload["decoding"] = [asdict(config) for config in filtered_configs]


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
    if isinstance(raw, list):
        configs = _load_resolved_decoding_configs(raw, temperature=temperature)
    else:
        configs = _load_source_decoding_configs(raw, temperature=temperature)
    for config in configs:
        config.validate()
    return configs


def _load_source_decoding_configs(raw: object, *, temperature: float) -> list[DecodingConfig]:
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
    return configs


def _load_resolved_decoding_configs(raw: object, *, temperature: float) -> list[DecodingConfig]:
    configs: list[DecodingConfig] = []
    for index, payload in enumerate(_expect_list(raw, label="Resolved decoding payload")):
        data = _expect_mapping(payload, label=f"Decoding condition {index}")
        configs.append(
            DecodingConfig(
                algorithm=str(data["algorithm"]),
                num_beams=_coerce_optional_int(data.get("num_beams"), label="Decoding num_beams"),
                penalty_alpha=_coerce_optional_float(
                    data.get("penalty_alpha"),
                    label="Decoding penalty_alpha",
                ),
                top_k=_coerce_optional_int(data.get("top_k"), label="Decoding top_k"),
                temperature=float(data.get("temperature", temperature)),
            )
        )
    return configs


def _decoding_condition_label(config: DecodingConfig) -> str:
    if config.algorithm == "greedy":
        return "greedy"
    if config.algorithm == "beam":
        return f"beam_num_beams_{config.num_beams}"
    return f"contrastive_penalty_alpha_{config.penalty_alpha}"


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


def _coerce_optional_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, label=label)


def _coerce_optional_float(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, label=label)


def _load_optional_string_list(raw: object) -> list[str] | None:
    if raw is None:
        return None
    return [str(name) for name in _expect_list(raw, label="Optional string list")]


def _coerce_runtime_value(value: object) -> bool | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return _coerce_int(value, label="Runtime value")


def _load_string_map(raw: object, *, label: str) -> dict[str, str]:
    return {str(name): str(text) for name, text in _expect_mapping(raw, label=label).items()}


def _load_algorithm_configs(
    raw: Mapping[str, object],
    *,
    shared_fragments: dict[str, str],
) -> dict[str, AlgorithmPromptConfig]:
    algorithms: dict[str, AlgorithmPromptConfig] = {}
    algorithm_mapping = _expect_mapping(raw["algorithms"], label="algorithms")
    for algorithm_name, algorithm_payload in algorithm_mapping.items():
        algorithm_config = _load_algorithm_config(
            algorithm_name=str(algorithm_name),
            algorithm_payload=algorithm_payload,
            shared_fragments=shared_fragments,
        )
        _validate_algorithm_fragments(algorithm_config)
        algorithms[algorithm_config.name] = algorithm_config
    return algorithms


def _load_algorithm_config(
    *,
    algorithm_name: str,
    algorithm_payload: object,
    shared_fragments: dict[str, str],
) -> AlgorithmPromptConfig:
    payload = _expect_mapping(algorithm_payload, label=f"Algorithm payload for {algorithm_name}")
    fragment_definitions = dict(shared_fragments)
    fragment_definitions.update(
        _load_string_map(
            payload.get("fragment_definitions", {}),
            label=f"fragment_definitions for {algorithm_name}",
        )
    )
    factors_payload = _expect_mapping(
        payload.get("factors", {}),
        label=f"factors for {algorithm_name}",
    )
    factors = {
        str(name): _load_factor_config(value)
        for name, value in factors_payload.items()
    }
    base_fragments = _expect_list(
        payload.get("base_fragments", []),
        label=f"base_fragments for {algorithm_name}",
    )
    return AlgorithmPromptConfig(
        name=algorithm_name,
        base_fragments=[str(name) for name in base_fragments],
        factors=factors,
        fragment_definitions=fragment_definitions,
        prompt_templates=_load_string_map(
            payload.get("prompt_templates", {}),
            label=f"prompt_templates for {algorithm_name}",
        ),
        pair_names=_load_optional_string_list(payload.get("pair_names")),
    )


def _load_run_config(raw: Mapping[str, object]) -> RunConfig:
    run = _expect_mapping(raw["run"], label="run")
    return RunConfig(
        provider=str(run["provider"]),
        output_root=str(run["output_root"]),
        replications=int(run["replications"]),
    )


def _load_runtime_config(raw: Mapping[str, object]) -> RuntimeConfig:
    runtime = _expect_mapping(raw["runtime"], label="runtime")
    return RuntimeConfig(
        seed=int(runtime["seed"]),
        temperature=float(runtime["temperature"]),
        quantization=str(runtime["quantization"]),
        device_policy=str(runtime["device_policy"]),
        context_policy=dict(_expect_mapping(runtime["context_policy"], label="context_policy")),
        max_new_tokens_by_schema={
            str(name): int(value)
            for name, value in _expect_mapping(
                runtime["max_new_tokens_by_schema"],
                label="max_new_tokens_by_schema",
            ).items()
        },
        thinking_mode_by_model={
            str(name): str(value)
            for name, value in _expect_mapping(
                runtime.get("thinking_mode_by_model", {}),
                label="thinking_mode_by_model",
            ).items()
        },
    )


def _load_models_config(raw: Mapping[str, object]) -> ModelsConfig:
    models = _expect_mapping(raw["models"], label="models")
    chat_models = _expect_list(models["chat_models"], label="chat_models")
    return ModelsConfig(
        chat_models=[str(model) for model in chat_models],
        embedding_model=str(models["embedding_model"]),
    )


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
    from llm_conceptual_modeling.hf_batch.planning import plan_paper_batch

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
