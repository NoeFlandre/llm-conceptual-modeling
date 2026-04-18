from __future__ import annotations

from itertools import product
from typing import Any, Callable

from llm_conceptual_modeling.common.graph_data import load_default_graph, load_graph_source
from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    RuntimeProfile,
    build_default_decoding_grid,
    supports_decoding_config,
    supports_explicit_thinking_disable,
)
from llm_conceptual_modeling.hf_batch.prompts import build_prompt_bundle
from llm_conceptual_modeling.hf_batch.types import HFRunSpec
from llm_conceptual_modeling.hf_batch.utils import condition_label, derive_run_seed


def plan_paper_batch(
    *,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    config: Any | None = None,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None = None,
) -> list[HFRunSpec]:
    return plan_paper_batch_specs(
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        algorithms=algorithms,
        config=config,
        runtime_profile_provider=runtime_profile_provider,
    )


def plan_paper_batch_specs(
    *,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None,
    config: Any | None,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None,
) -> list[HFRunSpec]:
    if config is not None:
        return _plan_paper_batch_from_config(
            config=config,
            algorithms=algorithms,
            runtime_profile_provider=runtime_profile_provider,
        )
    profile_provider = runtime_profile_provider or default_runtime_profile_provider
    selected_algorithms = set(algorithms or ("algo1", "algo2", "algo3"))
    specs: list[HFRunSpec] = []
    for model in models:
        runtime_profile = profile_provider(model)
        for decoding in build_default_decoding_grid():
            if not supports_decoding_config(model=model, decoding_config=decoding):
                continue
            for replication in range(replications):
                if "algo1" in selected_algorithms:
                    specs.extend(
                        _plan_algo1_runs(
                            model=model,
                            embedding_model=embedding_model,
                            decoding=decoding,
                            replication=replication,
                            runtime_profile=runtime_profile,
                        )
                    )
                if "algo2" in selected_algorithms:
                    specs.extend(
                        _plan_algo2_runs(
                            model=model,
                            embedding_model=embedding_model,
                            decoding=decoding,
                            replication=replication,
                            runtime_profile=runtime_profile,
                        )
                    )
                if "algo3" in selected_algorithms:
                    specs.extend(
                        _plan_algo3_runs(
                            model=model,
                            embedding_model=embedding_model,
                            decoding=decoding,
                            replication=replication,
                            runtime_profile=runtime_profile,
                        )
                    )
    return specs


def default_runtime_profile_provider(model: str) -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=supports_explicit_thinking_disable(model),
        context_limit=None,
    )


def _plan_algo1_runs(
    *,
    model: str,
    embedding_model: str,
    decoding: DecodingConfig,
    replication: int,
    runtime_profile: RuntimeProfile,
) -> list[HFRunSpec]:
    sg1, sg2, sg3, graph = load_default_graph()
    pair_lookup = {
        "sg1_sg2": (sg1, sg2),
        "sg2_sg3": (sg2, sg3),
        "sg3_sg1": (sg3, sg1),
    }
    specs: list[HFRunSpec] = []
    for pair_name, (subgraph1, subgraph2) in pair_lookup.items():
        for levels in product((-1, 1), repeat=5):
            explanation, example, counterexample, array_level, adjacency_level = levels
            prompt_factors = {
                "use_adjacency_notation": adjacency_level == 1,
                "use_array_representation": array_level == 1,
                "include_explanation": explanation == 1,
                "include_example": example == 1,
                "include_counterexample": counterexample == 1,
            }
            condition_bits = "".join("1" if level == 1 else "0" for level in levels)
            specs.append(
                HFRunSpec(
                    algorithm="algo1",
                    model=model,
                    embedding_model=embedding_model,
                    decoding=decoding,
                    replication=replication,
                    pair_name=pair_name,
                    condition_bits=condition_bits,
                    condition_label=condition_label(decoding),
                    prompt_factors=prompt_factors,
                    raw_context={
                        "pair_name": pair_name,
                        "Repetition": replication,
                        "Explanation": explanation,
                        "Example": example,
                        "Counterexample": counterexample,
                        "Array/List(1/-1)": array_level,
                        "Tag/Adjacency(1/-1)": adjacency_level,
                        "model": model,
                        "embedding_model": embedding_model,
                        "provider": "hf-transformers",
                        "decoding_algorithm": decoding.algorithm,
                        "decoding_condition": condition_label(decoding),
                    },
                    input_payload={
                        "subgraph1": subgraph1,
                        "subgraph2": subgraph2,
                        "graph": graph,
                    },
                    runtime_profile=runtime_profile,
                )
            )
    return specs


def _plan_algo2_runs(
    *,
    model: str,
    embedding_model: str,
    decoding: DecodingConfig,
    replication: int,
    runtime_profile: RuntimeProfile,
) -> list[HFRunSpec]:
    sg1, sg2, sg3, graph = load_default_graph()
    pair_lookup = {
        "sg1_sg2": (sg1, sg2),
        "sg2_sg3": (sg2, sg3),
        "sg3_sg1": (sg3, sg1),
    }
    specs: list[HFRunSpec] = []
    for pair_name, (subgraph1, subgraph2) in pair_lookup.items():
        for levels in product((-1, 1), repeat=6):
            explanation, example, counterexample, array_level, adjacency_level, convergence = levels
            prompt_factors = {
                "use_adjacency_notation": adjacency_level == 1,
                "use_array_representation": array_level == 1,
                "include_explanation": explanation == 1,
                "include_example": example == 1,
                "include_counterexample": counterexample == 1,
                "use_relaxed_convergence": convergence == 1,
            }
            condition_bits = "".join("1" if level == 1 else "0" for level in levels)
            specs.append(
                HFRunSpec(
                    algorithm="algo2",
                    model=model,
                    embedding_model=embedding_model,
                    decoding=decoding,
                    replication=replication,
                    pair_name=pair_name,
                    condition_bits=condition_bits,
                    condition_label=condition_label(decoding),
                    prompt_factors=prompt_factors,
                    raw_context={
                        "pair_name": pair_name,
                        "Repetition": replication,
                        "Explanation": explanation,
                        "Example": example,
                        "Counterexample": counterexample,
                        "Array/List(1/-1)": array_level,
                        "Tag/Adjacency(1/-1)": adjacency_level,
                        "Convergence": convergence,
                        "model": model,
                        "embedding_model": embedding_model,
                        "provider": "hf-transformers",
                        "decoding_algorithm": decoding.algorithm,
                        "decoding_condition": condition_label(decoding),
                    },
                    input_payload={
                        "subgraph1": subgraph1,
                        "subgraph2": subgraph2,
                        "graph": graph,
                    },
                    runtime_profile=runtime_profile,
                )
            )
    return specs


def _plan_algo3_runs(
    *,
    model: str,
    embedding_model: str,
    decoding: DecodingConfig,
    replication: int,
    runtime_profile: RuntimeProfile,
) -> list[HFRunSpec]:
    sg1, sg2, sg3, graph = load_default_graph()
    pair_lookup = {
        "subgraph_1_to_subgraph_3": ("subgraph_1", "subgraph_3", sg1, sg3),
        "subgraph_2_to_subgraph_1": ("subgraph_2", "subgraph_1", sg2, sg1),
        "subgraph_2_to_subgraph_3": ("subgraph_2", "subgraph_3", sg2, sg3),
    }
    specs: list[HFRunSpec] = []
    for pair_name, (source_name, target_name, source_graph, target_graph) in pair_lookup.items():
        for example, counterexample, number_of_words, depth in product(
            (-1, 1),
            (-1, 1),
            (3, 5),
            (1, 2),
        ):
            condition_bits = "".join(
                [
                    "1" if example == 1 else "0",
                    "1" if counterexample == 1 else "0",
                    "1" if number_of_words == 5 else "0",
                    "1" if depth == 2 else "0",
                ]
            )
            specs.append(
                HFRunSpec(
                    algorithm="algo3",
                    model=model,
                    embedding_model=embedding_model,
                    decoding=decoding,
                    replication=replication,
                    pair_name=pair_name,
                    condition_bits=condition_bits,
                    condition_label=condition_label(decoding),
                    prompt_factors={
                        "include_example": example == 1,
                        "include_counterexample": counterexample == 1,
                        "child_count": number_of_words,
                        "max_depth": depth,
                    },
                    raw_context={
                        "pair_name": pair_name,
                        "Repetition": replication,
                        "Example": example,
                        "Counter-Example": counterexample,
                        "Number of Words": number_of_words,
                        "Depth": depth,
                        "Source Subgraph Name": source_name,
                        "Target Subgraph Name": target_name,
                        "model": model,
                        "embedding_model": embedding_model,
                        "provider": "hf-transformers",
                        "decoding_algorithm": decoding.algorithm,
                        "decoding_condition": condition_label(decoding),
                    },
                    input_payload={
                        "source_graph": source_graph,
                        "target_graph": target_graph,
                        "mother_graph": graph,
                    },
                    runtime_profile=runtime_profile,
                )
            )
    return specs


def _plan_paper_batch_from_config(
    *,
    config: Any,
    algorithms: tuple[str, ...] | None,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None,
) -> list[HFRunSpec]:
    profile_provider = runtime_profile_provider or default_runtime_profile_provider
    selected_algorithms = set(algorithms or tuple(config.algorithms.keys()))
    specs: list[HFRunSpec] = []
    for model in config.models.chat_models:
        runtime_profile = profile_provider(model)
        for decoding in config.decoding:
            if not supports_decoding_config(model=model, decoding_config=decoding):
                continue
            for graph_source in config.graph_sources:
                for replication in range(config.run.replications):
                    for algorithm_name in selected_algorithms:
                        if algorithm_name == "algo1":
                            specs.extend(
                                _plan_algo1_runs_from_config(
                                    config=config,
                                    model=model,
                                    decoding=decoding,
                                    graph_source=graph_source,
                                    replication=replication,
                                    runtime_profile=runtime_profile,
                                )
                            )
                        elif algorithm_name == "algo2":
                            specs.extend(
                                _plan_algo2_runs_from_config(
                                    config=config,
                                    model=model,
                                    decoding=decoding,
                                    graph_source=graph_source,
                                    replication=replication,
                                    runtime_profile=runtime_profile,
                                )
                            )
                        elif algorithm_name == "algo3":
                            specs.extend(
                                _plan_algo3_runs_from_config(
                                    config=config,
                                    model=model,
                                    decoding=decoding,
                                    graph_source=graph_source,
                                    replication=replication,
                                    runtime_profile=runtime_profile,
                                )
                            )
                        else:
                            raise ValueError(
                                f"Unsupported algorithm in config: {algorithm_name}"
                            )
    return specs


def _plan_algo1_runs_from_config(
    *,
    config: Any,
    model: str,
    decoding: DecodingConfig,
    graph_source: str,
    replication: int,
    runtime_profile: RuntimeProfile,
) -> list[HFRunSpec]:
    sg1, sg2, sg3, graph = load_graph_source(graph_source)
    pair_lookup = {"sg1_sg2": (sg1, sg2), "sg2_sg3": (sg2, sg3), "sg3_sg1": (sg3, sg1)}
    algorithm_config = config.algorithms["algo1"]
    specs: list[HFRunSpec] = []
    for pair_name in algorithm_config.pair_names or list(pair_lookup.keys()):
        subgraph1, subgraph2 = pair_lookup[pair_name]
        specs.extend(
            _build_configured_specs_for_pairs(
                config=config,
                algorithm_name="algo1",
                algorithm_config=algorithm_config,
                model=model,
                decoding=decoding,
                graph_source=graph_source,
                replication=replication,
                pair_name=pair_name,
                runtime_profile=runtime_profile,
                payload={"subgraph1": subgraph1, "subgraph2": subgraph2, "graph": graph},
            )
        )
    return specs


def _plan_algo2_runs_from_config(
    *,
    config: Any,
    model: str,
    decoding: DecodingConfig,
    graph_source: str,
    replication: int,
    runtime_profile: RuntimeProfile,
) -> list[HFRunSpec]:
    sg1, sg2, sg3, graph = load_graph_source(graph_source)
    pair_lookup = {"sg1_sg2": (sg1, sg2), "sg2_sg3": (sg2, sg3), "sg3_sg1": (sg3, sg1)}
    algorithm_config = config.algorithms["algo2"]
    specs: list[HFRunSpec] = []
    for pair_name in algorithm_config.pair_names or list(pair_lookup.keys()):
        subgraph1, subgraph2 = pair_lookup[pair_name]
        specs.extend(
            _build_configured_specs_for_pairs(
                config=config,
                algorithm_name="algo2",
                algorithm_config=algorithm_config,
                model=model,
                decoding=decoding,
                graph_source=graph_source,
                replication=replication,
                pair_name=pair_name,
                runtime_profile=runtime_profile,
                payload={"subgraph1": subgraph1, "subgraph2": subgraph2, "graph": graph},
            )
        )
    return specs


def _plan_algo3_runs_from_config(
    *,
    config: Any,
    model: str,
    decoding: DecodingConfig,
    graph_source: str,
    replication: int,
    runtime_profile: RuntimeProfile,
) -> list[HFRunSpec]:
    sg1, sg2, sg3, graph = load_graph_source(graph_source)
    pair_lookup = {
        "subgraph_1_to_subgraph_3": ("subgraph_1", "subgraph_3", sg1, sg3),
        "subgraph_2_to_subgraph_1": ("subgraph_2", "subgraph_1", sg2, sg1),
        "subgraph_2_to_subgraph_3": ("subgraph_2", "subgraph_3", sg2, sg3),
    }
    algorithm_config = config.algorithms["algo3"]
    specs: list[HFRunSpec] = []
    for pair_name in algorithm_config.pair_names or list(pair_lookup.keys()):
        source_name, target_name, source_graph, target_graph = pair_lookup[pair_name]
        specs.extend(
            _build_configured_specs_for_pairs(
                config=config,
                algorithm_name="algo3",
                algorithm_config=algorithm_config,
                model=model,
                decoding=decoding,
                graph_source=graph_source,
                replication=replication,
                pair_name=pair_name,
                runtime_profile=runtime_profile,
                payload={
                    "source_name": source_name,
                    "target_name": target_name,
                    "source_graph": source_graph,
                    "target_graph": target_graph,
                    "mother_graph": graph,
                },
            )
        )
    return specs


def _build_configured_specs_for_pairs(
    *,
    config: Any,
    algorithm_name: str,
    algorithm_config: Any,
    model: str,
    decoding: DecodingConfig,
    graph_source: str,
    replication: int,
    pair_name: str,
    runtime_profile: RuntimeProfile,
    payload: dict[str, object],
) -> list[HFRunSpec]:
    factor_names = list(algorithm_config.factors.keys())
    factor_levels = [
        tuple(algorithm_config.factors[factor_name].levels)
        for factor_name in factor_names
    ]
    specs: list[HFRunSpec] = []
    for levels in product(*factor_levels):
        active_high_factors = [
            factor_name
            for factor_name, level in zip(factor_names, levels, strict=True)
            if level == algorithm_config.factors[factor_name].levels[-1]
        ]
        runtime_fields = algorithm_config.resolve_runtime_fields(active_high_factors)
        prompt_bundle = build_prompt_bundle(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            active_high_factors=active_high_factors,
            prompt_factors=runtime_fields,
        )
        condition_bits = "".join(
            "1" if level == algorithm_config.factors[factor_name].levels[-1] else "0"
            for factor_name, level in zip(factor_names, levels, strict=True)
        )
        specs.append(
            HFRunSpec(
                algorithm=algorithm_name,
                model=model,
                embedding_model=config.models.embedding_model,
                decoding=decoding,
                graph_source=graph_source,
                replication=replication,
                pair_name=pair_name,
                condition_bits=condition_bits,
                condition_label=condition_label(decoding),
                prompt_factors=runtime_fields,
                raw_context=_build_raw_context(
                    algorithm_name=algorithm_name,
                    algorithm_config=algorithm_config,
                    pair_name=pair_name,
                    levels=levels,
                    replication=replication,
                    model=model,
                    embedding_model=config.models.embedding_model,
                    decoding=decoding,
                    graph_source=graph_source,
                    payload=payload,
                ),
                input_payload={
                    key: value for key, value in payload.items() if not key.endswith("_name")
                },
                runtime_profile=runtime_profile,
                prompt_bundle=prompt_bundle,
                max_new_tokens_by_schema=config.runtime.max_new_tokens_by_schema,
                context_policy=config.runtime.context_policy,
                base_seed=config.runtime.seed,
                seed=derive_run_seed(
                    base_seed=config.runtime.seed,
                    algorithm=algorithm_name,
                    model=model,
                    pair_name=pair_name,
                    condition_bits=condition_bits,
                    decoding=decoding,
                    replication=replication,
                    graph_source=graph_source,
                ),
            )
        )
    return specs


def _build_raw_context(
    *,
    algorithm_name: str,
    algorithm_config: Any,
    pair_name: str,
    levels: tuple[int, ...],
    replication: int,
    model: str,
    embedding_model: str,
    decoding: DecodingConfig,
    graph_source: str,
    payload: dict[str, object],
) -> dict[str, object]:
    context: dict[str, object] = {
        "pair_name": pair_name,
        "Repetition": replication,
        "graph_source": graph_source,
        "model": model,
        "embedding_model": embedding_model,
        "provider": "hf-transformers",
        "decoding_algorithm": decoding.algorithm,
        "decoding_condition": condition_label(decoding),
    }
    context.update(algorithm_config.fixed_columns)
    for factor_name, level in zip(algorithm_config.factors.keys(), levels, strict=True):
        context[algorithm_config.factors[factor_name].column] = level
    if algorithm_name == "algo3":
        context["Source Subgraph Name"] = payload["source_name"]
        context["Target Subgraph Name"] = payload["target_name"]
    return context


def select_run_spec(
    *,
    config: Any,
    algorithm: str,
    model: str,
    graph_source: str | None = None,
    pair_name: str,
    condition_bits: str,
    decoding: DecodingConfig,
    replication: int,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None = None,
) -> HFRunSpec:
    resolved_graph_source = _resolve_selected_graph_source(
        config=config,
        graph_source=graph_source,
    )
    specs = plan_paper_batch_specs(
        models=[model],
        embedding_model=config.models.embedding_model,
        replications=max(replication + 1, 1),
        algorithms=(algorithm,),
        config=config,
        runtime_profile_provider=runtime_profile_provider,
    )
    for spec in specs:
        if (
            spec.algorithm == algorithm
            and spec.model == model
            and spec.graph_source == resolved_graph_source
            and spec.pair_name == pair_name
            and spec.condition_bits == condition_bits
            and spec.replication == replication
            and spec.decoding == decoding
        ):
            return spec
    raise ValueError(
        "No configured run spec matches "
        f"algorithm={algorithm!r}, model={model!r}, graph_source={resolved_graph_source!r}, "
        f"pair_name={pair_name!r}, "
        f"condition_bits={condition_bits!r}, decoding={decoding!r}, "
        f"replication={replication!r}."
    )


def _resolve_selected_graph_source(*, config: Any, graph_source: str | None) -> str:
    configured_graph_sources = list(config.graph_sources)
    if graph_source is not None:
        return graph_source
    if len(configured_graph_sources) == 1:
        return configured_graph_sources[0]
    raise ValueError(
        "graph_source is required when smoke-selecting from a config with multiple graph_sources."
    )
