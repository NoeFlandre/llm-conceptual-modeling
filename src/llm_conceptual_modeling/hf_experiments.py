from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

from llm_conceptual_modeling.algo1.factorial import (
    run_factorial_analysis as run_algo1_factorial_analysis,
)
from llm_conceptual_modeling.algo1.method import execute_method1
from llm_conceptual_modeling.algo1.mistral import (
    Method1PromptConfig,
    build_cove_verifier,
    build_edge_generator,
)
from llm_conceptual_modeling.algo2.factorial import (
    run_factorial_analysis as run_algo2_factorial_analysis,
)
from llm_conceptual_modeling.algo2.method import execute_method2
from llm_conceptual_modeling.algo2.mistral import (
    Method2PromptConfig,
    build_edge_suggester,
    build_label_proposer,
)
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as evaluate_algo3_results
from llm_conceptual_modeling.algo3.factorial import (
    run_factorial_analysis as run_algo3_factorial_analysis,
)
from llm_conceptual_modeling.algo3.method import build_tree_expander, execute_method3
from llm_conceptual_modeling.algo3.mistral import Method3PromptConfig, build_child_proposer
from llm_conceptual_modeling.analysis.replication_budget import write_replication_budget_analysis
from llm_conceptual_modeling.analysis.stability import write_grouped_metric_stability
from llm_conceptual_modeling.analysis.variability import write_output_variability_analysis
from llm_conceptual_modeling.common.evaluation_core import evaluate_connection_results_file
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus, load_default_graph
from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    HFTransformersRuntimeFactory,
    RuntimeProfile,
    build_default_decoding_grid,
    build_runtime_factory,
)

Edge = tuple[str, str]
RuntimeResult = dict[str, Any]
RuntimeFactory = Callable[["HFRunSpec"], RuntimeResult]


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

    @property
    def run_name(self) -> str:
        return (
            f"{self.algorithm}_{self.pair_name}_rep"
            f"{self.replication:02d}_cond{self.condition_bits}"
        )


def plan_paper_batch(
    *,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None = None,
) -> list[HFRunSpec]:
    profile_provider = runtime_profile_provider or _default_runtime_profile_provider
    selected_algorithms = set(algorithms or ("algo1", "algo2", "algo3"))
    specs: list[HFRunSpec] = []
    for model in models:
        runtime_profile = profile_provider(model)
        for decoding in build_default_decoding_grid():
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


def run_paper_batch(
    *,
    output_root: str | Path,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    runtime_factory: RuntimeFactory | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    hf_runtime = None if runtime_factory is not None else build_runtime_factory(
        hf_token=_resolve_hf_token()
    )
    if dry_run:
        profile_provider = None
    else:
        profile_provider = hf_runtime.profile_for_chat_model if hf_runtime else None
    planned_specs = plan_paper_batch(
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        algorithms=algorithms,
        runtime_profile_provider=profile_provider,
    )
    if runtime_factory is None:
        if hf_runtime is None:
            raise ValueError("Missing HF runtime for non-dry local execution.")
        runtime_factory = _runtime_factory_from_hf_runtime(hf_runtime)

    summary_rows: list[dict[str, object]] = []

    for spec in planned_specs:
        run_dir = (
            output_root_path
            / "runs"
            / spec.algorithm
            / _slugify_model(spec.model)
            / spec.condition_label
            / spec.pair_name
            / spec.condition_bits
            / f"rep_{spec.replication:02d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        raw_row_path = run_dir / "raw_row.json"

        if resume and summary_path.exists() and raw_row_path.exists():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append(cached)
            continue

        _write_json(run_dir / "manifest.json", _manifest_for_spec(spec))
        _write_json(run_dir / "state.json", {"status": "running"})

        try:
            runtime_result = _execute_run(
                spec=spec,
                runtime_factory=runtime_factory,
                dry_run=dry_run,
            )
        except Exception as error:
            _write_json(
                run_dir / "error.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "status": "failed",
                },
            )
            _write_json(run_dir / "state.json", {"status": "failed"})
            raise

        raw_row = runtime_result["raw_row"]
        _write_json(raw_row_path, raw_row)
        _write_json(run_dir / "state.json", {"status": "finished"})
        _write_json(run_dir / "runtime.json", runtime_result["runtime"])
        _write_text(run_dir / "raw_response.json", runtime_result["raw_response"])

        summary = {
            "algorithm": spec.algorithm,
            "model": spec.model,
            "embedding_model": spec.embedding_model,
            "decoding_algorithm": spec.decoding.algorithm,
            "condition_label": spec.condition_label,
            "pair_name": spec.pair_name,
            "condition_bits": spec.condition_bits,
            "replication": spec.replication,
            "status": "finished",
            "thinking_mode_supported": runtime_result["runtime"].get(
                "thinking_mode_supported", False
            ),
            "raw_row_path": str(raw_row_path),
        }
        _write_json(summary_path, summary)
        summary_rows.append(summary)

    summary_frame = pd.DataFrame.from_records(summary_rows)
    summary_frame.to_csv(output_root_path / "batch_summary.csv", index=False)
    if dry_run:
        return

    _write_aggregated_outputs(output_root_path, summary_frame)


def _execute_run(
    *,
    spec: HFRunSpec,
    runtime_factory: RuntimeFactory,
    dry_run: bool,
) -> RuntimeResult:
    if dry_run:
        return {
            "raw_row": dict(spec.raw_context),
            "runtime": {
                "thinking_mode_supported": spec.runtime_profile.supports_thinking_toggle,
                "device": spec.runtime_profile.device,
                "dtype": spec.runtime_profile.dtype,
                "quantization": spec.runtime_profile.quantization,
            },
            "raw_response": "[]",
        }
    return runtime_factory(spec)


def _runtime_factory_from_hf_runtime(hf_runtime: HFTransformersRuntimeFactory) -> RuntimeFactory:
    def runtime(spec: HFRunSpec) -> RuntimeResult:
        if spec.algorithm == "algo1":
            return _run_algo1(spec, hf_runtime=hf_runtime)
        if spec.algorithm == "algo2":
            return _run_algo2(spec, hf_runtime=hf_runtime)
        if spec.algorithm == "algo3":
            return _run_algo3(spec, hf_runtime=hf_runtime)
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    return runtime


def _run_algo1(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
) -> RuntimeResult:
    subgraph1 = _coerce_edges(spec.input_payload["subgraph1"])
    subgraph2 = _coerce_edges(spec.input_payload["subgraph2"])
    graph = _coerce_edges(spec.input_payload["graph"])
    prompt_config = _algo1_prompt_config(spec.prompt_factors)
    chat_client = hf_runtime.build_chat_client(model=spec.model, decoding_config=spec.decoding)
    recorder = _RecordingChatClient(chat_client)
    result = execute_method1(
        subgraph1=subgraph1,
        subgraph2=subgraph2,
        generate_edges=build_edge_generator(recorder, prompt_config=prompt_config),
        verify_edges=build_cove_verifier(recorder),
    )
    raw_row = {
        **spec.raw_context,
        "Result": repr(result.verified_edges),
        "graph": repr(graph),
        "subgraph1": repr(subgraph1),
        "subgraph2": repr(subgraph2),
    }
    return {
        "raw_row": raw_row,
        "runtime": _runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
    }


def _run_algo2(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
) -> RuntimeResult:
    subgraph1 = _coerce_edges(spec.input_payload["subgraph1"])
    subgraph2 = _coerce_edges(spec.input_payload["subgraph2"])
    graph = _coerce_edges(spec.input_payload["graph"])
    prompt_config = _algo2_prompt_config(spec.prompt_factors)
    chat_client = hf_runtime.build_chat_client(model=spec.model, decoding_config=spec.decoding)
    recorder = _RecordingChatClient(chat_client)
    embedding_client = hf_runtime.build_embedding_client(model=spec.embedding_model)
    source_labels = _collect_nodes(subgraph1)
    target_labels = _collect_nodes(subgraph2)
    seed_labels = source_labels + [label for label in target_labels if label not in source_labels]
    threshold = 0.02 if prompt_config.use_relaxed_convergence else 0.01
    thesaurus = load_algo2_thesaurus()
    result = execute_method2(
        seed_labels=seed_labels,
        existing_edges=list(subgraph1) + list(subgraph2),
        propose_labels=build_label_proposer(recorder, prompt_config=prompt_config),
        suggest_edges=build_edge_suggester(recorder, prompt_config=prompt_config),
        embedding_client=embedding_client,
        convergence_threshold=threshold,
        thesaurus=thesaurus,
    )
    raw_row = {
        **spec.raw_context,
        "Result": repr(result.normalized_edges),
        "graph": repr(graph),
        "subgraph1": repr(subgraph1),
        "subgraph2": repr(subgraph2),
    }
    return {
        "raw_row": raw_row,
        "runtime": _runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
    }


def _run_algo3(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
) -> RuntimeResult:
    source_graph = _coerce_edges(spec.input_payload["source_graph"])
    target_graph = _coerce_edges(spec.input_payload["target_graph"])
    mother_graph = _coerce_edges(spec.input_payload["mother_graph"])
    source_labels = _collect_nodes(source_graph)
    target_labels = _collect_nodes(target_graph)
    prompt_factors = dict(spec.prompt_factors)
    child_count = int(prompt_factors.pop("child_count"))
    max_depth = int(prompt_factors.pop("max_depth"))
    prompt_config = _algo3_prompt_config(prompt_factors)
    chat_client = hf_runtime.build_chat_client(model=spec.model, decoding_config=spec.decoding)
    recorder = _RecordingChatClient(chat_client)
    result = execute_method3(
        source_labels=source_labels,
        target_labels=target_labels,
        child_count=child_count,
        max_depth=max_depth,
        expand_tree=build_tree_expander(
            build_child_proposer(recorder, prompt_config=prompt_config)
        ),
    )
    result_edges = [(node.parent_label, node.label) for node in result.expanded_nodes]
    raw_row = {
        **spec.raw_context,
        "Source Graph": repr(source_graph),
        "Target Graph": repr(target_graph),
        "Mother Graph": repr(mother_graph),
        "Results": repr(result_edges),
        "Recall": 0.0,
    }
    return {
        "raw_row": raw_row,
        "runtime": _runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
    }


def _write_aggregated_outputs(output_root: Path, summary_frame: pd.DataFrame) -> None:
    aggregated_root = output_root / "aggregated"
    for group_key, group_frame in summary_frame.groupby(
        ["algorithm", "model", "condition_label"],
        dropna=False,
    ):
        algorithm, model, condition_label = cast(tuple[object, object, object], group_key)
        combo_root = (
            aggregated_root / str(algorithm) / _slugify_model(str(model)) / str(condition_label)
        )
        combo_root.mkdir(parents=True, exist_ok=True)
        raw_rows = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in group_frame["raw_row_path"].tolist()
        ]
        raw_frame = pd.DataFrame.from_records(raw_rows)
        raw_path = combo_root / "raw.csv"
        raw_frame.to_csv(raw_path, index=False)

        evaluated_path = combo_root / "evaluated.csv"
        factorial_path = combo_root / "factorial.csv"
        variability_path = combo_root / "output_variability.csv"
        stability_path = combo_root / "condition_stability.csv"
        strict_budget_path = combo_root / "replication_budget_strict.csv"
        relaxed_budget_path = combo_root / "replication_budget_relaxed.csv"

        if algorithm in {"algo1", "algo2"}:
            evaluate_connection_results_file(raw_path, evaluated_path)
            if algorithm == "algo1":
                run_algo1_factorial_analysis([evaluated_path], factorial_path)
                write_grouped_metric_stability(
                    [evaluated_path],
                    stability_path,
                    group_by=[
                        "pair_name",
                        "Explanation",
                        "Example",
                        "Counterexample",
                        "Array/List(1/-1)",
                        "Tag/Adjacency(1/-1)",
                    ],
                    metrics=["accuracy", "recall", "precision"],
                )
            else:
                run_algo2_factorial_analysis([evaluated_path], factorial_path)
                write_grouped_metric_stability(
                    [evaluated_path],
                    stability_path,
                    group_by=[
                        "pair_name",
                        "Convergence",
                        "Explanation",
                        "Example",
                        "Counterexample",
                        "Array/List(1/-1)",
                        "Tag/Adjacency(1/-1)",
                    ],
                    metrics=["accuracy", "recall", "precision"],
                )
            write_output_variability_analysis(
                [raw_path],
                variability_path,
                group_by=[
                    "pair_name",
                    "Explanation",
                    "Example",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                ]
                + (["Convergence"] if algorithm == "algo2" else []),
                result_column="Result",
            )
        else:
            evaluate_algo3_results(raw_path, evaluated_path)
            run_algo3_factorial_analysis(evaluated_path, factorial_path)
            write_grouped_metric_stability(
                [evaluated_path],
                stability_path,
                group_by=[
                    "pair_name",
                    "Depth",
                    "Number of Words",
                    "Example",
                    "Counter-Example",
                ],
                metrics=["Recall"],
            )
            write_output_variability_analysis(
                [raw_path],
                variability_path,
                group_by=[
                    "pair_name",
                    "Depth",
                    "Number of Words",
                    "Example",
                    "Counter-Example",
                ],
                result_column="Results",
            )

        write_replication_budget_analysis(
            [stability_path],
            strict_budget_path,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        write_replication_budget_analysis(
            [stability_path],
            relaxed_budget_path,
            relative_half_width_target=0.10,
            z_score=1.645,
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
                    condition_label=_condition_label(decoding),
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
                        "decoding_condition": _condition_label(decoding),
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
                    condition_label=_condition_label(decoding),
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
                        "decoding_condition": _condition_label(decoding),
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
                    condition_label=_condition_label(decoding),
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
                        "decoding_condition": _condition_label(decoding),
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


def _manifest_for_spec(spec: HFRunSpec) -> dict[str, object]:
    return {
        "provider": "hf-transformers",
        "algorithm": spec.algorithm,
        "model": spec.model,
        "embedding_model": spec.embedding_model,
        "temperature": 0.0,
        "decoding": asdict(spec.decoding),
        "replication": spec.replication,
        "pair_name": spec.pair_name,
        "condition_bits": spec.condition_bits,
        "condition_label": spec.condition_label,
        "prompt_factors": spec.prompt_factors,
        "runtime": _runtime_details(spec.runtime_profile),
        "input_payload": _stringify_payload(spec.input_payload),
    }


def _stringify_payload(payload: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            serialized[key] = repr(value)
        else:
            serialized[key] = value
    return serialized


def _runtime_details(profile: RuntimeProfile) -> dict[str, object]:
    return {
        "device": profile.device,
        "dtype": profile.dtype,
        "quantization": profile.quantization,
        "thinking_mode_supported": profile.supports_thinking_toggle,
        "context_limit": profile.context_limit,
    }


def _resolve_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _default_runtime_profile_provider(_model: str) -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=False,
        context_limit=None,
    )


def _condition_label(decoding: DecodingConfig) -> str:
    if decoding.algorithm == "greedy":
        return "greedy"
    if decoding.algorithm == "beam":
        return f"beam_num_beams_{decoding.num_beams}"
    return f"contrastive_penalty_alpha_{decoding.penalty_alpha}"


def _slugify_model(model: str) -> str:
    return model.replace("/", "__")


def _coerce_edges(raw_edges: object) -> list[Edge]:
    if not isinstance(raw_edges, list):
        raise ValueError(f"Expected list of edges, got {type(raw_edges)!r}")
    edges: list[Edge] = []
    for edge in raw_edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError(f"Invalid edge payload: {edge!r}")
        edges.append((str(edge[0]), str(edge[1])))
    return edges


def _collect_nodes(edges: list[Edge]) -> list[str]:
    ordered_nodes: list[str] = []
    seen_nodes: set[str] = set()
    for left, right in edges:
        if left not in seen_nodes:
            ordered_nodes.append(left)
            seen_nodes.add(left)
        if right not in seen_nodes:
            ordered_nodes.append(right)
            seen_nodes.add(right)
    return ordered_nodes


def _algo1_prompt_config(prompt_factors: dict[str, bool | int]) -> Method1PromptConfig:
    return Method1PromptConfig(
        use_adjacency_notation=bool(prompt_factors["use_adjacency_notation"]),
        use_array_representation=bool(prompt_factors["use_array_representation"]),
        include_explanation=bool(prompt_factors["include_explanation"]),
        include_example=bool(prompt_factors["include_example"]),
        include_counterexample=bool(prompt_factors["include_counterexample"]),
    )


def _algo2_prompt_config(prompt_factors: dict[str, bool | int]) -> Method2PromptConfig:
    return Method2PromptConfig(
        use_adjacency_notation=bool(prompt_factors["use_adjacency_notation"]),
        use_array_representation=bool(prompt_factors["use_array_representation"]),
        include_explanation=bool(prompt_factors["include_explanation"]),
        include_example=bool(prompt_factors["include_example"]),
        include_counterexample=bool(prompt_factors["include_counterexample"]),
        use_relaxed_convergence=bool(prompt_factors["use_relaxed_convergence"]),
    )


def _algo3_prompt_config(prompt_factors: dict[str, bool | int]) -> Method3PromptConfig:
    return Method3PromptConfig(
        include_example=bool(prompt_factors["include_example"]),
        include_counterexample=bool(prompt_factors["include_counterexample"]),
    )


class _RecordingChatClient:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.records: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        response = self._inner.complete_json(
            prompt=prompt,
            schema_name=schema_name,
            schema=schema,
        )
        self.records.append(
            {
                "schema_name": schema_name,
                "prompt": prompt,
                "response": response,
            }
        )
        return response


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")
