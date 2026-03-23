import logging
from itertools import product
from pathlib import Path

from llm_conceptual_modeling.algo3.mistral import (
    ChatCompletionClient,
    Method3PromptConfig,
    build_tree_expansion_prompt,
)
from llm_conceptual_modeling.algo3.probe import Algo3ProbeSpec, run_algo3_probe
from llm_conceptual_modeling.common.graph_data import load_default_graph
from llm_conceptual_modeling.experiment_manifest import write_manifest

logger = logging.getLogger(__name__)


def build_algo3_experiment_specs(
    *,
    pair_name: str,
    model: str,
    output_root: Path,
    replications: int = 5,
    resume: bool = False,
) -> list[Algo3ProbeSpec]:
    source_labels, target_labels = _load_pair_labels(pair_name)
    condition_bits = list(product([0, 1], repeat=4))
    experiment_specs: list[Algo3ProbeSpec] = []

    for repetition_index in range(replications):
        for condition_bit_tuple in condition_bits:
            condition_name = "".join(str(bit) for bit in condition_bit_tuple)
            run_name = f"algo3_{pair_name}_rep{repetition_index}_cond{condition_name}"
            prompt_config = _build_prompt_config(condition_bit_tuple)
            child_count = 3 if condition_bit_tuple[2] == 0 else 5
            max_depth = 1 if condition_bit_tuple[3] == 0 else 2
            output_dir = (
                output_root / "algo3" / pair_name / f"rep{repetition_index}_cond{condition_name}"
            )
            experiment_spec = Algo3ProbeSpec(
                run_name=run_name,
                model=model,
                source_labels=source_labels,
                target_labels=target_labels,
                prompt_config=prompt_config,
                child_count=child_count,
                max_depth=max_depth,
                output_dir=output_dir,
                resume=resume,
            )
            write_manifest(
                spec=experiment_spec,
                algorithm="algo3",
                provider="mistral",
                temperature=0.0,
                top_p=None,
                max_tokens=None,
                full_prompt=build_tree_expansion_prompt(
                    source_labels=source_labels,
                    child_count=child_count,
                    prompt_config=prompt_config,
                ),
                pair_name=pair_name,
                condition_bits=condition_name,
                repetitions=replications,
                yaml_path=output_dir / "manifest.yaml",
            )
            experiment_specs.append(experiment_spec)

    return experiment_specs


def run_algo3_experiment(
    *,
    specs: list[Algo3ProbeSpec],
    chat_client: ChatCompletionClient,
) -> list[dict[str, object]]:
    summary_records: list[dict[str, object]] = []

    for spec in specs:
        try:
            summary_record = run_algo3_probe(
                spec=spec,
                chat_client=chat_client,
            )
        except Exception:
            logger.exception("Method 3 experiment probe failed: run_name=%s", spec.run_name)
            continue
        summary_records.append(summary_record)

    return summary_records


def _load_pair_labels(pair_name: str) -> tuple[list[str], list[str]]:
    sg1, sg2, sg3, _ = load_default_graph()
    pair_map = {
        "subgraph_1_to_subgraph_3": (sg1, sg3),
        "subgraph_2_to_subgraph_1": (sg2, sg1),
        "subgraph_2_to_subgraph_3": (sg2, sg3),
    }
    if pair_name not in pair_map:
        msg = f"Unsupported Method 3 pair name: {pair_name}"
        raise ValueError(msg)

    source_edges, target_edges = pair_map[pair_name]
    source_labels = _collect_ordered_nodes(source_edges)
    target_labels = _collect_ordered_nodes(target_edges)
    return source_labels, target_labels


def _build_prompt_config(condition_bit_tuple: tuple[int, int, int, int]) -> Method3PromptConfig:
    example_flag, counterexample_flag, _, _ = condition_bit_tuple
    prompt_config = Method3PromptConfig(
        include_example=bool(example_flag),
        include_counterexample=bool(counterexample_flag),
    )
    return prompt_config


def _collect_ordered_nodes(edges: list[tuple[str, str]]) -> list[str]:
    ordered_nodes: list[str] = []
    seen_nodes: set[str] = set()

    for source, target in edges:
        for node in (source, target):
            if node in seen_nodes:
                continue
            ordered_nodes.append(node)
            seen_nodes.add(node)

    return ordered_nodes
