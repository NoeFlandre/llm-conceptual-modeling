import logging
from itertools import product
from pathlib import Path

from llm_conceptual_modeling.algo2.embeddings import EmbeddingClient
from llm_conceptual_modeling.algo2.mistral import (
    ChatCompletionClient,
    Method2PromptConfig,
)
from llm_conceptual_modeling.algo2.probe import Algo2ProbeSpec, run_algo2_probe
from llm_conceptual_modeling.common.graph_data import load_default_graph

logger = logging.getLogger(__name__)


def build_algo2_experiment_specs(
    *,
    pair_name: str,
    output_root: Path,
    replications: int = 5,
    resume: bool = False,
) -> list[Algo2ProbeSpec]:
    seed_labels, subgraph1, subgraph2 = _load_pair_labels(pair_name)
    condition_bits = list(product([0, 1], repeat=5))
    experiment_specs: list[Algo2ProbeSpec] = []

    for repetition_index in range(replications):
        for condition_bit_tuple in condition_bits:
            condition_name = "".join(str(bit) for bit in condition_bit_tuple)
            run_name = f"algo2_{pair_name}_rep{repetition_index}_cond{condition_name}"
            prompt_config = _build_prompt_config(condition_bit_tuple)
            output_dir = (
                output_root
                / "algo2"
                / pair_name
                / f"rep{repetition_index}_cond{condition_name}"
            )
            experiment_spec = Algo2ProbeSpec(
                run_name=run_name,
                model="",
                seed_labels=seed_labels,
                subgraph1=subgraph1,
                subgraph2=subgraph2,
                prompt_config=prompt_config,
                convergence_threshold=0.01,
                output_dir=output_dir,
                resume=resume,
            )
            experiment_specs.append(experiment_spec)

    return experiment_specs


def run_algo2_experiment(
    *,
    specs: list[Algo2ProbeSpec],
    chat_client: ChatCompletionClient,
    embedding_client: EmbeddingClient,
) -> list[dict[str, object]]:
    summary_records: list[dict[str, object]] = []

    for spec in specs:
        try:
            summary_record = run_algo2_probe(
                spec=spec,
                chat_client=chat_client,
                embedding_client=embedding_client,
            )
        except Exception:
            logger.exception("Method 2 experiment probe failed: run_name=%s", spec.run_name)
            continue
        summary_records.append(summary_record)

    return summary_records


def _load_pair_labels(
    pair_name: str,
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    sg1, sg2, sg3, _ = load_default_graph()
    pair_map = {
        "sg1_sg2": (sg1, sg2),
        "sg2_sg3": (sg2, sg3),
        "sg3_sg1": (sg3, sg1),
    }
    if pair_name not in pair_map:
        msg = f"Unsupported Method 2 pair name: {pair_name}"
        raise ValueError(msg)

    subgraph1, subgraph2 = pair_map[pair_name]
    seed_labels = _collect_ordered_nodes(subgraph1) + _collect_ordered_nodes(subgraph2)
    return seed_labels, subgraph1, subgraph2


def _build_prompt_config(
    condition_bit_tuple: tuple[int, int, int, int, int],
) -> Method2PromptConfig:
    adjacency_bit, array_bit, explanation_bit, example_bit, counterexample_bit = (
        condition_bit_tuple
    )
    prompt_config = Method2PromptConfig(
        use_adjacency_notation=bool(adjacency_bit),
        use_array_representation=bool(array_bit),
        include_explanation=bool(explanation_bit),
        include_example=bool(example_bit),
        include_counterexample=bool(counterexample_bit),
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
