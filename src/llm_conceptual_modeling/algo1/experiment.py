import logging
from itertools import product
from pathlib import Path

from llm_conceptual_modeling.algo1.mistral import (
    ChatCompletionClient,
    Method1PromptConfig,
    build_direct_edge_prompt,
)
from llm_conceptual_modeling.algo1.probe import Algo1ProbeSpec, run_algo1_probe
from llm_conceptual_modeling.common.graph_data import load_default_graph
from llm_conceptual_modeling.experiment_manifest import write_manifest

logger = logging.getLogger(__name__)


def build_algo1_experiment_specs(
    *,
    pair_name: str,
    model: str,
    output_root: Path,
    replications: int = 5,
    resume: bool = False,
) -> list[Algo1ProbeSpec]:
    pair_subgraphs = _load_pair_subgraphs(pair_name)
    subgraph1, subgraph2 = pair_subgraphs
    condition_bits = list(product([0, 1], repeat=5))
    experiment_specs: list[Algo1ProbeSpec] = []

    for repetition_index in range(replications):
        for condition_bit_tuple in condition_bits:
            condition_name = "".join(str(bit) for bit in condition_bit_tuple)
            run_name = f"algo1_{pair_name}_rep{repetition_index}_cond{condition_name}"
            prompt_config = _build_prompt_config(condition_bit_tuple)
            output_dir = (
                output_root / "algo1" / pair_name / f"rep{repetition_index}_cond{condition_name}"
            )
            experiment_spec = Algo1ProbeSpec(
                run_name=run_name,
                model=model,
                subgraph1=subgraph1,
                subgraph2=subgraph2,
                prompt_config=prompt_config,
                output_dir=output_dir,
                resume=resume,
            )
            write_manifest(
                spec=experiment_spec,
                algorithm="algo1",
                provider="mistral",
                temperature=0.0,
                top_p=None,
                max_tokens=None,
                full_prompt=build_direct_edge_prompt(
                    subgraph1=subgraph1,
                    subgraph2=subgraph2,
                    prompt_config=prompt_config,
                ),
                pair_name=pair_name,
                condition_bits=condition_name,
                repetitions=replications,
                yaml_path=output_dir / "manifest.yaml",
            )
            experiment_specs.append(experiment_spec)

    return experiment_specs


def run_algo1_experiment(
    *,
    specs: list[Algo1ProbeSpec],
    chat_client: ChatCompletionClient,
) -> list[dict[str, object]]:
    summary_records: list[dict[str, object]] = []

    for spec in specs:
        try:
            summary_record = run_algo1_probe(
                spec=spec,
                chat_client=chat_client,
            )
        except Exception:
            logger.exception("Method 1 experiment probe failed: run_name=%s", spec.run_name)
            continue
        summary_records.append(summary_record)

    return summary_records


def _load_pair_subgraphs(pair_name: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    sg1, sg2, sg3, _ = load_default_graph()
    pair_map = {
        "sg1_sg2": (sg1, sg2),
        "sg2_sg3": (sg2, sg3),
        "sg3_sg1": (sg3, sg1),
    }
    if pair_name not in pair_map:
        msg = f"Unsupported Method 1 pair name: {pair_name}"
        raise ValueError(msg)

    subgraphs = pair_map[pair_name]
    return subgraphs


def _build_prompt_config(
    condition_bit_tuple: tuple[int, int, int, int, int],
) -> Method1PromptConfig:
    (
        adjacency_flag,
        array_flag,
        explanation_flag,
        example_flag,
        counterexample_flag,
    ) = condition_bit_tuple
    prompt_config = Method1PromptConfig(
        use_adjacency_notation=bool(adjacency_flag),
        use_array_representation=bool(array_flag),
        include_explanation=bool(explanation_flag),
        include_example=bool(example_flag),
        include_counterexample=bool(counterexample_flag),
    )
    return prompt_config
