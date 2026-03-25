from itertools import product

import pandas as pd

from llm_conceptual_modeling.common.baseline import (
    propose_direct_cross_subgraph_edges,
    propose_random_cross_subgraph_edges,
)
from llm_conceptual_modeling.common.graph_data import load_default_graph
from llm_conceptual_modeling.common.types import PathLike


def write_baseline_results_file(
    pair_name: str,
    output_csv_path: PathLike,
    strategy: str = "random-uniform-subset",
) -> None:
    sg1, sg2, sg3, graph = load_default_graph()
    pair_lookup = {
        "subgraph_1_to_subgraph_3": ("subgraph_1", "subgraph_3", sg1, sg3),
        "subgraph_2_to_subgraph_1": ("subgraph_2", "subgraph_1", sg2, sg1),
        "subgraph_2_to_subgraph_3": ("subgraph_2", "subgraph_3", sg2, sg3),
    }
    if pair_name not in pair_lookup:
        raise ValueError(f"Unsupported pair: {pair_name}")

    source_name, target_name, source_graph, target_graph = pair_lookup[pair_name]
    if strategy == "direct-cross-graph":
        baseline_result = propose_direct_cross_subgraph_edges(graph, source_graph, target_graph)
    else:
        baseline_result = propose_random_cross_subgraph_edges(source_graph, target_graph)

    rows: list[dict[str, object]] = []
    for repetition in range(5):
        for example, counterexample, number_of_words, depth in product(
            (-1, 1), (-1, 1), (3, 5), (1, 2)
        ):
            rows.append(
                {
                    "Repetition": repetition,
                    "Example": example,
                    "Counter-Example": counterexample,
                    "Number of Words": number_of_words,
                    "Depth": depth,
                    "Source Subgraph Name": source_name,
                    "Target Subgraph Name": target_name,
                    "Source Graph": repr(source_graph),
                    "Target Graph": repr(target_graph),
                    "Mother Graph": repr(graph),
                    "Recall": 0.0,
                    "Results": repr(baseline_result),
                }
            )

    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
