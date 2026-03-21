from itertools import product

import pandas as pd

from llm_conceptual_modeling.common.baseline import propose_direct_cross_subgraph_edges
from llm_conceptual_modeling.common.graph_data import load_default_graph
from llm_conceptual_modeling.common.types import PathLike


def write_baseline_results_file(pair_name: str, output_csv_path: PathLike) -> None:
    sg1, sg2, sg3, graph = load_default_graph()
    pair_lookup = {
        "sg1_sg2": (sg1, sg2),
        "sg2_sg3": (sg2, sg3),
        "sg3_sg1": (sg3, sg1),
    }
    if pair_name not in pair_lookup:
        raise ValueError(f"Unsupported pair: {pair_name}")

    subgraph1, subgraph2 = pair_lookup[pair_name]
    baseline_result = propose_direct_cross_subgraph_edges(graph, subgraph1, subgraph2)

    rows: list[dict[str, object]] = []
    for repetition in range(5):
        for explanation, example, counterexample, array_list, tag_adjacency, convergence in product(
            (-1, 1), repeat=6
        ):
            rows.append(
                {
                    "Repetition": repetition,
                    "Result": repr(baseline_result),
                    "Explanation": explanation,
                    "Example": example,
                    "Counterexample": counterexample,
                    "Array/List(1/-1)": array_list,
                    "Tag/Adjacency(1/-1)": tag_adjacency,
                    "Convergence": convergence,
                    "subgraph1": repr(subgraph1),
                    "subgraph2": repr(subgraph2),
                    "graph": repr(graph),
                }
            )

    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
