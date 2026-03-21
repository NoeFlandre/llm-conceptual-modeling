from llm_conceptual_modeling.common.graph_data import CATEGORIES_CSV, EDGES_CSV
from llm_conceptual_modeling.common.types import GenerationManifest


def build_generation_manifest(*, fixture_only: bool) -> dict[str, object]:
    return GenerationManifest(
        algorithm="algo3",
        mode="offline-manifest",
        implemented=False,
        requires_live_llm=True,
        fixture_only=fixture_only,
        next_step="provide_fixture_dataset" if fixture_only else "implement_provider_adapter",
        input_data={
            "categories_csv": str(CATEGORIES_CSV),
            "edges_csv": str(EDGES_CSV),
        },
        condition_count=16,
        replications=5,
        subgraph_pairs=[
            "subgraph_1_to_subgraph_3",
            "subgraph_2_to_subgraph_1",
            "subgraph_2_to_subgraph_3",
        ],
        method_contract={
            "method_name": "Tree-based approach for indirect matches",
            "phases": [
                "source_tree_expansion",
                "target_match_check",
                "recursive_expansion",
            ],
            "child_count_levels": [3, 5],
            "depth_levels": [1, 2],
            "stops_descending_from_matched_words": True,
            "expansion_scope": "source_labels_only",
        },
        prompt_preview=(
            "Your task is to recommend related concept names for each concept in the input."
        ),
    ).to_dict()
