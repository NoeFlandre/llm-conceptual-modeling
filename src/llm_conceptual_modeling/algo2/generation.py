from llm_conceptual_modeling.common.graph_data import CATEGORIES_CSV, EDGES_CSV


def build_generation_manifest(*, fixture_only: bool) -> dict[str, object]:
    return {
        "algorithm": "algo2",
        "mode": "offline-manifest",
        "implemented": False,
        "requires_live_llm": True,
        "fixture_only": fixture_only,
        "next_step": "provide_fixture_dataset" if fixture_only else "implement_provider_adapter",
        "input_data": {
            "categories_csv": str(CATEGORIES_CSV),
            "edges_csv": str(EDGES_CSV),
        },
        "condition_count": 64,
        "replications": 5,
        "subgraph_pairs": ["sg1_sg2", "sg2_sg3", "sg3_sg1"],
        "prompt_preview": (
            "Your task is to recommend 5 more nodes in relation to those already "
            "in the two knowledge maps."
        ),
    }
