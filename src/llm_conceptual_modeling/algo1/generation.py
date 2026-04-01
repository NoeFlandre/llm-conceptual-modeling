from llm_conceptual_modeling.common.graph_data import categories_csv_path, edges_csv_path
from llm_conceptual_modeling.common.types import GenerationManifest


def build_generation_manifest(*, fixture_only: bool) -> dict[str, object]:
    method_contract: dict[str, object] = {
        "method_name": "Direct combination",
        "phases": [
            "edge_generation",
            "chain_of_verification",
        ],
        "uses_chain_of_verification": True,
        "verification_output": "Y/N list aligned to candidate pairs",
        "allows_new_nodes": True,
    }
    return GenerationManifest(
        algorithm="algo1",
        mode="offline-manifest",
        implemented=True,
        requires_live_llm=True,
        fixture_only=fixture_only,
        next_step="provide_fixture_dataset"
        if fixture_only
        else "provide_model_pair_and_output_root",
        input_data={
            "categories_csv": str(categories_csv_path()),
            "edges_csv": str(edges_csv_path()),
        },
        condition_count=32,
        replications=5,
        subgraph_pairs=["sg1_sg2", "sg2_sg3", "sg3_sg1"],
        prompt_preview=(
            "You will get two inputs: Knowledge map 1: ... Knowledge map 2: ... "
            "Your task is to recommend more links between the two maps."
        ),
        method_contract=method_contract,
    ).to_dict()
