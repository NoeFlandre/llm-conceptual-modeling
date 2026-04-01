from llm_conceptual_modeling.common.graph_data import (
    algo2_thesaurus_json_path,
    categories_csv_path,
    edges_csv_path,
    load_algo2_thesaurus,
)
from llm_conceptual_modeling.common.types import GenerationManifest


def build_generation_manifest(*, fixture_only: bool) -> dict[str, object]:
    thesaurus = load_algo2_thesaurus()
    synonym_map = thesaurus["synonyms"]
    antonym_map = thesaurus["antonyms"]
    method_contract: dict[str, object] = {
        "method_name": "Grow candidate links through creative thinking and word embeddings",
        "phases": [
            "label_expansion",
            "edge_suggestion",
            "thesaurus_normalization",
            "chain_of_verification",
        ],
        "embedding_model": "mistral-embed-2312",
        "embedding_provider": "mistral",
        "paper_embedding_model": "text-embedding-3-large",
        "paper_embedding_provider": "openrouter",
        "convergence_rule": "absolute_cosine_similarity_delta <= threshold",
        "convergence_threshold_levels": [0.01, 0.02],
        "uses_domain_thesaurus": True,
        "uses_chain_of_verification": True,
        "thesaurus_path": str(algo2_thesaurus_json_path()),
        "synonym_entry_count": len(synonym_map),
        "antonym_entry_count": len(antonym_map),
    }
    return GenerationManifest(
        algorithm="algo2",
        mode="offline-manifest",
        implemented=True,
        requires_live_llm=True,
        fixture_only=fixture_only,
        next_step="provide_fixture_dataset" if fixture_only else "implement_provider_adapter",
        input_data={
            "categories_csv": str(categories_csv_path()),
            "edges_csv": str(edges_csv_path()),
            "thesaurus_json": str(algo2_thesaurus_json_path()),
        },
        condition_count=64,
        replications=5,
        subgraph_pairs=["sg1_sg2", "sg2_sg3", "sg3_sg1"],
        prompt_preview=(
            "You are a helpful assistant who understands Knowledge Maps. Your task is to "
            "recommend 5 more nodes in relation to those already in the two knowledge maps."
        ),
        method_contract=method_contract,
    ).to_dict()
