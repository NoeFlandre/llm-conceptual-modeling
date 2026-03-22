from llm_conceptual_modeling.common.graph_data import (
    ALGO2_THESAURUS_JSON,
    CATEGORIES_CSV,
    EDGES_CSV,
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
        "convergence_rule": "absolute_cosine_similarity_delta <= threshold",
        "convergence_threshold": 0.01,
        "uses_domain_thesaurus": True,
        "uses_chain_of_verification": True,
        "thesaurus_path": str(ALGO2_THESAURUS_JSON),
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
            "categories_csv": str(CATEGORIES_CSV),
            "edges_csv": str(EDGES_CSV),
            "thesaurus_json": str(ALGO2_THESAURUS_JSON),
        },
        condition_count=64,
        replications=5,
        subgraph_pairs=["sg1_sg2", "sg2_sg3", "sg3_sg1"],
        prompt_preview=(
            "Your task is to recommend 5 more nodes in relation to those already "
            "in the two knowledge maps."
        ),
        method_contract=method_contract,
    ).to_dict()
