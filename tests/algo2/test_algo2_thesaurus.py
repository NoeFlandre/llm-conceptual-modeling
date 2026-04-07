from llm_conceptual_modeling.algo2.thesaurus import (
    build_term_normalizer,
    normalize_edge_terms,
)
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus


def test_build_term_normalizer_maps_synonyms_and_antonyms_to_base_terms() -> None:
    thesaurus = load_algo2_thesaurus()

    normalizer = build_term_normalizer(thesaurus)

    assert normalizer["Cholesterol"] == "Blood saturated fatty acid level"
    assert normalizer["Weight loss"] == "Obesity"
    assert normalizer["Obesity"] == "Obesity"


def test_normalize_edge_terms_rewrites_known_variants_and_preserves_unknown_terms() -> None:
    thesaurus = load_algo2_thesaurus()
    edges = [
        ("Cholesterol", "Weight gain"),
        ("Weight loss", "Body image"),
        ("Untracked term", "Stress level"),
    ]

    actual = normalize_edge_terms(edges, thesaurus)

    assert actual == [
        ("Blood saturated fatty acid level", "Obesity"),
        ("Obesity", "Poor body image"),
        ("Untracked term", "Stress"),
    ]


def test_build_term_normalizer_treats_missing_sections_as_empty() -> None:
    normalizer = build_term_normalizer({"synonyms": {}})

    assert normalizer == {}


def test_normalize_edge_terms_accepts_thesaurus_without_antonyms() -> None:
    edges = [("Adiposity", "Body image"), ("Untracked term", "Stress level")]

    actual = normalize_edge_terms(
        edges,
        {
            "synonyms": {
                "Obesity": ["Adiposity"],
                "Poor body image": ["Body image"],
            }
        },
    )

    assert actual == [
        ("Obesity", "Poor body image"),
        ("Untracked term", "Stress level"),
    ]
