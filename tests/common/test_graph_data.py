from llm_conceptual_modeling.algo1.generation import build_generation_manifest as algo1_manifest
from llm_conceptual_modeling.algo2.generation import build_generation_manifest as algo2_manifest
from llm_conceptual_modeling.algo3.generation import build_generation_manifest as algo3_manifest
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus, load_default_graph


def test_default_graph_partitions_are_non_empty() -> None:
    sg1, sg2, sg3, mother = load_default_graph()

    assert sg1
    assert sg2
    assert sg3
    assert mother


def test_default_graph_subgraphs_are_edge_disjoint() -> None:
    sg1, sg2, sg3, _ = load_default_graph()

    assert set(sg1).isdisjoint(sg2)
    assert set(sg1).isdisjoint(sg3)
    assert set(sg2).isdisjoint(sg3)


def test_generation_manifests_preserve_legacy_condition_counts() -> None:
    assert algo1_manifest(fixture_only=False)["condition_count"] == 32
    assert algo2_manifest(fixture_only=False)["condition_count"] == 64
    assert algo3_manifest(fixture_only=False)["condition_count"] == 16


def test_algo2_thesaurus_is_tracked_and_non_empty() -> None:
    thesaurus = load_algo2_thesaurus()
    synonyms = thesaurus["synonyms"]
    antonyms = thesaurus["antonyms"]

    assert synonyms
    assert antonyms
    assert "Obesity" in synonyms
    assert "Obesity" in antonyms
