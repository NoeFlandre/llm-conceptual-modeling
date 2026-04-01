from llm_conceptual_modeling.common.baseline import (
    propose_direct_cross_subgraph_edges,
    propose_edit_distance_cross_subgraph_edges,
    propose_wordnet_cross_subgraph_edges,
)


def test_propose_direct_cross_subgraph_edges_returns_missing_cross_edges_only() -> None:
    mother_graph = [
        ("a", "b"),
        ("b", "c"),
        ("a", "d"),
        ("x", "y"),
    ]
    subgraph1 = [("a", "b")]
    subgraph2 = [("c", "d")]

    actual = propose_direct_cross_subgraph_edges(mother_graph, subgraph1, subgraph2)

    assert actual == [("a", "d"), ("b", "c")]


def test_propose_direct_cross_subgraph_edges_excludes_existing_subgraph_edges() -> None:
    mother_graph = [
        ("a", "c"),
        ("c", "a"),
        ("a", "b"),
    ]
    subgraph1 = [("a", "c")]
    subgraph2 = [("c", "a")]

    actual = propose_direct_cross_subgraph_edges(mother_graph, subgraph1, subgraph2)

    assert actual == []


def test_propose_wordnet_cross_subgraph_edges_prefers_semantic_matches() -> None:
    subgraph1 = [
        ("car", "road"),
        ("doctor", "clinic"),
    ]
    subgraph2 = [
        ("automobile", "garage"),
        ("physician", "hospital"),
    ]
    wordnet_lexicon = {
        "car": ["auto", "automobile", "car"],
        "doctor": ["doctor", "physician"],
        "automobile": ["auto", "automobile", "car"],
        "physician": ["doctor", "physician"],
    }

    actual = propose_wordnet_cross_subgraph_edges(
        subgraph1,
        subgraph2,
        sample_count=2,
        wordnet_lexicon=wordnet_lexicon,
    )

    assert actual[0] == ("automobile", "car")
    assert ("doctor", "physician") in actual


def test_propose_edit_distance_cross_subgraph_edges_prefers_lexically_similar_labels() -> None:
    subgraph1 = [
        ("physical activity", "exercise"),
    ]
    subgraph2 = [
        ("physical activities", "training"),
    ]

    actual = propose_edit_distance_cross_subgraph_edges(
        subgraph1,
        subgraph2,
        sample_count=2,
    )

    assert actual[0] == ("physical activities", "physical activity")
