from llm_conceptual_modeling.common.baseline import propose_direct_cross_subgraph_edges


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
