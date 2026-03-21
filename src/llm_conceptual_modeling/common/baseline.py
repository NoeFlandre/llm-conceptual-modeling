from collections.abc import Iterable

Edge = tuple[str, str]


def propose_direct_cross_subgraph_edges(
    mother_graph: Iterable[Edge],
    subgraph1: Iterable[Edge],
    subgraph2: Iterable[Edge],
) -> list[Edge]:
    subgraph1_nodes = _nodes_from_edges(subgraph1)
    subgraph2_nodes = _nodes_from_edges(subgraph2)
    existing_edges = {_normalize_edge(edge) for edge in [*subgraph1, *subgraph2]}

    proposed_edges: set[Edge] = set()
    for edge in mother_graph:
        normalized = _normalize_edge(edge)
        left, right = normalized
        crosses_subgraphs = (left in subgraph1_nodes and right in subgraph2_nodes) or (
            left in subgraph2_nodes and right in subgraph1_nodes
        )
        if not crosses_subgraphs:
            continue
        if normalized in existing_edges:
            continue
        proposed_edges.add(normalized)

    return sorted(proposed_edges)


def _nodes_from_edges(edges: Iterable[Edge]) -> set[str]:
    nodes: set[str] = set()
    for left, right in edges:
        nodes.add(left)
        nodes.add(right)
    return nodes


def _normalize_edge(edge: Edge) -> Edge:
    left, right = edge
    return (left, right) if left <= right else (right, left)
