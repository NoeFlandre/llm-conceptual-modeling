from collections.abc import Iterable

import networkx as nx

Edge = tuple[str, str]


def _nodes_from_edges(edges: Iterable[Edge]) -> set[str]:
    nodes: set[str] = set()
    for start, end in edges:
        nodes.add(start)
        nodes.add(end)
    return nodes


def _sorted_edge(left: str, right: str) -> Edge:
    return (left, right) if left <= right else (right, left)


def find_valid_connections(
    graph: nx.DiGraph,
    subgraph1: list[Edge],
    subgraph2: list[Edge],
) -> set[tuple[str, str]]:
    nodes1 = _nodes_from_edges(subgraph1)
    nodes2 = _nodes_from_edges(subgraph2)
    undirected_graph = graph.to_undirected()
    connections: set[tuple[str, str]] = set()

    for start_node in nodes1:
        for end_node in nodes2:
            if start_node == end_node:
                continue
            if start_node not in undirected_graph or end_node not in undirected_graph:
                continue

            try:
                paths = list(nx.all_shortest_paths(undirected_graph, start_node, end_node))
            except nx.NetworkXNoPath:
                continue

            for path in paths:
                if not any(
                    (node in nodes1 and node != start_node)
                    or (node in nodes2 and node != end_node)
                    for node in path
                ):
                    connections.add(_sorted_edge(start_node, end_node))

    for start_node in nodes2:
        for end_node in nodes1:
            if start_node == end_node:
                continue
            if start_node not in undirected_graph or end_node not in undirected_graph:
                continue

            try:
                paths = list(nx.all_shortest_paths(undirected_graph, start_node, end_node))
            except nx.NetworkXNoPath:
                continue

            for path in paths:
                if not any(
                    (node in nodes1 and node != end_node)
                    or (node in nodes2 and node != start_node)
                    for node in path
                ):
                    connections.add(_sorted_edge(start_node, end_node))

    return connections
