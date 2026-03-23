from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

Edge = tuple[str, str]


def _nodes_from_edges(edges: Iterable[Edge]) -> set[str]:
    nodes: set[str] = set()
    for start, end in edges:
        nodes.add(start)
        nodes.add(end)
    return nodes


def _sorted_edge(left: str, right: str) -> Edge:
    return (left, right) if left <= right else (right, left)


def _edges_from_graph(graph: Any) -> list[Edge]:
    if hasattr(graph, "edges"):
        edges_attr = graph.edges
        edges = edges_attr() if callable(edges_attr) else edges_attr
        return [(str(start), str(end)) for start, end in edges]

    return [(str(start), str(end)) for start, end in graph]


def _build_undirected_adjacency(edges: Iterable[Edge]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for start, end in edges:
        adjacency[start].add(end)
        adjacency[end].add(start)
    return adjacency


def _has_path(adjacency: dict[str, set[str]], start: str, end: str) -> bool:
    if start == end:
        return True

    visited = {start}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor == end:
                return True
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def find_valid_connections(
    graph: Any,
    subgraph1: list[Edge],
    subgraph2: list[Edge],
) -> set[tuple[str, str]]:
    nodes1 = _nodes_from_edges(subgraph1)
    nodes2 = _nodes_from_edges(subgraph2)
    adjacency = _build_undirected_adjacency(_edges_from_graph(graph))
    connections: set[tuple[str, str]] = set()

    for start_node in nodes1:
        for end_node in nodes2:
            if start_node == end_node:
                continue
            if start_node not in adjacency or end_node not in adjacency:
                continue
            if _has_path(adjacency, start_node, end_node):
                connections.add(_sorted_edge(start_node, end_node))

    for start_node in nodes2:
        for end_node in nodes1:
            if start_node == end_node:
                continue
            if start_node not in adjacency or end_node not in adjacency:
                continue
            if _has_path(adjacency, start_node, end_node):
                connections.add(_sorted_edge(start_node, end_node))

    return connections
