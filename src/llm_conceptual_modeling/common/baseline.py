from __future__ import annotations

import random
from collections.abc import Iterable

Edge = tuple[str, str]

# Fixed seed for reproducibility across runs and environments
_RANDOM_SEED = 42

# Fixed number of edges to sample from all cross-subgraph node pairs
_SAMPLE_COUNT = 20


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


def propose_random_cross_subgraph_edges(
    subgraph1: Iterable[Edge],
    subgraph2: Iterable[Edge],
    *,
    sample_count: int = _SAMPLE_COUNT,
    seed: int = _RANDOM_SEED,
) -> list[Edge]:
    """Sample `sample_count` random cross-subgraph edges from all possible pairs.

    Unlike ``propose_direct_cross_subgraph_edges`` this does NOT consult the
    mother graph — it only knows which nodes belong to each subgraph.  It is
    therefore a much weaker baseline: it has no signal about which node pairs
    are actually connected, only which pairs *could* be connected.
    """
    rng = random.Random(seed)
    sg1_nodes = sorted(_nodes_from_edges(subgraph1))
    sg2_nodes = sorted(_nodes_from_edges(subgraph2))

    all_pairs: list[Edge] = [
        (min(a, b), max(a, b)) for a in sg1_nodes for b in sg2_nodes if a != b
    ]
    return sorted(rng.sample(all_pairs, min(sample_count, len(all_pairs))))


def propose_random_k_edges(
    candidate_edges: Iterable[Edge],
    k: int,
    *,
    seed: int = _RANDOM_SEED,
) -> list[Edge]:
    """Sample exactly k edges uniformly at random from a candidate edge set.

    This is the random-k baseline used in the non-LLM comparison: for each
    LLM output row, the baseline samples exactly k edges (where k equals the
    number of edges the LLM proposed) from the mother graph, with no knowledge
    of which edges are cross-subgraph.  Both the LLM and the baseline are
    then evaluated against the ground-truth cross edges, giving a fair,
    volume-matched comparison.
    """
    rng = random.Random(seed)
    candidates = [_normalize_edge(e) for e in candidate_edges]
    if k >= len(candidates):
        return sorted(set(candidates))
    return sorted(rng.sample(candidates, k))


def _nodes_from_edges(edges: Iterable[Edge]) -> set[str]:
    nodes: set[str] = set()
    for left, right in edges:
        nodes.add(left)
        nodes.add(right)
    return nodes


def _normalize_edge(edge: Edge) -> Edge:
    left, right = edge
    return (left, right) if left <= right else (right, left)
