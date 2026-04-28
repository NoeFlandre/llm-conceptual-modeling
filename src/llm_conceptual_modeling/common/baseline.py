from __future__ import annotations

import random
import re
from collections.abc import Iterable
from typing import Callable

from llm_conceptual_modeling.common.graph_data import load_wordnet_label_lexicon
from llm_conceptual_modeling.common.types import Edge

# Fixed seed for reproducibility across runs and environments
_RANDOM_SEED = 42

# Fixed number of edges to sample from all cross-subgraph node pairs
_SAMPLE_COUNT = 20
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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

    This helper samples from an explicit candidate set. The manuscript-facing
    random-k comparison samples from all admissible cross-subgraph pairs in
    ``analysis._baseline_sampling`` rather than from the mother graph.
    """
    rng = random.Random(seed)
    candidates = sorted({_normalize_edge(e) for e in candidate_edges})
    if k >= len(candidates):
        return candidates
    return sorted(rng.sample(candidates, k))


def propose_strategy_cross_subgraph_edges(
    mother_graph: Iterable[Edge],
    subgraph1: Iterable[Edge],
    subgraph2: Iterable[Edge],
    *,
    strategy: str,
    sample_count: int = _SAMPLE_COUNT,
) -> list[Edge]:
    if strategy == "direct-cross-graph":
        return propose_direct_cross_subgraph_edges(mother_graph, subgraph1, subgraph2)
    if strategy == "random-uniform-subset":
        return propose_random_cross_subgraph_edges(
            subgraph1,
            subgraph2,
            sample_count=sample_count,
        )
    if strategy == "wordnet-ontology-match":
        return propose_wordnet_cross_subgraph_edges(
            subgraph1,
            subgraph2,
            sample_count=sample_count,
        )
    if strategy == "edit-distance":
        return propose_edit_distance_cross_subgraph_edges(
            subgraph1,
            subgraph2,
            sample_count=sample_count,
        )
    raise ValueError(f"Unsupported baseline strategy: {strategy}")


def propose_wordnet_cross_subgraph_edges(
    subgraph1: Iterable[Edge],
    subgraph2: Iterable[Edge],
    *,
    sample_count: int = _SAMPLE_COUNT,
    wordnet_lexicon: dict[str, list[str]] | None = None,
) -> list[Edge]:
    lexicon = wordnet_lexicon if wordnet_lexicon is not None else load_wordnet_label_lexicon()
    ranked_pairs = _rank_cross_subgraph_edges(
        subgraph1,
        subgraph2,
        scorer=lambda left, right: _wordnet_similarity(left, right, lexicon),
    )
    return ranked_pairs[:sample_count]


def propose_edit_distance_cross_subgraph_edges(
    subgraph1: Iterable[Edge],
    subgraph2: Iterable[Edge],
    *,
    sample_count: int = _SAMPLE_COUNT,
) -> list[Edge]:
    ranked_pairs = _rank_cross_subgraph_edges(
        subgraph1,
        subgraph2,
        scorer=_edit_distance_similarity,
    )
    return ranked_pairs[:sample_count]


def _nodes_from_edges(edges: Iterable[Edge]) -> set[str]:
    nodes: set[str] = set()
    for left, right in edges:
        nodes.add(left)
        nodes.add(right)
    return nodes


def _normalize_edge(edge: Edge) -> Edge:
    left, right = edge
    return (left, right) if left <= right else (right, left)


def _rank_cross_subgraph_edges(
    subgraph1: Iterable[Edge],
    subgraph2: Iterable[Edge],
    *,
    scorer: Callable[[str, str], float],
) -> list[Edge]:
    sg1_nodes = sorted(_nodes_from_edges(subgraph1))
    sg2_nodes = sorted(_nodes_from_edges(subgraph2))

    scored_edges: list[tuple[float, Edge]] = []
    for left in sg1_nodes:
        for right in sg2_nodes:
            if left == right:
                continue
            edge = _normalize_edge((left, right))
            score = scorer(left, right)
            scored_edges.append((score, edge))

    ranked_edges = sorted(
        scored_edges,
        key=lambda item: (-item[0], item[1][0], item[1][1]),
    )
    return [edge for _, edge in ranked_edges]


def _wordnet_similarity(
    left_label: str,
    right_label: str,
    wordnet_lexicon: dict[str, list[str]],
) -> float:
    left_terms = _expanded_label_terms(left_label, wordnet_lexicon)
    right_terms = _expanded_label_terms(right_label, wordnet_lexicon)
    union = left_terms | right_terms
    if not union:
        return 0.0
    overlap = left_terms & right_terms
    return len(overlap) / len(union)


def _expanded_label_terms(
    label: str,
    wordnet_lexicon: dict[str, list[str]],
) -> set[str]:
    normalized_label = label.lower().strip()
    expanded_terms = {
        term.lower().strip()
        for term in wordnet_lexicon.get(normalized_label, [])
        if term.strip()
    }
    expanded_terms.update(_tokenize_label(normalized_label))
    return expanded_terms


def _edit_distance_similarity(left_label: str, right_label: str) -> float:
    normalized_left = " ".join(_tokenize_label(left_label))
    normalized_right = " ".join(_tokenize_label(right_label))
    if not normalized_left or not normalized_right:
        return 0.0
    max_length = max(len(normalized_left), len(normalized_right))
    if max_length == 0:
        return 1.0
    distance = _levenshtein_distance(normalized_left, normalized_right)
    return 1.0 - (distance / max_length)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion_cost = current_row[right_index - 1] + 1
            deletion_cost = previous_row[right_index] + 1
            substitution_cost = previous_row[right_index - 1] + (left_char != right_char)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row
    return previous_row[-1]


def _tokenize_label(label: str) -> list[str]:
    return _TOKEN_PATTERN.findall(label.lower())
