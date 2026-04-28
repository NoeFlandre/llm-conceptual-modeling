"""Baseline edge sampling and count computation for baseline comparison.

Provides cached sampling for three non-LLM baseline strategies:
- random-k: sample k direct edges uniformly at random from all admissible
  cross-subgraph node pairs
- wordnet-ontology-match: rank by WordNet semantic overlap
- edit-distance: rank by edit-distance similarity

And a cached baseline-count pipeline that feeds into metric row construction.
"""

from __future__ import annotations

from functools import lru_cache

from llm_conceptual_modeling.common.baseline import (
    propose_random_cross_subgraph_edges,
    propose_strategy_cross_subgraph_edges,
)
from llm_conceptual_modeling.common.connection_eval import find_valid_connections

_RANDOM_SEED = 42


def _sample_baseline_edges(
    *,
    baseline_strategy: str,
    k: int,
    mother_edges: list[tuple[str, str]],
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
    random_seed: int = _RANDOM_SEED,
) -> set[tuple[str, str]]:
    mother_key = tuple(mother_edges)
    subgraph1_key = tuple(subgraph1_edges)
    subgraph2_key = tuple(subgraph2_edges)
    if k == 0:
        return set()
    ranked_edges = _ranked_baseline_edges(
        baseline_strategy=baseline_strategy,
        mother_edges=mother_key,
        subgraph1_edges=subgraph1_key,
        subgraph2_edges=subgraph2_key,
        random_seed=random_seed,
    )
    return set(ranked_edges[:k])


@lru_cache(maxsize=None)
def _ranked_baseline_edges(
    *,
    baseline_strategy: str,
    mother_edges: tuple[tuple[str, str], ...],
    subgraph1_edges: tuple[tuple[str, str], ...],
    subgraph2_edges: tuple[tuple[str, str], ...],
    random_seed: int,
) -> tuple[tuple[str, str], ...]:
    if baseline_strategy == "random-k":
        pair_count = _cross_subgraph_pair_count(subgraph1_edges, subgraph2_edges)
        return tuple(
            propose_random_cross_subgraph_edges(
                subgraph1_edges,
                subgraph2_edges,
                sample_count=pair_count,
                seed=random_seed,
            )
        )
    subgraph1_nodes = {node for edge in subgraph1_edges for node in edge}
    subgraph2_nodes = {node for edge in subgraph2_edges for node in edge}
    max_pair_count = len(subgraph1_nodes) * len(subgraph2_nodes)
    return tuple(
        propose_strategy_cross_subgraph_edges(
            mother_edges,
            subgraph1_edges,
            subgraph2_edges,
            strategy=baseline_strategy,
            sample_count=max(max_pair_count, len(mother_edges)),
        )
    )


def _compute_baseline_counts(
    *,
    baseline_strategy: str,
    k: int,
    mother_edges: list[tuple[str, str]],
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
    random_seed: int = _RANDOM_SEED,
) -> dict[str, int]:
    return dict(
        _compute_baseline_counts_cached(
            baseline_strategy=baseline_strategy,
            k=k,
            mother_edges=tuple(mother_edges),
            subgraph1_edges=tuple(subgraph1_edges),
            subgraph2_edges=tuple(subgraph2_edges),
            ground_truth=tuple(sorted(ground_truth)),
            random_seed=random_seed,
        )
    )


@lru_cache(maxsize=None)
def _compute_baseline_counts_cached(
    *,
    baseline_strategy: str,
    k: int,
    mother_edges: tuple[tuple[str, str], ...],
    subgraph1_edges: tuple[tuple[str, str], ...],
    subgraph2_edges: tuple[tuple[str, str], ...],
    ground_truth: tuple[tuple[str, str], ...],
    random_seed: int = _RANDOM_SEED,
) -> tuple[tuple[str, int], ...]:
    baseline_edges = _sample_baseline_edges(
        baseline_strategy=baseline_strategy,
        k=k,
        mother_edges=list(mother_edges),
        subgraph1_edges=list(subgraph1_edges),
        subgraph2_edges=list(subgraph2_edges),
        random_seed=random_seed,
    )
    proposed_edges = [*subgraph1_edges, *subgraph2_edges, *sorted(baseline_edges)]
    generated_connections = find_valid_connections(
        proposed_edges,
        list(subgraph1_edges),
        list(subgraph2_edges),
    )
    ground_truth_edges = set(ground_truth)
    return tuple(
        {
            "tp": len(generated_connections & ground_truth_edges),
            "fp": len(generated_connections - ground_truth_edges),
            "fn": len(ground_truth_edges - generated_connections),
        }.items()
    )


def _scored_connection_count(
    proposed_edges: list[tuple[str, str]],
    *,
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
) -> int:
    generated_connections = find_valid_connections(
        [*subgraph1_edges, *subgraph2_edges, *proposed_edges],
        subgraph1_edges,
        subgraph2_edges,
    )
    return len(generated_connections)


def _cross_subgraph_pair_count(
    subgraph1_edges: tuple[tuple[str, str], ...],
    subgraph2_edges: tuple[tuple[str, str], ...],
) -> int:
    subgraph1_nodes = {node for edge in subgraph1_edges for node in edge}
    subgraph2_nodes = {node for edge in subgraph2_edges for node in edge}
    return len(
        {
            tuple(sorted((left, right)))
            for left in subgraph1_nodes
            for right in subgraph2_nodes
            if left != right
        }
    )
