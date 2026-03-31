from dataclasses import dataclass
from typing import Protocol

from llm_conceptual_modeling.algo2.embeddings import (
    EmbeddingClient,
    compute_average_best_match_similarity,
)
from llm_conceptual_modeling.algo2.expansion import run_label_expansion
from llm_conceptual_modeling.algo2.thesaurus import normalize_edge_terms

Edge = tuple[str, str]
Thesaurus = dict[str, dict[str, list[str]]]


class LabelProposalFunction(Protocol):
    def __call__(self, current_labels: list[str]) -> list[str]: ...


class EdgeSuggestionFunction(Protocol):
    def __call__(self, expanded_label_context: list[str]) -> list[Edge]: ...


class EdgeVerificationFunction(Protocol):
    def __call__(self, candidate_edges: list[Edge]) -> list[Edge]: ...


@dataclass(frozen=True)
class Method2ExecutionResult:
    expanded_labels: list[str]
    raw_edges: list[Edge]
    normalized_edges: list[Edge]
    final_similarity: float
    iteration_count: int


def execute_method2(
    *,
    seed_labels: list[str],
    existing_edges: list[Edge] | None = None,
    propose_labels: LabelProposalFunction,
    suggest_edges: EdgeSuggestionFunction,
    verify_edges: EdgeVerificationFunction | None = None,
    embedding_client: EmbeddingClient,
    convergence_threshold: float,
    thesaurus: Thesaurus,
) -> Method2ExecutionResult:
    def measure_similarity(candidate_labels: list[str], source_labels: list[str]) -> float:
        similarity: float = compute_average_best_match_similarity(
            candidate_labels=candidate_labels,
            seed_labels=source_labels,
            client=embedding_client,
        )
        return similarity

    expansion_result = run_label_expansion(
        seed_labels=seed_labels,
        propose_labels=propose_labels,
        measure_similarity=measure_similarity,  # type: ignore[arg-type]
        threshold=convergence_threshold,
    )
    expanded_label_context = list(seed_labels) + expansion_result.expanded_labels
    raw_edges = suggest_edges(expanded_label_context)
    normalized_edges = normalize_edge_terms(raw_edges, thesaurus)
    existing_edges = existing_edges or []
    existing_edge_set = {
        _normalize_edge(edge) for edge in normalize_edge_terms(existing_edges, thesaurus)
    }
    filtered_candidate_edges = [
        edge for edge in normalized_edges if _normalize_edge(edge) not in existing_edge_set
    ]
    verification_function = verify_edges or _identity_verifier
    verified_edges = verification_function(filtered_candidate_edges)
    final_iteration = expansion_result.iterations[-1]
    final_similarity = final_iteration.similarity
    iteration_count = len(expansion_result.iterations)
    return Method2ExecutionResult(
        expanded_labels=expansion_result.expanded_labels,
        raw_edges=raw_edges,
        normalized_edges=verified_edges,
        final_similarity=final_similarity,
        iteration_count=iteration_count,
    )


def _normalize_edge(edge: Edge) -> Edge:
    left, right = edge
    return (left, right) if left <= right else (right, left)


def _identity_verifier(candidate_edges: list[Edge]) -> list[Edge]:
    return candidate_edges
