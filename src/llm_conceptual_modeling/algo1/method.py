from dataclasses import dataclass

from llm_conceptual_modeling.algo1.types import CoveVerifier, Edge, EdgeGenerator


@dataclass(frozen=True)
class Method1ExecutionResult:
    candidate_edges: list[Edge]
    verified_edges: list[Edge]


def execute_method1(
    *,
    subgraph1: list[Edge],
    subgraph2: list[Edge],
    generate_edges: EdgeGenerator,
    verify_edges: CoveVerifier,
) -> Method1ExecutionResult:
    candidate_edges = generate_edges(subgraph1=subgraph1, subgraph2=subgraph2)
    verified_edges = verify_edges(candidate_edges)
    return Method1ExecutionResult(
        candidate_edges=candidate_edges,
        verified_edges=verified_edges,
    )
