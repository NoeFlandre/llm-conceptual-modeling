from dataclasses import dataclass
from typing import Protocol

Edge = tuple[str, str]


class EdgeGenerator(Protocol):
    def __call__(self, *, subgraph1: list[Edge], subgraph2: list[Edge]) -> list[Edge]: ...


class CoveVerifier(Protocol):
    def __call__(self, candidate_edges: list[Edge]) -> list[Edge]: ...


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
