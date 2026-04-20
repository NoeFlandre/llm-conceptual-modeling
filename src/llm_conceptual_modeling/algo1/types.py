from typing import Protocol

from llm_conceptual_modeling.common.types import Edge


class EdgeGenerator(Protocol):
    def __call__(self, *, subgraph1: list[Edge], subgraph2: list[Edge]) -> list[Edge]: ...


class CoveVerifier(Protocol):
    def __call__(self, candidate_edges: list[Edge]) -> list[Edge]: ...
