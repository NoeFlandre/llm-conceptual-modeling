from __future__ import annotations

from typing import Protocol


class LabelProposalFunction(Protocol):
    def __call__(self, current_labels: list[str]) -> list[str]: ...


class SimilarityFunction(Protocol):
    def __call__(self, candidate_labels: list[str], seed_labels: list[str]) -> float: ...


class EdgeSuggestionFunction(Protocol):
    def __call__(self, expanded_label_context: list[str]) -> list[tuple[str, str]]: ...


class EdgeVerificationFunction(Protocol):
    def __call__(self, candidate_edges: list[tuple[str, str]]) -> list[tuple[str, str]]: ...
