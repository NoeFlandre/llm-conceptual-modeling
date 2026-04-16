from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TreeExpansionNode:
    root_label: str
    parent_label: str
    label: str
    depth: int
    matched_target: bool


class ChildProposer(Protocol):
    def __call__(self, label: str, *, child_count: int) -> list[str]: ...


class TreeExpansionFunction(Protocol):
    def __call__(
        self,
        *,
        source_labels: list[str],
        target_labels: list[str],
        child_count: int,
        max_depth: int,
    ) -> list[TreeExpansionNode]: ...


class ChildDictionaryProposer(Protocol):
    def __call__(
        self,
        source_labels: list[str],
        *,
        child_count: int,
    ) -> dict[str, list[str]]: ...
