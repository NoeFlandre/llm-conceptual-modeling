from dataclasses import dataclass
from typing import Protocol

from llm_conceptual_modeling.algo3.tree import TreeExpansionNode, expand_source_tree


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


@dataclass(frozen=True)
class Method3ExecutionResult:
    expanded_nodes: list[TreeExpansionNode]
    matched_labels: list[str]


def execute_method3(
    *,
    source_labels: list[str],
    target_labels: list[str],
    child_count: int,
    max_depth: int,
    expand_tree: TreeExpansionFunction,
) -> Method3ExecutionResult:
    expanded_nodes = expand_tree(
        source_labels=source_labels,
        target_labels=target_labels,
        child_count=child_count,
        max_depth=max_depth,
    )

    matched_labels: list[str] = []

    for expanded_node in expanded_nodes:
        if not expanded_node.matched_target:
            continue

        matched_labels.append(expanded_node.label)

    return Method3ExecutionResult(
        expanded_nodes=expanded_nodes,
        matched_labels=matched_labels,
    )


def build_tree_expander(
    propose_children_by_label: ChildDictionaryProposer,
) -> TreeExpansionFunction:
    def expand_tree(
        *,
        source_labels: list[str],
        target_labels: list[str],
        child_count: int,
        max_depth: int,
    ) -> list[TreeExpansionNode]:
        def propose_children(label: str, *, child_count: int) -> list[str]:
            children_by_label = propose_children_by_label(
                [label],
                child_count=child_count,
            )
            child_labels = children_by_label.get(label, [])
            return child_labels

        expanded_nodes = expand_source_tree(
            source_labels=source_labels,
            target_labels=target_labels,
            child_count=child_count,
            max_depth=max_depth,
            propose_children=propose_children,
        )
        return expanded_nodes

    return expand_tree
