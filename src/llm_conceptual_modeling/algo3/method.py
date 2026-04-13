from collections import deque
from dataclasses import dataclass
from typing import Protocol

from llm_conceptual_modeling.algo3.tree import TreeExpansionNode


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
        target_label_set = set(target_labels)
        expanded_nodes: list[TreeExpansionNode] = []
        queued_nodes: deque[tuple[str, str, int]] = deque(
            (source_label, source_label, 0) for source_label in source_labels
        )

        while queued_nodes:
            current_level_size = len(queued_nodes)
            level_nodes: list[tuple[str, str, int]] = []
            frontier_labels: list[str] = []

            for _ in range(current_level_size):
                root_label, parent_label, parent_depth = queued_nodes.popleft()
                next_depth = parent_depth + 1
                if next_depth > max_depth:
                    continue
                level_nodes.append((root_label, parent_label, next_depth))
                frontier_labels.append(parent_label)

            if not frontier_labels:
                continue

            children_by_label = propose_children_by_label(
                frontier_labels,
                child_count=child_count,
            )

            for root_label, parent_label, next_depth in level_nodes:
                for child_label in children_by_label.get(parent_label, []):
                    child_matched_target = child_label in target_label_set
                    expanded_nodes.append(
                        TreeExpansionNode(
                            root_label=root_label,
                            parent_label=parent_label,
                            label=child_label,
                            depth=next_depth,
                            matched_target=child_matched_target,
                        )
                    )
                    if not child_matched_target:
                        queued_nodes.append((root_label, child_label, next_depth))

        return expanded_nodes

    return expand_tree
