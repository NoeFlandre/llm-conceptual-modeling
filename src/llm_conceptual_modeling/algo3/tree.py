from collections import deque
from dataclasses import dataclass
from typing import Protocol


class ChildProposer(Protocol):
    def __call__(self, label: str, *, child_count: int) -> list[str]: ...


@dataclass(frozen=True)
class TreeExpansionNode:
    root_label: str
    parent_label: str
    label: str
    depth: int
    matched_target: bool


def expand_source_tree(
    *,
    source_labels: list[str],
    target_labels: list[str],
    child_count: int,
    max_depth: int,
    propose_children: ChildProposer,
) -> list[TreeExpansionNode]:
    target_label_set = set(target_labels)
    queued_nodes: deque[tuple[str, str, int]] = deque()
    expanded_nodes: list[TreeExpansionNode] = []

    for source_label in source_labels:
        root_entry = (source_label, source_label, 0)
        queued_nodes.append(root_entry)

    while queued_nodes:
        root_label, parent_label, parent_depth = queued_nodes.popleft()
        next_depth = parent_depth + 1
        depth_limit_reached = next_depth > max_depth

        if depth_limit_reached:
            continue

        proposed_children = propose_children(parent_label, child_count=child_count)

        for child_label in proposed_children:
            child_matched_target = child_label in target_label_set
            child_node = TreeExpansionNode(
                root_label=root_label,
                parent_label=parent_label,
                label=child_label,
                depth=next_depth,
                matched_target=child_matched_target,
            )
            expanded_nodes.append(child_node)

            if child_matched_target:
                continue

            queued_child = (root_label, child_label, next_depth)
            queued_nodes.append(queued_child)

    return expanded_nodes
