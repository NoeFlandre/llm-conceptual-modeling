from llm_conceptual_modeling.algo3.method import (
    Method3ExecutionResult,
    build_tree_expander,
    execute_method3,
)
from llm_conceptual_modeling.algo3.tree import TreeExpansionNode


def test_execute_method3_returns_tree_expansion_result() -> None:
    expansion_calls: list[tuple[list[str], list[str], int, int]] = []

    def expand_tree(
        *,
        source_labels: list[str],
        target_labels: list[str],
        child_count: int,
        max_depth: int,
    ) -> list[TreeExpansionNode]:
        call_record = (
            source_labels,
            target_labels,
            child_count,
            max_depth,
        )
        expansion_calls.append(call_record)
        return [
            TreeExpansionNode("source_a", "source_a", "bridge_hit", 1, True),
            TreeExpansionNode("source_a", "source_a", "bridge_miss", 1, False),
        ]

    actual = execute_method3(
        source_labels=["source_a"],
        target_labels=["target_a", "bridge_hit"],
        child_count=3,
        max_depth=2,
        expand_tree=expand_tree,
    )

    assert actual == Method3ExecutionResult(
        expanded_nodes=[
            TreeExpansionNode("source_a", "source_a", "bridge_hit", 1, True),
            TreeExpansionNode("source_a", "source_a", "bridge_miss", 1, False),
        ],
        matched_labels=["bridge_hit"],
    )
    assert expansion_calls == [
        (["source_a"], ["target_a", "bridge_hit"], 3, 2),
    ]


def test_build_tree_expander_adapts_dictionary_proposer_to_tree_function() -> None:
    proposal_calls: list[tuple[list[str], int]] = []

    def propose_children_by_label(
        source_labels: list[str],
        *,
        child_count: int,
    ) -> dict[str, list[str]]:
        proposal_calls.append((source_labels, child_count))
        return {
            "source_a": ["bridge_hit", "bridge_miss"],
            "bridge_miss": ["deep_child", "unused_child"],
        }

    expand_tree = build_tree_expander(propose_children_by_label)
    actual = expand_tree(
        source_labels=["source_a"],
        target_labels=["bridge_hit", "target_z"],
        child_count=2,
        max_depth=2,
    )

    assert proposal_calls == [
        (["source_a"], 2),
        (["bridge_miss"], 2),
    ]
    assert actual == [
        TreeExpansionNode("source_a", "source_a", "bridge_hit", 1, True),
        TreeExpansionNode("source_a", "source_a", "bridge_miss", 1, False),
        TreeExpansionNode("source_a", "bridge_miss", "deep_child", 2, False),
        TreeExpansionNode("source_a", "bridge_miss", "unused_child", 2, False),
    ]


def test_build_tree_expander_batches_frontier_labels_per_depth_level() -> None:
    proposal_calls: list[tuple[list[str], int]] = []

    def propose_children_by_label(
        source_labels: list[str],
        *,
        child_count: int,
    ) -> dict[str, list[str]]:
        proposal_calls.append((source_labels, child_count))
        if source_labels == ["source_a"]:
            return {
                "source_a": ["bridge_left", "bridge_right"],
            }
        if source_labels == ["bridge_left", "bridge_right"]:
            return {
                "bridge_left": ["left_deep"],
                "bridge_right": ["right_deep"],
            }
        raise AssertionError(f"unexpected batched frontier: {source_labels}")

    expand_tree = build_tree_expander(propose_children_by_label)
    actual = expand_tree(
        source_labels=["source_a"],
        target_labels=["target_z"],
        child_count=2,
        max_depth=2,
    )

    assert proposal_calls == [
        (["source_a"], 2),
        (["bridge_left", "bridge_right"], 2),
    ]
    assert actual == [
        TreeExpansionNode("source_a", "source_a", "bridge_left", 1, False),
        TreeExpansionNode("source_a", "source_a", "bridge_right", 1, False),
        TreeExpansionNode("source_a", "bridge_left", "left_deep", 2, False),
        TreeExpansionNode("source_a", "bridge_right", "right_deep", 2, False),
    ]
