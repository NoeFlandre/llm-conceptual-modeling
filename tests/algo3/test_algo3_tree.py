from llm_conceptual_modeling.algo3.tree import (
    TreeExpansionNode,
    expand_source_tree,
)


def test_expand_source_tree_stops_descending_from_matched_words() -> None:
    expansion_calls: list[str] = []

    def propose_children(label: str, *, child_count: int) -> list[str]:
        expansion_calls.append(label)
        expansions = {
            "source_a": ["bridge_hit", "bridge_miss"],
            "bridge_miss": ["deep_child", "unused_child"],
        }
        return expansions.get(label, [])[:child_count]

    actual = expand_source_tree(
        source_labels=["source_a"],
        target_labels=["bridge_hit", "target_z"],
        child_count=2,
        max_depth=2,
        propose_children=propose_children,
    )

    assert expansion_calls == ["source_a", "bridge_miss"]
    assert actual == [
        TreeExpansionNode(
            root_label="source_a",
            parent_label="source_a",
            label="bridge_hit",
            depth=1,
            matched_target=True,
        ),
        TreeExpansionNode(
            root_label="source_a",
            parent_label="source_a",
            label="bridge_miss",
            depth=1,
            matched_target=False,
        ),
        TreeExpansionNode(
            root_label="source_a",
            parent_label="bridge_miss",
            label="deep_child",
            depth=2,
            matched_target=False,
        ),
        TreeExpansionNode(
            root_label="source_a",
            parent_label="bridge_miss",
            label="unused_child",
            depth=2,
            matched_target=False,
        ),
    ]


def test_expand_source_tree_stops_at_max_depth() -> None:
    expansion_calls: list[str] = []

    def propose_children(label: str, *, child_count: int) -> list[str]:
        expansion_calls.append(label)
        return ["depth_two", "depth_two_b"][:child_count]

    actual = expand_source_tree(
        source_labels=["source_a"],
        target_labels=["target_z"],
        child_count=1,
        max_depth=1,
        propose_children=propose_children,
    )

    assert expansion_calls == ["source_a"]
    assert actual == [
        TreeExpansionNode(
            root_label="source_a",
            parent_label="source_a",
            label="depth_two",
            depth=1,
            matched_target=False,
        )
    ]
