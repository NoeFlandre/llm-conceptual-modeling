from llm_conceptual_modeling.algo3.mistral import (
    Method3PromptConfig,
    build_child_proposer,
    build_tree_expansion_prompt,
)


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        call_record = {
            "prompt": prompt,
            "schema_name": schema_name,
        }
        self.calls.append(call_record)
        return {
            "children_by_label": {
                "source_a": ["bridge_one", "bridge_two"],
                "source_b": ["bridge_three", "bridge_four"],
            }
        }


def test_build_tree_expansion_prompt_reflects_method3_contract() -> None:
    actual = build_tree_expansion_prompt(
        source_labels=["source_a", "source_b"],
        child_count=3,
    )

    assert "recommend 3 related concept names for each of the names in the input" in actual
    assert "dictionary format" in actual
    assert "source_a" in actual
    assert "source_b" in actual


def test_build_tree_expansion_prompt_can_include_example_and_counterexample() -> None:
    prompt_config = Method3PromptConfig(
        include_example=True,
        include_counterexample=True,
    )

    actual = build_tree_expansion_prompt(
        source_labels=["source_a", "source_b"],
        child_count=3,
        prompt_config=prompt_config,
    )

    assert "All concept names must have a clear meaning" in actual
    assert "Here is an example of a desired output for your task." in actual
    assert "Here is an example of a bad output that we do not want to see." in actual


def test_build_tree_expansion_prompt_can_omit_example_and_counterexample() -> None:
    prompt_config = Method3PromptConfig(
        include_example=False,
        include_counterexample=False,
    )

    actual = build_tree_expansion_prompt(
        source_labels=["source_a", "source_b"],
        child_count=5,
        prompt_config=prompt_config,
    )

    assert "Here is an example of a desired output for your task." not in actual
    assert "Here is an example of a bad output that we do not want to see." not in actual


def test_build_child_proposer_returns_dictionary_children() -> None:
    chat_client = FakeChatClient()
    propose_children = build_child_proposer(chat_client)

    actual = propose_children(
        ["source_a", "source_b"],
        child_count=3,
    )

    assert actual == {
        "source_a": ["bridge_one", "bridge_two"],
        "source_b": ["bridge_three", "bridge_four"],
    }
    assert len(chat_client.calls) == 1
    assert chat_client.calls[0]["schema_name"] == "children_by_label"
