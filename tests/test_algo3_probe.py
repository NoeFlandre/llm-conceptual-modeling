import json

from llm_conceptual_modeling.algo3.mistral import Method3PromptConfig
from llm_conceptual_modeling.algo3.probe import (
    Algo3ProbeSpec,
    run_algo3_probe,
)


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | object]] = []

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        call_record: dict[str, str | object] = {
            "prompt": prompt,
            "schema_name": schema_name,
        }

        self.calls.append(call_record)
        return {
            "children_by_label": {
                "source_a": ["bridge_hit", "bridge_miss"],
                "bridge_miss": ["deep_child", "unused_child"],
            }
        }


def test_run_algo3_probe_writes_auditable_artifacts(tmp_path) -> None:
    chat_client = FakeChatClient()
    probe_dir = tmp_path / "algo3_probe"
    spec = Algo3ProbeSpec(
        run_name="algo3_single_row_v1",
        model="mistral-small-2603",
        source_labels=["source_a"],
        target_labels=["bridge_hit", "target_z"],
        prompt_config=Method3PromptConfig(
            include_example=True,
            include_counterexample=False,
        ),
        child_count=2,
        max_depth=2,
        output_dir=probe_dir,
    )

    actual = run_algo3_probe(
        spec=spec,
        chat_client=chat_client,
    )

    manifest = json.loads((probe_dir / "manifest.json").read_text())
    summary = json.loads((probe_dir / "summary.json").read_text())
    event_lines = (probe_dir / "events.jsonl").read_text().splitlines()
    tree_prompt = (probe_dir / "tree_expansion_prompt.txt").read_text()

    assert actual == {
        "run_name": "algo3_single_row_v1",
        "model": "mistral-small-2603",
        "expanded_nodes": [
            {
                "root_label": "source_a",
                "parent_label": "source_a",
                "label": "bridge_hit",
                "depth": 1,
                "matched_target": True,
            },
            {
                "root_label": "source_a",
                "parent_label": "source_a",
                "label": "bridge_miss",
                "depth": 1,
                "matched_target": False,
            },
            {
                "root_label": "source_a",
                "parent_label": "bridge_miss",
                "label": "deep_child",
                "depth": 2,
                "matched_target": False,
            },
            {
                "root_label": "source_a",
                "parent_label": "bridge_miss",
                "label": "unused_child",
                "depth": 2,
                "matched_target": False,
            },
        ],
        "matched_labels": ["bridge_hit"],
    }
    assert manifest == {
        "run_name": "algo3_single_row_v1",
        "model": "mistral-small-2603",
        "source_labels": ["source_a"],
        "target_labels": ["bridge_hit", "target_z"],
        "prompt_config": {
            "include_example": True,
            "include_counterexample": False,
        },
        "child_count": 2,
        "max_depth": 2,
    }
    assert summary == actual
    assert len(event_lines) == 2
    assert json.loads(event_lines[0])["event"] == "probe_started"
    assert json.loads(event_lines[1])["event"] == "probe_finished"
    assert "You are a helpful assistant who understands Knowledge Maps." in tree_prompt
    assert "recommend 2 related concept names for each of the names in the input" in tree_prompt
    assert "dictionary format" in tree_prompt
    assert "Here is an example of a desired output for your task." in tree_prompt
    assert "Here is an example of a bad output that we do not want to see." not in tree_prompt
    assert len(chat_client.calls) == 2
