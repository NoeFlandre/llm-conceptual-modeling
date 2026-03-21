import json
from pathlib import Path

from llm_conceptual_modeling.algo3.experiment import run_algo3_experiment
from llm_conceptual_modeling.algo3.mistral import Method3PromptConfig
from llm_conceptual_modeling.algo3.probe import Algo3ProbeSpec


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
                "source_a": ["bridge_hit", "bridge_miss"],
                "bridge_miss": ["deep_child", "unused_child"],
            }
        }


def test_run_algo3_experiment_executes_specs_and_returns_summaries(tmp_path: Path) -> None:
    chat_client = FakeChatClient()
    prompt_config = Method3PromptConfig(
        include_example=False,
        include_counterexample=False,
    )
    first_spec = Algo3ProbeSpec(
        run_name="run_a",
        model="mistral-small-2603",
        source_labels=["source_a"],
        target_labels=["bridge_hit", "target_z"],
        prompt_config=prompt_config,
        child_count=2,
        max_depth=2,
        output_dir=tmp_path / "run_a",
    )
    second_spec = Algo3ProbeSpec(
        run_name="run_b",
        model="mistral-small-2603",
        source_labels=["source_a"],
        target_labels=["bridge_hit", "target_z"],
        prompt_config=prompt_config,
        child_count=2,
        max_depth=2,
        output_dir=tmp_path / "run_b",
    )

    actual = run_algo3_experiment(
        specs=[first_spec, second_spec],
        chat_client=chat_client,
    )

    assert [summary["run_name"] for summary in actual] == ["run_a", "run_b"]
    assert actual[0]["matched_labels"] == ["bridge_hit"]
    assert actual[1]["matched_labels"] == ["bridge_hit"]
    assert len(chat_client.calls) == 4

    first_summary = json.loads((tmp_path / "run_a" / "summary.json").read_text())
    second_summary = json.loads((tmp_path / "run_b" / "summary.json").read_text())

    assert first_summary["run_name"] == "run_a"
    assert second_summary["run_name"] == "run_b"
