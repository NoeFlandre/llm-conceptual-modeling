import json
from pathlib import Path

from llm_conceptual_modeling.algo1.experiment import run_algo1_experiment
from llm_conceptual_modeling.algo1.mistral import Method1PromptConfig
from llm_conceptual_modeling.algo1.probe import Algo1ProbeSpec


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
        call_record = {
            "prompt": prompt,
            "schema_name": schema_name,
        }
        self.calls.append(call_record)

        if schema_name == "edge_list":
            return {
                "edges": [
                    {"source": "alpha", "target": "bridge_node"},
                    {"source": "bridge_node", "target": "delta"},
                ]
            }

        return {"votes": ["Y", "N"]}


def test_run_algo1_experiment_executes_specs_and_returns_summaries(tmp_path: Path) -> None:
    chat_client = FakeChatClient()
    prompt_config = Method1PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )
    first_spec = Algo1ProbeSpec(
        run_name="run_a",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        prompt_config=prompt_config,
        output_dir=tmp_path / "run_a",
    )
    second_spec = Algo1ProbeSpec(
        run_name="run_b",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        prompt_config=prompt_config,
        output_dir=tmp_path / "run_b",
    )

    actual = run_algo1_experiment(
        specs=[first_spec, second_spec],
        chat_client=chat_client,
    )

    assert [summary["run_name"] for summary in actual] == ["run_a", "run_b"]
    assert actual[0]["verified_edges"] == [["alpha", "bridge_node"]]
    assert actual[1]["verified_edges"] == [["alpha", "bridge_node"]]
    assert len(chat_client.calls) == 4

    first_summary = json.loads((tmp_path / "run_a" / "summary.json").read_text())
    second_summary = json.loads((tmp_path / "run_b" / "summary.json").read_text())

    assert first_summary["run_name"] == "run_a"
    assert second_summary["run_name"] == "run_b"


def test_run_algo1_experiment_skips_failed_specs_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    call_order: list[str] = []

    def fake_run_probe(*, spec, chat_client):
        call_order.append(spec.run_name)
        if spec.run_name == "run_a":
            raise RuntimeError("transient failure")
        return {"run_name": spec.run_name, "verified_edges": [["alpha", "bridge_node"]]}

    monkeypatch.setattr(
        "llm_conceptual_modeling.algo1.experiment.run_algo1_probe",
        fake_run_probe,
    )

    prompt_config = Method1PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )
    first_spec = Algo1ProbeSpec(
        run_name="run_a",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        prompt_config=prompt_config,
        output_dir=tmp_path / "run_a",
    )
    second_spec = Algo1ProbeSpec(
        run_name="run_b",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        prompt_config=prompt_config,
        output_dir=tmp_path / "run_b",
    )

    actual = run_algo1_experiment(
        specs=[first_spec, second_spec],
        chat_client=FakeChatClient(),
    )

    assert [summary["run_name"] for summary in actual] == ["run_b"]
    assert call_order == ["run_a", "run_b"]
