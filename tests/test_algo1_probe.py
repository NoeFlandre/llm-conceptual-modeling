import json

from llm_conceptual_modeling.algo1.probe import (
    Algo1ProbeSpec,
    run_algo1_probe,
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

        if schema_name == "edge_list":
            return {
                "edges": [
                    {"source": "alpha", "target": "bridge_node"},
                    {"source": "bridge_node", "target": "delta"},
                ]
            }

        return {"votes": ["Y", "N"]}


def test_run_algo1_probe_writes_auditable_artifacts(tmp_path) -> None:
    chat_client = FakeChatClient()
    probe_dir = tmp_path / "algo1_probe"
    spec = Algo1ProbeSpec(
        run_name="algo1_single_row_v1",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        output_dir=probe_dir,
    )

    actual = run_algo1_probe(
        spec=spec,
        chat_client=chat_client,
    )

    manifest = json.loads((probe_dir / "manifest.json").read_text())
    summary = json.loads((probe_dir / "summary.json").read_text())
    event_lines = (probe_dir / "events.jsonl").read_text().splitlines()
    edge_prompt = (probe_dir / "edge_generation_prompt.txt").read_text()
    cove_prompt = (probe_dir / "cove_prompt.txt").read_text()

    assert actual == {
        "run_name": "algo1_single_row_v1",
        "model": "mistral-small-2603",
        "candidate_edges": [
            ["alpha", "bridge_node"],
            ["bridge_node", "delta"],
        ],
        "verified_edges": [["alpha", "bridge_node"]],
    }
    assert manifest == {
        "run_name": "algo1_single_row_v1",
        "model": "mistral-small-2603",
        "subgraph1": [["alpha", "beta"]],
        "subgraph2": [["gamma", "delta"]],
    }
    assert summary == actual
    assert len(event_lines) == 2
    assert json.loads(event_lines[0])["event"] == "probe_started"
    assert json.loads(event_lines[1])["event"] == "probe_finished"
    assert "recommend more links between the two maps" in edge_prompt
    assert "causal relationship exists between the source and target concepts" in cove_prompt
    assert len(chat_client.calls) == 2
