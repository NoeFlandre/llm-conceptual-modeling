import json

from llm_conceptual_modeling.algo1.probe import (
    Algo1ProbeSpec,
    Method1PromptConfig,
    _edges_to_json_compatible,
    run_algo1_probe,
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
        subgraph1=[("alpha", "beta"), ("beta", "gamma")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=True,
            include_explanation=True,
            include_example=False,
            include_counterexample=False,
        ),
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
        "subgraph1": [["alpha", "beta"], ["beta", "gamma"]],
        "subgraph2": [["delta", "epsilon"]],
        "prompt_config": {
            "use_adjacency_notation": True,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": False,
            "include_counterexample": False,
        },
    }
    assert summary == actual
    assert len(event_lines) == 2
    assert json.loads(event_lines[0])["event"] == "probe_started"
    assert json.loads(event_lines[1])["event"] == "probe_finished"
    assert "recommend more links between the two maps" in edge_prompt
    assert "Knowledge map 1: {'nodes': ['alpha', 'beta', 'gamma']" in edge_prompt
    assert "causal relationship exists between the source and target concepts" in cove_prompt
    assert len(chat_client.calls) == 2


def test_run_algo1_probe_can_resume_from_cached_summary(tmp_path) -> None:
    chat_client = FakeChatClient()
    probe_dir = tmp_path / "algo1_probe"
    spec = Algo1ProbeSpec(
        run_name="algo1_single_row_v1",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=True,
            include_explanation=True,
            include_example=False,
            include_counterexample=False,
        ),
        output_dir=probe_dir,
    )

    initial_summary = run_algo1_probe(
        spec=spec,
        chat_client=chat_client,
    )

    class FailingChatClient:
        def complete_json(self, *, prompt: str, schema_name: str, schema: dict[str, object]):
            raise AssertionError("resume should not call the provider again")

    resumed_spec = Algo1ProbeSpec(
        run_name="algo1_single_row_v1",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=True,
            include_explanation=True,
            include_example=False,
            include_counterexample=False,
        ),
        output_dir=probe_dir,
        resume=True,
    )

    resumed_summary = run_algo1_probe(
        spec=resumed_spec,
        chat_client=FailingChatClient(),
    )

    assert resumed_summary == initial_summary


def test_run_algo1_probe_records_errors(tmp_path) -> None:
    class FailingChatClient:
        def complete_json(
            self,
            *,
            prompt: str,
            schema_name: str,
            schema: dict[str, object],
        ) -> dict[str, object]:
            raise RuntimeError("provider unavailable")

    probe_dir = tmp_path / "algo1_probe"
    spec = Algo1ProbeSpec(
        run_name="algo1_single_row_v1",
        model="mistral-small-2603",
        subgraph1=[("alpha", "beta")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=True,
            include_explanation=True,
            include_example=False,
            include_counterexample=False,
        ),
        output_dir=probe_dir,
    )

    try:
        run_algo1_probe(
            spec=spec,
            chat_client=FailingChatClient(),
        )
    except RuntimeError as error:
        assert str(error) == "provider unavailable"
    else:
        raise AssertionError("expected RuntimeError")

    error_record = json.loads((probe_dir / "error.json").read_text())

    assert error_record["error_type"] == "RuntimeError"
    assert error_record["error_message"] == "provider unavailable"


def test_edges_to_json_compatible_tolerates_scalar_entries() -> None:
    assert _edges_to_json_compatible(["alpha", ("beta", "gamma")]) == [
        ["alpha"],
        ["beta", "gamma"],
    ]
