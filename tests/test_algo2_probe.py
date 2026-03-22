import json

from llm_conceptual_modeling.algo2.mistral import Method2PromptConfig
from llm_conceptual_modeling.algo2.probe import (
    Algo2ProbeSpec,
    run_algo2_probe,
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
        call_record = {
            "prompt": prompt,
            "schema_name": schema_name,
        }
        # type: ignore[arg-type]

        self.calls.append(call_record)
        if schema_name == "label_list":
            return {"labels": ["bridge_a", "bridge_b"]}
        return {
            "edges": [
                {"source": "Cholesterol", "target": "Weight loss"},
                {"source": "bridge_b", "target": "beta"},
            ]
        }


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        embeddings_by_label = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "bridge_a": [1.0, 0.0],
            "bridge_b": [0.0, 1.0],
        }
        return {text: embeddings_by_label[text] for text in texts}


def test_run_algo2_probe_writes_auditable_artifacts(tmp_path) -> None:
    chat_client = FakeChatClient()
    embedding_client = FakeEmbeddingClient()
    probe_dir = tmp_path / "algo2_probe"
    spec = Algo2ProbeSpec(
        run_name="algo2_single_row_v1",
        model="mistral-small-2603",
        seed_labels=["alpha", "beta"],
        subgraph1=[("alpha", "beta"), ("beta", "gamma")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=Method2PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=True,
            include_explanation=True,
            include_example=False,
            include_counterexample=False,
        ),
        convergence_threshold=0.01,
        output_dir=probe_dir,
    )

    actual = run_algo2_probe(
        spec=spec,
        chat_client=chat_client,
        embedding_client=embedding_client,
    )

    manifest = json.loads((probe_dir / "manifest.json").read_text())
    summary = json.loads((probe_dir / "summary.json").read_text())
    event_lines = (probe_dir / "events.jsonl").read_text().splitlines()
    label_prompt = (probe_dir / "label_expansion_prompt.txt").read_text()
    edge_prompt = (probe_dir / "edge_suggestion_prompt.txt").read_text()

    assert actual["run_name"] == "algo2_single_row_v1"
    assert actual["model"] == "mistral-small-2603"
    assert actual["expanded_labels"] == ["bridge_a", "bridge_b"]
    assert actual["normalized_edges"] == [
        ["Blood saturated fatty acid level", "Obesity"],
        ["bridge_b", "beta"],
    ]
    assert manifest == {
        "run_name": "algo2_single_row_v1",
        "model": "mistral-small-2603",
        "seed_labels": ["alpha", "beta"],
        "subgraph1": [["alpha", "beta"], ["beta", "gamma"]],
        "subgraph2": [["delta", "epsilon"]],
        "prompt_config": {
            "use_adjacency_notation": True,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": False,
            "include_counterexample": False,
        },
        "convergence_threshold": 0.01,
    }
    assert summary == actual
    assert len(event_lines) == 2
    assert json.loads(event_lines[0])["event"] == "probe_started"
    assert json.loads(event_lines[1])["event"] == "probe_finished"
    assert "recommend 5 new related concept names" in label_prompt
    assert "Knowledge map 1: {'nodes': ['alpha', 'beta', 'gamma']" in label_prompt
    assert "suggest edges that directly link concepts" in edge_prompt
    assert "Knowledge map 1: {'nodes': ['alpha', 'beta', 'gamma']" in edge_prompt
    assert len(chat_client.calls) == 3
