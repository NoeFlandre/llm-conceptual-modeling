import json
from pathlib import Path

import pytest

from llm_conceptual_modeling.hf_batch_utils import RecordingChatClient


class _FailingChatClient:
    def __init__(self) -> None:
        self.last_call_metrics = {"duration_seconds": 1.0}
        self.last_failed_response_text = "assistant\n[(no node)]"

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        _ = (prompt, schema_name, schema)
        raise ValueError("Could not parse tuple content: no node")


def test_recording_chat_client_persists_failed_raw_response(tmp_path: Path) -> None:
    persist_path = tmp_path / "raw_response.json"
    active_stage_path = tmp_path / "active_stage.json"
    client = RecordingChatClient(
        _FailingChatClient(),
        persist_path=persist_path,
        active_stage_path=active_stage_path,
        active_stage_context={"pair_name": "sg3_sg1"},
    )

    with pytest.raises(ValueError, match="Could not parse tuple content: no node"):
        client.complete_json(
            prompt="prompt text",
            schema_name="edge_list",
            schema={"type": "object"},
        )

    records = json.loads(persist_path.read_text(encoding="utf-8"))
    assert records == [
        {
            "error": "Could not parse tuple content: no node",
            "metrics": {"duration_seconds": 1.0},
            "prompt": "prompt text",
            "raw_text": "assistant\n[(no node)]",
            "schema_name": "edge_list",
        }
    ]
    active_stage = json.loads(active_stage_path.read_text(encoding="utf-8"))
    assert active_stage["status"] == "failed"
    assert active_stage["raw_text"] == "assistant\n[(no node)]"
