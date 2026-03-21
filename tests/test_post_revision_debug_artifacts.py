import json

from llm_conceptual_modeling.post_revision_debug.artifacts import (
    append_jsonl_event,
    build_probe_result_record,
)
from llm_conceptual_modeling.post_revision_debug.run_context import ProbeRunContext


def test_append_jsonl_event_writes_one_json_object_per_line(tmp_path) -> None:
    output_path = tmp_path / "events.jsonl"

    append_jsonl_event(
        output_path,
        {"event": "probe_started", "model": "mistral-small-2603"},
        timestamp="2026-03-21T12:00:00Z",
    )
    append_jsonl_event(
        output_path,
        {"event": "probe_finished", "model": "mistral-small-2603"},
        timestamp="2026-03-21T12:00:01Z",
    )

    lines = output_path.read_text().splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "timestamp": "2026-03-21T12:00:00Z",
        "event": "probe_started",
        "model": "mistral-small-2603",
    }
    assert json.loads(lines[1]) == {
        "timestamp": "2026-03-21T12:00:01Z",
        "event": "probe_finished",
        "model": "mistral-small-2603",
    }


def test_build_probe_result_record_tracks_historical_delta() -> None:
    actual = build_probe_result_record(
        algorithm="algo3",
        row_index=0,
        model="mistral-small-2603",
        metric_name="recall",
        historical_score=0.0,
        probe_score=0.36363636363636365,
        parsed_edge_count=10,
    )

    assert actual["algorithm"] == "algo3"
    assert actual["row_index"] == 0
    assert actual["model"] == "mistral-small-2603"
    assert actual["metric_name"] == "recall"
    assert actual["historical_score"] == 0.0
    assert actual["probe_score"] == 0.36363636363636365
    assert actual["score_delta"] == 0.36363636363636365
    assert actual["parsed_edge_count"] == 10


def test_probe_run_context_tracks_state_and_log(tmp_path) -> None:
    context = ProbeRunContext(
        output_dir=tmp_path / "run",
        run_name="algo1_row0",
        algorithm="algo1",
    )

    context.record_manifest({"run_name": "algo1_row0"})
    context.record_prompt("prompt.txt", "hello world", stage="prompt_written")
    context.record_checkpoint(
        "checkpoint.json",
        {"status": "done"},
        stage="checkpoint_written",
    )
    context.mark_stage_complete("probe_finished", details={"summary_path": "summary.json"})

    state = json.loads((tmp_path / "run" / "state.json").read_text())
    run_log = (tmp_path / "run" / "run.log").read_text().splitlines()

    assert state["algorithm"] == "algo1"
    assert state["run_name"] == "algo1_row0"
    assert state["completed_stages"] == [
        "manifest_written",
        "prompt_written",
        "checkpoint_written",
        "probe_finished",
    ]
    assert "manifest written" in run_log[0]
    assert "prompt written to prompt.txt" in run_log[1]
    assert "checkpoint written to checkpoint.json" in run_log[2]
