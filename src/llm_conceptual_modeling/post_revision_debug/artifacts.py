import json
from datetime import UTC, datetime
from pathlib import Path

from llm_conceptual_modeling.common.types import PathLike


def append_jsonl_event(
    output_path: PathLike,
    event: dict[str, object],
    *,
    timestamp: str | None = None,
) -> None:
    event_timestamp = timestamp or datetime.now(UTC).isoformat()
    event_record = {
        "timestamp": event_timestamp,
        **event,
    }
    line = json.dumps(event_record, sort_keys=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def build_probe_result_record(
    *,
    algorithm: str,
    row_index: int,
    model: str,
    metric_name: str,
    historical_score: float,
    probe_score: float,
    parsed_edge_count: int,
) -> dict[str, object]:
    score_delta = probe_score - historical_score
    return {
        "algorithm": algorithm,
        "row_index": row_index,
        "model": model,
        "metric_name": metric_name,
        "historical_score": historical_score,
        "probe_score": probe_score,
        "score_delta": score_delta,
        "parsed_edge_count": parsed_edge_count,
    }
