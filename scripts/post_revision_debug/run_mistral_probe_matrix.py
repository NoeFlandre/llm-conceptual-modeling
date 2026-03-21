import argparse
import json
import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pandas as pd

from llm_conceptual_modeling.post_revision_debug.artifacts import (
    append_jsonl_event,
    build_probe_result_record,
)
from llm_conceptual_modeling.post_revision_debug.mistral_probe import (
    extract_edge_list_from_chat_content,
    score_algo3_row,
    score_connection_row,
)


@dataclass(frozen=True)
class ProbeSpec:
    algorithm: str
    raw_path: str
    evaluated_path: str
    row_index: int
    historical_model: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
    )
    parser.add_argument(
        "--run-name",
        default=None,
    )
    parser.add_argument(
        "--algo1-row",
        action="append",
        dest="algo1_rows",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--algo2-row",
        action="append",
        dest="algo2_rows",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--algo3-row",
        action="append",
        dest="algo3_rows",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    args = parser.parse_args()

    models = args.models or [
        "mistral-small-2506",
        "mistral-small-2603",
        "mistral-medium-2508",
    ]
    api_key = os.environ["MISTRAL_API_KEY"]
    run_name = args.run_name or datetime.now(UTC).strftime("matrix_%Y%m%dT%H%M%SZ")
    run_dir = (
        Path("data/analysis_artifacts/post_revision_debug/mistral/2026-03-21") / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _build_logger(run_dir / "run.log")
    events_path = run_dir / "events.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "probe_results.csv"
    grouped_summary_path = run_dir / "probe_summary_by_algorithm_and_model.csv"
    findings_path = run_dir / "findings.md"

    probe_specs = _default_probe_specs(
        algo1_rows=args.algo1_rows,
        algo2_rows=args.algo2_rows,
        algo3_rows=args.algo3_rows,
    )
    manifest = {
        "run_name": run_name,
        "models": models,
        "resume": args.resume,
        "requested_rows": {
            "algo1": args.algo1_rows,
            "algo2": args.algo2_rows,
            "algo3": args.algo3_rows,
        },
        "probe_specs": [asdict(spec) for spec in probe_specs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    append_jsonl_event(events_path, {"event": "run_started", "run_name": run_name})
    logger.info("Run started: %s", run_name)

    result_records: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []
    for probe_spec in probe_specs:
        raw_frame = pd.read_csv(probe_spec.raw_path)
        evaluated_frame = pd.read_csv(probe_spec.evaluated_path)
        raw_row = raw_frame.iloc[probe_spec.row_index]
        evaluated_row = evaluated_frame.iloc[probe_spec.row_index]

        prompt = _build_prompt(probe_spec.algorithm, raw_row)
        probe_dir = run_dir / probe_spec.algorithm / f"row_{probe_spec.row_index}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "prompt.txt").write_text(prompt)

        logger.info(
            "Probe started: algorithm=%s row_index=%s historical_model=%s",
            probe_spec.algorithm,
            probe_spec.row_index,
            probe_spec.historical_model,
        )
        append_jsonl_event(
            events_path,
            {
                "event": "probe_started",
                "algorithm": probe_spec.algorithm,
                "row_index": probe_spec.row_index,
                "historical_model": probe_spec.historical_model,
            },
        )

        for model in models:
            try:
                logger.info(
                    "Model call started: algorithm=%s row_index=%s model=%s",
                    probe_spec.algorithm,
                    probe_spec.row_index,
                    model,
                )
                append_jsonl_event(
                    events_path,
                    {
                        "event": "model_call_started",
                        "algorithm": probe_spec.algorithm,
                        "row_index": probe_spec.row_index,
                        "model": model,
                    },
                )
                response_path = probe_dir / f"{model}_response.json"
                response_data = _load_existing_response(
                    response_path=response_path,
                    resume=args.resume,
                )
                response_source = "cache" if response_data is not None else "live"
                if response_data is None:
                    response_data = _call_mistral_with_retry(
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                        logger=logger,
                    )
                    response_path.write_text(json.dumps(response_data, indent=2))
                else:
                    logger.info(
                        "Model response loaded from cache: algorithm=%s row_index=%s model=%s",
                        probe_spec.algorithm,
                        probe_spec.row_index,
                        model,
                    )
                    append_jsonl_event(
                        events_path,
                        {
                            "event": "model_response_loaded",
                            "algorithm": probe_spec.algorithm,
                            "row_index": probe_spec.row_index,
                            "model": model,
                            "response_source": response_source,
                        },
                    )
                content = response_data["choices"][0]["message"]["content"]
                edges = extract_edge_list_from_chat_content(content)

                if probe_spec.algorithm == "algo3":
                    probe_metrics = score_algo3_row(raw_row, content)
                    historical_metrics = {"recall": float(evaluated_row["Recall"])}
                else:
                    literal_content = repr(edges)
                    probe_metrics = score_connection_row(raw_row, literal_content)
                    historical_metrics = {
                        "accuracy": float(evaluated_row["accuracy"]),
                        "recall": float(evaluated_row["recall"]),
                        "precision": float(evaluated_row["precision"]),
                    }

                for metric_name, historical_score in historical_metrics.items():
                    probe_score = float(probe_metrics[metric_name])
                    result_record = build_probe_result_record(
                        algorithm=probe_spec.algorithm,
                        row_index=probe_spec.row_index,
                        model=model,
                        metric_name=metric_name,
                        historical_score=historical_score,
                        probe_score=probe_score,
                        parsed_edge_count=len(edges),
                    )
                    result_record["historical_model"] = probe_spec.historical_model
                    result_record["response_source"] = response_source
                    result_records.append(result_record)

                logger.info(
                    "Model call finished: algorithm=%s row_index=%s model=%s parsed_edge_count=%s",
                    probe_spec.algorithm,
                    probe_spec.row_index,
                    model,
                    len(edges),
                )
                append_jsonl_event(
                    events_path,
                    {
                        "event": "model_call_finished",
                        "algorithm": probe_spec.algorithm,
                        "row_index": probe_spec.row_index,
                        "model": model,
                        "parsed_edge_count": len(edges),
                    },
                )
            except Exception as error:
                failure_record = _build_model_failure_record(
                    algorithm=probe_spec.algorithm,
                    row_index=probe_spec.row_index,
                    model=model,
                    historical_model=probe_spec.historical_model,
                    error=error,
                )
                failure_records.append(failure_record)
                logger.exception(
                    "Model call failed: algorithm=%s row_index=%s model=%s",
                    probe_spec.algorithm,
                    probe_spec.row_index,
                    model,
                )
                append_jsonl_event(
                    events_path,
                    {
                        "event": "model_call_failed",
                        **failure_record,
                    },
                )
                continue

    result_frame = pd.DataFrame(result_records)
    result_frame.to_csv(summary_path, index=False)

    grouped_frame = (
        result_frame.groupby(["algorithm", "model", "metric_name"], dropna=False)
        .agg(
            row_count=("row_index", "count"),
            mean_historical_score=("historical_score", "mean"),
            mean_probe_score=("probe_score", "mean"),
            mean_score_delta=("score_delta", "mean"),
            mean_parsed_edge_count=("parsed_edge_count", "mean"),
        )
        .reset_index()
    )
    grouped_frame.to_csv(grouped_summary_path, index=False)
    if failure_records:
        failure_frame = pd.DataFrame(failure_records)
        failure_frame.to_csv(run_dir / "model_failures.csv", index=False)
    findings_text = _build_findings_text(grouped_frame)
    findings_path.write_text(findings_text)

    logger.info("Run finished: %s", run_name)
    append_jsonl_event(events_path, {"event": "run_finished", "run_name": run_name})
    return 0


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("post_revision_debug.mistral")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _default_probe_specs(
    *,
    algo1_rows: list[int],
    algo2_rows: list[int],
    algo3_rows: list[int],
) -> list[ProbeSpec]:
    probe_specs: list[ProbeSpec] = []
    probe_specs.extend(
        _build_probe_specs_for_algorithm(
            algorithm="algo1",
            raw_path="data/results/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv",
            evaluated_path="data/results/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
            row_indexes=algo1_rows or [0, 80],
        )
    )
    probe_specs.extend(
        _build_probe_specs_for_algorithm(
            algorithm="algo2",
            raw_path="data/results/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv",
            evaluated_path="data/results/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv",
            row_indexes=algo2_rows or [0, 160],
        )
    )
    probe_specs.extend(
        _build_probe_specs_for_algorithm(
            algorithm="algo3",
            raw_path="data/results/algo3/gpt-5/raw/method3_results_gpt5.csv",
            evaluated_path="data/results/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv",
            row_indexes=algo3_rows or [0, 1],
        )
    )
    return probe_specs


def _build_probe_specs_for_algorithm(
    *,
    algorithm: str,
    raw_path: str,
    evaluated_path: str,
    row_indexes: list[int],
) -> list[ProbeSpec]:
    return [
        ProbeSpec(
            algorithm=algorithm,
            raw_path=raw_path,
            evaluated_path=evaluated_path,
            row_index=row_index,
            historical_model="gpt-5",
        )
        for row_index in row_indexes
    ]


def _build_prompt(algorithm: str, row: pd.Series) -> str:
    if algorithm == "algo3":
        source_graph = str(row["Source Graph"])
        target_graph = str(row["Target Graph"])
        mother_graph = str(row["Mother Graph"])
        prompt = (
            "Given a source graph, a target graph, and the full mother graph, "
            "propose additional edges that help connect source and target concepts. "
            "Use only exact node names that already appear in the provided graphs. "
            "Do not invent new concept names. "
            "Return a JSON object matching the requested schema. "
            "Prefer edges that improve connectivity between the two subgraphs.\n\n"
            f"Source graph: {source_graph}\n\n"
            f"Target graph: {target_graph}\n\n"
            f"Mother graph: {mother_graph}\n"
        )
        return prompt

    subgraph_1 = str(row["subgraph1"])
    subgraph_2 = str(row["subgraph2"])
    graph = str(row["graph"])
    prompt = (
        "Given two knowledge maps and the full mother graph, recommend a small, "
        "high-precision set of additional edges between the two maps. "
        "Use only exact node names that already appear in the provided graphs. "
        "Do not invent new concept names. "
        "Return a JSON object matching the requested schema. "
        "Prefer fewer, well-supported edges over many speculative ones.\n\n"
        f"Knowledge map 1: {subgraph_1}\n\n"
        f"Knowledge map 2: {subgraph_2}\n\n"
        f"Mother graph: {graph}\n"
    )
    return prompt


def _load_existing_response(
    *,
    response_path: Path,
    resume: bool,
) -> dict[str, Any] | None:
    if not resume:
        return None
    if not response_path.exists():
        return None

    response_text = response_path.read_text()
    parsed_response: dict[str, Any] = json.loads(response_text)
    return parsed_response


def _call_mistral_with_retry(
    *,
    api_key: str,
    model: str,
    prompt: str,
    logger: logging.Logger | None,
    max_attempts: int = 8,
    base_delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay_seconds: float = 60.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "edge_list",
                "schema": {
                    "type": "object",
                    "properties": {
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                },
                                "required": ["source", "target"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["edges"],
                    "additionalProperties": False,
                },
            },
        },
    }
    request = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    attempt = 1
    delay_seconds = base_delay_seconds
    while True:
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_text = response.read().decode("utf-8")
            response_data: dict[str, Any] = json.loads(response_text)
            return response_data
        except (HTTPError, URLError) as error:
            retryable = isinstance(error, URLError) or error.code == 429
            if not retryable or attempt >= max_attempts:
                raise
            if logger is not None:
                logger.warning(
                    "Transient Mistral transport error; retrying attempt=%s model=%s error_type=%s",
                    attempt,
                    model,
                    type(error).__name__,
                )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * backoff_factor, max_delay_seconds)
            attempt += 1


def _call_mistral(*, api_key: str, model: str, prompt: str) -> dict[str, Any]:
    return _call_mistral_with_retry(
        api_key=api_key,
        model=model,
        prompt=prompt,
        logger=None,
    )


def _build_model_failure_record(
    *,
    algorithm: str,
    row_index: int,
    model: str,
    historical_model: str,
    error: Exception,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "row_index": row_index,
        "model": model,
        "historical_model": historical_model,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _build_findings_text(grouped_frame: pd.DataFrame) -> str:
    lines = [
        "# Mistral Probe Findings",
        "",
        "These findings summarize the current post-revision probe matrix.",
        "",
    ]
    for _, row in grouped_frame.sort_values(
        ["algorithm", "metric_name", "mean_score_delta"],
        ascending=[True, True, False],
    ).iterrows():
        line = (
            f"- {row['algorithm']} {row['metric_name']} {row['model']}: "
            f"historical_mean={row['mean_historical_score']:.6f}, "
            f"probe_mean={row['mean_probe_score']:.6f}, "
            f"delta={row['mean_score_delta']:.6f}, "
            f"mean_parsed_edge_count={row['mean_parsed_edge_count']:.2f}"
        )
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
