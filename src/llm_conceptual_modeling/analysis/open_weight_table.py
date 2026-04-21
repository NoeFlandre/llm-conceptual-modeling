from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MODEL_LABELS = {
    "allenai/Olmo-3-7B-Instruct": "OLMo",
    "Qwen/Qwen3.5-9B": "Qwen",
    "mistralai/Ministral-3-8B-Instruct-2512": "Mistral",
}

DECODING_LABELS = (
    "greedy",
    "beam_num_beams_2",
    "beam_num_beams_6",
    "contrastive_penalty_alpha_0.2",
    "contrastive_penalty_alpha_0.8",
)

ALGORITHMS = ("algo1", "algo2", "algo3")
MODELS = ("OLMo", "Qwen", "Mistral")


@dataclass(frozen=True)
class MetricSummary:
    runs: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None


def aggregate_open_weight_ablation(results_root: Path) -> dict[tuple[str, str, str], MetricSummary]:
    rows = _load_finished_rows(results_root)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["algorithm"]),
            str(row["condition_label"]),
            MODEL_LABELS[str(row["model"])],
        )
        grouped.setdefault(key, []).append(row)

    metrics: dict[tuple[str, str, str], MetricSummary] = {}
    for key, group in grouped.items():
        accuracy_vals = _float_values(group, "accuracy")
        precision_vals = _float_values(group, "precision")
        recall_vals = _float_values(group, "recall")
        f1_vals = _float_values(group, "f1")
        metrics[key] = MetricSummary(
            runs=len(group),
            accuracy=_mean(accuracy_vals),
            precision=_mean(precision_vals),
            recall=_mean(recall_vals),
            f1=_mean(f1_vals),
        )

    for algorithm in ALGORITHMS:
        for condition in DECODING_LABELS:
            for model in MODELS:
                metrics.setdefault(
                    (algorithm, condition, model),
                    MetricSummary(runs=0, accuracy=None, precision=None, recall=None, f1=None),
                )

    return metrics


def read_ledger_counts(results_root: Path) -> tuple[int | None, int | None]:
    ledger_path = results_root / "ledger.json"
    if not ledger_path.exists():
        return None, None
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    finished = _coerce_int(ledger.get("finished_count"))
    expected = _coerce_int(ledger.get("expected_total_runs"))
    return finished, expected


def _load_finished_rows(results_root: Path) -> list[dict[str, object]]:
    runs_root = results_root / "runs"
    if not runs_root.exists():
        return []
    rows: list[dict[str, object]] = []
    for summary_path in runs_root.rglob("summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("status") != "finished":
            continue
        if payload.get("condition_label") not in DECODING_LABELS:
            continue
        if payload.get("algorithm") not in ALGORITHMS:
            continue
        if payload.get("model") not in MODEL_LABELS:
            continue
        rows.append(payload)
    return rows


def _float_values(rows: Iterable[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            values.append(float(raw))
            continue
        if not isinstance(raw, str):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None
