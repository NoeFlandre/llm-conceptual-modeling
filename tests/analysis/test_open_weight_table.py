from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.analysis.open_weight_table import (
    MetricSummary,
    aggregate_open_weight_ablation,
    read_ledger_counts,
)


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_aggregate_open_weight_ablation_filters_and_averages(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    runs_root = results_root / "runs"
    _write_summary(
        runs_root / "algo1/allenai__Olmo-3-7B-Instruct/greedy/sg1_sg2/00001/rep_00/summary.json",
        {
            "status": "finished",
            "algorithm": "algo1",
            "model": "allenai/Olmo-3-7B-Instruct",
            "condition_label": "greedy",
            "accuracy": 0.5,
            "precision": 0.75,
            "recall": 0.5,
            "f1": 0.6,
        },
    )
    _write_summary(
        runs_root / "algo1/allenai__Olmo-3-7B-Instruct/greedy/sg1_sg2/00002/rep_00/summary.json",
        {
            "status": "finished",
            "algorithm": "algo1",
            "model": "allenai/Olmo-3-7B-Instruct",
            "condition_label": "greedy",
            "accuracy": 0.7,
            "precision": 0.85,
            "recall": 0.7,
            "f1": 0.77,
        },
    )
    _write_summary(
        runs_root / "algo1/allenai__Olmo-3-7B-Instruct/greedy/sg1_sg2/00003/rep_00/summary.json",
        {
            "status": "failed",
            "algorithm": "algo1",
            "model": "allenai/Olmo-3-7B-Instruct",
            "condition_label": "greedy",
            "accuracy": 0.9,
        },
    )
    _write_summary(
        runs_root
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "contrastive_penalty_alpha_0.2"
        / "sg1_sg2"
        / "00004"
        / "rep_00"
        / "summary.json",
        {
            "status": "finished",
            "algorithm": "algo1",
            "model": "allenai/Olmo-3-7B-Instruct",
            "condition_label": "contrastive_penalty_alpha_0.2",
            "accuracy": 0.1,
            "precision": 0.2,
            "recall": 0.1,
            "f1": 0.12,
        },
    )

    metrics = aggregate_open_weight_ablation(results_root)
    summary = metrics[("algo1", "greedy", "OLMo")]

    assert summary == MetricSummary(
        runs=2,
        accuracy=0.6,
        precision=0.8,
        recall=0.6,
        f1=0.685,
    )


def test_read_ledger_counts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    ledger_path = results_root / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "finished_count": 12,
                "expected_total_runs": 25,
            }
        ),
        encoding="utf-8",
    )

    finished, expected = read_ledger_counts(results_root)
    assert finished == 12
    assert expected == 25


def test_aggregate_open_weight_ablation_ignores_boolean_metrics(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    runs_root = results_root / "runs"
    _write_summary(
        runs_root / "algo1/allenai__Olmo-3-7B-Instruct/greedy/sg1_sg2/00001/rep_00/summary.json",
        {
            "status": "finished",
            "algorithm": "algo1",
            "model": "allenai/Olmo-3-7B-Instruct",
            "condition_label": "greedy",
            "accuracy": True,
            "precision": 0.75,
            "recall": False,
            "f1": "0.6",
        },
    )

    metrics = aggregate_open_weight_ablation(results_root)
    summary = metrics[("algo1", "greedy", "OLMo")]

    assert summary == MetricSummary(
        runs=1,
        accuracy=None,
        precision=0.75,
        recall=None,
        f1=0.6,
    )


def test_read_ledger_counts_ignores_boolean_counts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    ledger_path = results_root / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "finished_count": True,
                "expected_total_runs": "25",
            }
        ),
        encoding="utf-8",
    )

    finished, expected = read_ledger_counts(results_root)
    assert finished is None
    assert expected == 25
