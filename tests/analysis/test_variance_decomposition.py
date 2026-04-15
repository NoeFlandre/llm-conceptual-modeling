"""Tests for deterministic Qwen/Mistral variance decomposition tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.analysis.variance_decomposition import (
    compute_variance_decomposition,
    extract_variance_rows_by_algorithm_and_model,
    generate_variance_decomposition_bundle,
    render_variance_decomposition_table,
)

QWEN = "Qwen/Qwen3.5-9B"
MISTRAL = "mistralai/Ministral-3-8B-Instruct-2512"


def test_extract_variance_rows_by_algorithm_and_model_derives_expected_factors() -> None:
    ledger = {
        "records": [
            _finished_record(
                algorithm="algo1",
                model=QWEN,
                condition_bits="10100",
                condition_label="beam_num_beams_6",
                pair_name="sg1_sg2",
                replication=2,
                accuracy=0.5,
                precision=0.4,
                recall=0.3,
            ),
            _finished_record(
                algorithm="algo2",
                model=MISTRAL,
                condition_bits="101001",
                condition_label="contrastive_penalty_alpha_0.8",
                pair_name="sg2_sg3",
                replication=1,
                accuracy=0.5,
                precision=0.4,
                recall=0.3,
            ),
            _finished_record(
                algorithm="algo3",
                model=QWEN,
                condition_bits="1111",
                condition_label="greedy",
                pair_name="subgraph_1_to_subgraph_3",
                replication=0,
                recall=0.9,
            ),
            {
                "status": "retryable_failed",
                "identity": {
                    "algorithm": "algo1",
                    "model": QWEN,
                    "condition_bits": "00000",
                    "condition_label": "greedy",
                    "pair_name": "sg1_sg2",
                    "replication": 0,
                },
            },
        ]
    }

    rows_by_key = extract_variance_rows_by_algorithm_and_model(ledger)

    algo1_rows = rows_by_key[("algo1", "Qwen")]
    assert len(algo1_rows) == 1
    assert algo1_rows[0]["Example"] == -1
    assert algo1_rows[0]["Explanation"] == 1
    assert algo1_rows[0]["Counterexample"] == 1
    assert algo1_rows[0]["Array/List"] == -1
    assert algo1_rows[0]["Tag/Adjacency"] == -1
    assert algo1_rows[0]["Decoding Family"] == "beam"
    assert algo1_rows[0]["Beam Width"] == 1
    assert algo1_rows[0]["Contrastive Penalty"] == 0

    algo2_rows = rows_by_key[("algo2", "Mistral")]
    assert algo2_rows[0]["Convergence"] == 1
    assert algo2_rows[0]["Decoding Family"] == "contrastive"
    assert algo2_rows[0]["Contrastive Penalty"] == 1

    algo3_rows = rows_by_key[("algo3", "Qwen")]
    assert algo3_rows[0]["Example"] == 1
    assert algo3_rows[0]["Counterexample"] == 1
    assert algo3_rows[0]["Number of Words"] == 1
    assert algo3_rows[0]["Depth"] == 1


def test_compute_variance_decomposition_closes_to_100_for_algo1() -> None:
    frame = pd.DataFrame(_synthetic_rows("algo1", "Qwen"))

    decomposition = compute_variance_decomposition(frame, "algo1", "Qwen")

    accuracy_rows = decomposition[decomposition["metric"] == "accuracy"]
    recall_rows = decomposition[decomposition["metric"] == "recall"]
    precision_rows = decomposition[decomposition["metric"] == "precision"]

    for metric_rows in (accuracy_rows, recall_rows, precision_rows):
        assert pytest.approx(metric_rows["pct_with_error"].sum(), abs=1e-8) == 100.0
        non_error = metric_rows[metric_rows["feature"] != "Error"]
        assert pytest.approx(non_error["pct_without_error"].sum(), abs=1e-8) == 100.0
        error_row = metric_rows[metric_rows["feature"] == "Error"].iloc[0]
        assert error_row["pct_without_error"] == 0.0

    assert "Greedy vs Beam/Contrastive" in set(decomposition["feature"])
    assert "Beam vs Contrastive" in set(decomposition["feature"])
    assert "Beam Width" in set(decomposition["feature"])
    assert "Contrastive Penalty" in set(decomposition["feature"])
    assert "Example & Greedy vs Beam/Contrastive" in set(decomposition["feature"])
    assert "Example & Beam vs Contrastive" in set(decomposition["feature"])
    assert "Error" in set(decomposition["feature"])


def test_render_variance_decomposition_table_for_algo3_is_recall_only() -> None:
    decomposition = pd.DataFrame(
        [
            {
                "algorithm": "algo3",
                "model": "Qwen",
                "feature": "Example",
                "metric": "recall",
                "pct_with_error": 10.0,
                "pct_without_error": 12.5,
                "ss": 1.0,
            },
            {
                "algorithm": "algo3",
                "model": "Qwen",
                "feature": "Error",
                "metric": "recall",
                "pct_with_error": 20.0,
                "pct_without_error": 20.0,
                "ss": 2.0,
            },
            {
                "algorithm": "algo3",
                "model": "Mistral",
                "feature": "Example",
                "metric": "recall",
                "pct_with_error": 5.0,
                "pct_without_error": 5.0,
                "ss": 1.0,
            },
            {
                "algorithm": "algo3",
                "model": "Mistral",
                "feature": "Error",
                "metric": "recall",
                "pct_with_error": 0.0,
                "pct_without_error": 0.0,
                "ss": 0.0,
            },
        ]
    )

    latex = render_variance_decomposition_table("algo3", decomposition)

    assert "\\textbf{recall}" in latex
    assert "\\textbf{accuracy}" not in latex
    assert "\\textbf{precision}" not in latex
    assert "12.50 vs 10.00" in latex
    assert "Error term" in latex


def test_generate_variance_decomposition_bundle_is_deterministic(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    output_root = results_root / "variance_decomposition"
    results_root.mkdir(parents=True, exist_ok=True)
    ledger_path = results_root / "ledger.json"
    ledger_path.write_text(
        json.dumps({"records": list(_synthetic_ledger_records())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    first = generate_variance_decomposition_bundle(results_root, output_root)
    second = generate_variance_decomposition_bundle(results_root, output_root)

    assert first["decomposition_csv"].read_text(
        encoding="utf-8"
    ) == second["decomposition_csv"].read_text(encoding="utf-8")
    for algorithm in ("algo1", "algo2", "algo3"):
        assert first["tables"][algorithm] == second["tables"][algorithm]
        assert (output_root / f"variance_decomposition_{algorithm}.tex").exists()
        algorithm_csv = output_root / f"variance_decomposition_{algorithm}.csv"
        assert algorithm_csv.exists()
        algorithm_frame = pd.read_csv(algorithm_csv)
        assert set(algorithm_frame["algorithm"]) == {algorithm}

    decomposition = pd.read_csv(first["decomposition_csv"])
    for _, group in decomposition.groupby(["algorithm", "model", "metric"]):
        assert pytest.approx(group["pct_with_error"].sum(), abs=1e-8) == 100.0
        non_error = group[group["feature"] != "Error"]
        assert pytest.approx(non_error["pct_without_error"].sum(), abs=1e-8) == 100.0
        error_row = group[group["feature"] == "Error"].iloc[0]
        assert error_row["pct_without_error"] == 0.0
    assert first["combined_table"].parent == output_root


def test_variance_decomposition_bundle_defaults_to_subfolder(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "ledger.json").write_text(
        json.dumps({"records": list(_synthetic_ledger_records())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    bundle = generate_variance_decomposition_bundle(results_root)

    expected_dir = results_root / "variance_decomposition"
    assert bundle["combined_table"].parent == expected_dir
    assert bundle["decomposition_csv"].parent == expected_dir
    assert (expected_dir / "variance_decomposition_algo1.csv").exists()


def _synthetic_ledger_records() -> Iterable[dict[str, object]]:
    for model in (QWEN, MISTRAL):
        for row in _synthetic_rows("algo1", "Qwen" if model == QWEN else "Mistral"):
            yield _record_from_row("algo1", model, row)
        for row in _synthetic_rows("algo2", "Qwen" if model == QWEN else "Mistral"):
            yield _record_from_row("algo2", model, row)
        for row in _synthetic_rows("algo3", "Qwen" if model == QWEN else "Mistral"):
            yield _record_from_row("algo3", model, row)


def _record_from_row(algorithm: str, model: str, row: dict[str, object]) -> dict[str, object]:
    metrics = {"recall": row["recall"]}
    if algorithm != "algo3":
        metrics["accuracy"] = row["accuracy"]
        metrics["precision"] = row["precision"]
    return _finished_record(
        algorithm=algorithm,
        model=model,
        condition_bits=str(row["condition_bits"]),
        condition_label=str(row["condition_label"]),
        pair_name=str(row["pair_name"]),
        replication=int(row["replication"]),
        **metrics,
    )


def _finished_record(
    *,
    algorithm: str,
    model: str,
    condition_bits: str,
    condition_label: str,
    pair_name: str,
    replication: int,
    recall: float,
    accuracy: float | None = None,
    precision: float | None = None,
) -> dict[str, object]:
    metrics: dict[str, float] = {"recall": recall}
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    if precision is not None:
        metrics["precision"] = precision
    return {
        "status": "finished",
        "identity": {
            "algorithm": algorithm,
            "model": model,
            "condition_bits": condition_bits,
            "condition_label": condition_label,
            "pair_name": pair_name,
            "replication": replication,
        },
        "winner": {
            "metrics": metrics,
        },
    }


def _synthetic_rows(algorithm: str, model_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if algorithm == "algo1":
        pair_names = ("sg1_sg2", "sg2_sg3")
        for pair_index, pair_name in enumerate(pair_names):
            for replication in (0, 1):
                for levels in product((-1, 1), repeat=5):
                    explanation, example, counterexample, array_level, adjacency_level = levels
                    for condition_label in (
                        "greedy",
                        "beam_num_beams_2",
                        "beam_num_beams_6",
                        "contrastive_penalty_alpha_0.2",
                        "contrastive_penalty_alpha_0.8",
                    ):
                        decoding = _decoding_basis(condition_label)
                        signal = (
                            6.0 * example
                            + 2.0 * counterexample
                            + 1.5 * explanation
                            + 1.2 * decoding["decoding_algorithm_greedy_vs_rest"]
                            + 2.2 * decoding["beam_width_level"]
                            + 3.5 * example * decoding["decoding_algorithm_beam_vs_contrastive"]
                        )
                        noise = _replicate_noise(pair_index, replication)
                        rows.append(
                            {
                                "condition_bits": _bits_from_levels(levels),
                                "condition_label": condition_label,
                                "pair_name": pair_name,
                                "replication": replication,
                                "accuracy": 50.0 + signal + noise,
                                "recall": 40.0 + (0.8 * signal) + noise,
                                "precision": 60.0 + (1.1 * signal) + noise,
                            }
                        )
    elif algorithm == "algo2":
        pair_names = ("sg1_sg2", "sg2_sg3")
        for pair_index, pair_name in enumerate(pair_names):
            for replication in (0, 1):
                for levels in product((-1, 1), repeat=6):
                    (
                        explanation,
                        example,
                        counterexample,
                        array_level,
                        adjacency_level,
                        convergence,
                    ) = levels
                    for condition_label in (
                        "greedy",
                        "beam_num_beams_2",
                        "beam_num_beams_6",
                        "contrastive_penalty_alpha_0.2",
                        "contrastive_penalty_alpha_0.8",
                    ):
                        decoding = _decoding_basis(condition_label)
                        signal = (
                            4.0 * convergence
                            + 2.0 * explanation
                            + 3.0 * example
                            + 1.2 * counterexample
                            + 1.1 * decoding["decoding_algorithm_beam_vs_contrastive"]
                            + 2.8 * convergence * decoding["contrastive_penalty_level"]
                        )
                        noise = _replicate_noise(pair_index, replication)
                        rows.append(
                            {
                                "condition_bits": _bits_from_levels(levels),
                                "condition_label": condition_label,
                                "pair_name": pair_name,
                                "replication": replication,
                                "accuracy": 55.0 + signal + noise,
                                "recall": 45.0 + (0.7 * signal) + noise,
                                "precision": 65.0 + (1.3 * signal) + noise,
                            }
                        )
    else:
        pair_names = (
            "subgraph_1_to_subgraph_3",
            "subgraph_2_to_subgraph_1",
        )
        for pair_index, pair_name in enumerate(pair_names):
            for replication in (0, 1):
                for example, counterexample, number_of_words, depth in product(
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1)
                ):
                    for condition_label in (
                        "greedy",
                        "beam_num_beams_2",
                        "beam_num_beams_6",
                        "contrastive_penalty_alpha_0.2",
                        "contrastive_penalty_alpha_0.8",
                    ):
                        decoding = _decoding_basis(condition_label)
                        signal = (
                            3.0 * example
                            + 2.0 * counterexample
                            + 4.5 * depth
                            + 1.4 * decoding["decoding_algorithm_greedy_vs_rest"]
                            + 1.8 * number_of_words * decoding["beam_width_level"]
                        )
                        noise = _replicate_noise(pair_index, replication)
                        rows.append(
                            {
                                "condition_bits": _bits_from_levels(
                                    (example, counterexample, number_of_words, depth)
                                ),
                                "condition_label": condition_label,
                                "pair_name": pair_name,
                                "replication": replication,
                                "recall": 35.0 + signal + noise,
                            }
                        )
    return rows


def _replicate_noise(pair_index: int, replication: int) -> float:
    noise_lookup = {
        (0, 0): -0.5,
        (0, 1): 0.5,
        (1, 0): 0.5,
        (1, 1): -0.5,
    }
    return noise_lookup[(pair_index, replication)]


def _bits_from_levels(levels: tuple[int, ...]) -> str:
    return "".join("1" if level == 1 else "0" for level in levels)


def _decoding_basis(condition_label: str) -> dict[str, int]:
    if condition_label == "greedy":
        return {
            "decoding_algorithm_greedy_vs_rest": 4,
            "decoding_algorithm_beam_vs_contrastive": 0,
            "beam_width_level": 0,
            "contrastive_penalty_level": 0,
        }
    if condition_label == "beam_num_beams_2":
        return {
            "decoding_algorithm_greedy_vs_rest": -1,
            "decoding_algorithm_beam_vs_contrastive": 1,
            "beam_width_level": -1,
            "contrastive_penalty_level": 0,
        }
    if condition_label == "beam_num_beams_6":
        return {
            "decoding_algorithm_greedy_vs_rest": -1,
            "decoding_algorithm_beam_vs_contrastive": 1,
            "beam_width_level": 1,
            "contrastive_penalty_level": 0,
        }
    if condition_label == "contrastive_penalty_alpha_0.2":
        return {
            "decoding_algorithm_greedy_vs_rest": -1,
            "decoding_algorithm_beam_vs_contrastive": -1,
            "beam_width_level": 0,
            "contrastive_penalty_level": -1,
        }
    return {
        "decoding_algorithm_greedy_vs_rest": -1,
        "decoding_algorithm_beam_vs_contrastive": -1,
        "beam_width_level": 0,
        "contrastive_penalty_level": 1,
    }
