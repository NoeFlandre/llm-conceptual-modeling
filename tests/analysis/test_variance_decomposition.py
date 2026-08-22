"""Tests for deterministic Qwen/Mistral variance decomposition tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from llm_conceptual_modeling.analysis import variance_decomposition as variance_module
from llm_conceptual_modeling.analysis.variance_decomposition import (
    _assert_map_extension_recall_matches_batch_summary,
    _binary_contrast_levels,
    _build_graph_recall_summary_row,
    _build_open_weight_summary_row,
    _build_variance_row,
    _center_metric_values,
    _compute_decomposition_rows,
    _compute_metric_decomposition_rows,
    _extract_metric_row,
    _generate_map_extension_bundle,
    _generate_standard_variance_bundle,
    _group_key_tuple,
    _group_open_weight_summary_frame,
    _integer_value,
    _load_batch_summary_record,
    _load_batch_summary_records,
    _load_batch_summary_source_frame,
    _load_map_extension_batch_summary_frame,
    _load_map_extension_evaluated_frame,
    _load_variance_source_frame,
    _materialized_artifact_candidates,
    _normalize_map_extension_frame,
    _numeric_metric,
    _prepare_map_recall_frame,
    _prepare_open_weight_summary_frame,
    _read_raw_row,
    _resolve_algorithm_and_model,
    _resolve_condition_label,
    _resolve_materialized_artifact_path,
    _variance_record_context,
    _VarianceRecordContext,
    _write_map_extension_variance_decomposition,
    _write_map_specific_decompositions,
    build_map_recall_summary,
    build_open_weight_map_extension_summary,
    compute_variance_decomposition,
    extract_variance_rows_by_algorithm_and_model,
    generate_variance_decomposition_bundle,
    render_variance_decomposition_table,
)

QWEN = "Qwen/Qwen3.5-9B"
MISTRAL = "mistralai/Ministral-3-8B-Instruct-2512"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (True, None), (3, 3.0), ("3.5", 3.5), ("not a number", None)],
)
def test_numeric_metric_coerces_supported_values(value: object, expected: float | None) -> None:
    assert _numeric_metric(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (False, None), (3, 3), ("3", 3), ("not an integer", None)],
)
def test_integer_value_coerces_supported_values(value: object, expected: int | None) -> None:
    assert _integer_value(value) == expected


def test_resolve_materialized_artifact_path_handles_relative_and_existing_paths(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    output_root.mkdir()
    relative_path = Path("runs") / "row.json"

    assert _resolve_materialized_artifact_path(
        results_root=output_root,
        artifact_path=relative_path,
    ) == output_root / relative_path

    existing_path = tmp_path / "existing.json"
    existing_path.write_text("{}", encoding="utf-8")
    assert _resolve_materialized_artifact_path(
        results_root=output_root,
        artifact_path=existing_path,
    ) == existing_path


def test_resolve_materialized_artifact_path_maps_workspace_results_paths(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    mapped_path = output_root / "runs" / "row.json"
    mapped_path.parent.mkdir(parents=True)
    mapped_path.write_text("{}", encoding="utf-8")

    artifact_path = Path("/workspace/results/results/runs/row.json")

    assert _resolve_materialized_artifact_path(
        results_root=output_root,
        artifact_path=artifact_path,
    ) == mapped_path


def test_resolve_materialized_artifact_path_keeps_unmapped_absolute_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    artifact_path = Path("/other/results/row.json")

    assert _resolve_materialized_artifact_path(
        results_root=output_root,
        artifact_path=artifact_path,
    ) == artifact_path


def test_resolve_materialized_artifact_path_keeps_missing_workspace_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    artifact_path = Path("/workspace/results/results/missing/row.json")

    assert _resolve_materialized_artifact_path(
        results_root=output_root,
        artifact_path=artifact_path,
    ) == artifact_path


def test_load_batch_summary_source_frame_handles_empty_and_summary_only_inputs(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()

    assert _load_batch_summary_source_frame(results_root) is None

    pd.DataFrame(columns=["graph_source", "recall"]).to_csv(
        results_root / "batch_summary.csv",
        index=False,
    )
    assert _load_batch_summary_source_frame(results_root) is None

    pd.DataFrame([{"graph_source": "default", "recall": 0.5}]).to_csv(
        results_root / "batch_summary.csv",
        index=False,
    )
    loaded = _load_batch_summary_source_frame(results_root)

    assert loaded is not None
    assert loaded.to_dict(orient="records") == [
        {"graph_source": "default", "recall": 0.5}
    ]


def test_load_batch_summary_source_frame_merges_existing_and_missing_raw_rows(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    existing_raw_row_path = results_root / "existing_raw_row.json"
    existing_raw_row_path.write_text(
        json.dumps({"Result": "[]", "raw_only": "preserved"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "graph_source": "default",
                "recall": 0.5,
                "raw_row_path": str(existing_raw_row_path),
            },
            {
                "graph_source": "default",
                "recall": 0.0,
                "raw_row_path": str(results_root / "missing_raw_row.json"),
            },
        ]
    ).to_csv(results_root / "batch_summary.csv", index=False)

    loaded = _load_batch_summary_source_frame(results_root)

    assert loaded is not None
    assert loaded.loc[0, "raw_only"] == "preserved"
    assert loaded.loc[0, "recall"] == 0.5
    assert pd.isna(loaded.loc[1, "raw_only"])


def test_load_map_extension_batch_summary_frame_supports_three_bit_conditions(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    batch_summary_path = results_root / "batch_summary.csv"

    pd.DataFrame(
        [
            {"condition_bits": "000", "recall": 0.25},
            {"condition_bits": "111", "recall": 0.75},
        ]
    ).to_csv(batch_summary_path, index=False)

    loaded = _load_map_extension_batch_summary_frame(results_root)

    assert loaded is not None
    assert loaded["Example"].tolist() == [-1, 1]
    assert loaded["Number of Words"].tolist() == [3, 5]
    assert loaded["Depth"].tolist() == [1, 2]


def test_load_map_extension_batch_summary_frame_rejects_unsupported_width(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    pd.DataFrame({"condition_bits": ["00000"], "recall": [0.5]}).to_csv(
        results_root / "batch_summary.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="Unsupported map-extension condition bit width"):
        _load_map_extension_batch_summary_frame(results_root)


def test_summary_builders_reject_missing_and_unknown_models() -> None:
    with pytest.raises(ValueError, match="Missing required open-weight summary columns"):
        build_open_weight_map_extension_summary(pd.DataFrame([{"model": QWEN}]))

    open_weight_missing = pd.DataFrame(
        [
            {
                "algorithm": "algo3",
                "condition_label": "greedy",
                "graph_source": "default",
                "pair_name": "sg1_sg2",
                "Number of Words": 3,
                "model": QWEN,
                "recall": 0.5,
            }
        ]
    )
    with pytest.raises(ValueError) as open_weight_missing_error:
        _prepare_open_weight_summary_frame(open_weight_missing)
    assert str(open_weight_missing_error.value) == (
        "Missing required open-weight summary columns: Depth, Example"
    )

    valid_map_frame = pd.DataFrame(
        [
            {
                "algorithm": "algo3",
                "condition_label": "greedy",
                "graph_source": "default",
                "pair_name": "sg1_sg2",
                "Example": -1,
                "Number of Words": 3,
                "Depth": 1,
                "model": "unknown/model",
                "recall": 0.5,
            }
        ]
    )
    with pytest.raises(ValueError, match="Unsupported model"):
        build_open_weight_map_extension_summary(valid_map_frame)

    with pytest.raises(ValueError, match="Missing required map recall summary columns"):
        build_map_recall_summary(pd.DataFrame([{"model": QWEN}]))


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


def test_extract_variance_rows_by_algorithm_and_model_skips_boolean_metrics() -> None:
    ledger = {
        "records": [
            _finished_record(
                algorithm="algo3",
                model=QWEN,
                condition_bits="1111",
                condition_label="beam_num_beams_6",
                pair_name="subgraph_1_to_subgraph_3",
                replication=0,
                recall=True,
            )
        ]
    }

    rows_by_key = extract_variance_rows_by_algorithm_and_model(ledger)

    assert rows_by_key == {}


def test_extract_variance_rows_accepts_flattened_records_and_skips_invalid_contexts() -> None:
    flattened = {
        "status": "finished",
        "algorithm": "algo1",
        "model": QWEN,
        "condition_bits": "10100",
        "condition_label": "beam_num_beams_6",
        "pair_name": "sg1_sg2",
        "replication": 0,
        "accuracy": 0.5,
        "precision": 0.4,
        "recall": 0.3,
    }
    invalid_algorithm = {**flattened, "algorithm": "unknown"}
    invalid_model = {**flattened, "model": "unknown/model"}
    invalid_condition = {**flattened, "condition_label": "unknown"}
    invalid_replication = {**flattened, "replication": "not an integer"}
    malformed_identity = {**flattened, "identity": []}

    rows_by_key = extract_variance_rows_by_algorithm_and_model(
        {
            "records": [
                flattened,
                invalid_algorithm,
                invalid_model,
                invalid_condition,
                invalid_replication,
                malformed_identity,
            ]
        }
    )

    assert list(rows_by_key) == [("algo1", "Qwen")]
    assert len(rows_by_key[("algo1", "Qwen")]) == 1


def test_compute_variance_decomposition_closes_to_100_for_algo1() -> None:
    frame = pd.DataFrame(_synthetic_rows("algo1", "Qwen"))

    decomposition = compute_variance_decomposition(frame, "algo1", "Qwen")

    assert set(decomposition["algorithm"]) == {"algo1"}
    assert set(decomposition["model"]) == {"Qwen"}
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


def test_compute_variance_decomposition_handles_constant_metrics() -> None:
    frame = pd.DataFrame(_synthetic_rows("algo1", "Qwen"))
    frame["accuracy"] = 1.0
    frame["recall"] = 1.0
    frame["precision"] = 1.0

    decomposition = compute_variance_decomposition(frame, "algo1", "Qwen")

    for metric in ("accuracy", "recall", "precision"):
        metric_rows = decomposition[decomposition["metric"] == metric]
        assert metric_rows["pct_with_error"].sum() == pytest.approx(100.0)
        assert metric_rows.loc[metric_rows["feature"] == "Error", "ss"].iloc[0] == 1.0


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

    assert first["decomposition_csv"].read_text(encoding="utf-8") == second[
        "decomposition_csv"
    ].read_text(encoding="utf-8")
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


def test_build_open_weight_map_extension_summary_groups_by_graph_source_and_pair_name() -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "algorithm": "algo3",
                "condition_label": "beam_num_beams_6",
                "graph_source": "babs_johnson",
                "pair_name": "subgraph_1_to_subgraph_3",
                "Example": -1,
                "Number of Words": 3,
                "Depth": 1,
                "model": QWEN,
                "recall": 0.25,
            },
            {
                "algorithm": "algo3",
                "condition_label": "beam_num_beams_6",
                "graph_source": "babs_johnson",
                "pair_name": "subgraph_1_to_subgraph_3",
                "Example": -1,
                "Number of Words": 3,
                "Depth": 1,
                "model": MISTRAL,
                "recall": 0.5,
            },
            {
                "algorithm": "algo3",
                "condition_label": "beam_num_beams_6",
                "graph_source": "clarice_starling",
                "pair_name": "subgraph_2_to_subgraph_1",
                "Example": 1,
                "Number of Words": 5,
                "Depth": 2,
                "model": QWEN,
                "recall": 0.75,
            },
            {
                "algorithm": "algo3",
                "condition_label": "beam_num_beams_6",
                "graph_source": "clarice_starling",
                "pair_name": "subgraph_2_to_subgraph_1",
                "Example": 1,
                "Number of Words": 5,
                "Depth": 2,
                "model": MISTRAL,
                "recall": 0.125,
            },
        ]
    )

    summary = build_open_weight_map_extension_summary(frame)

    assert list(summary.columns) == [
        "algorithm",
        "condition_label",
        "graph_source",
        "pair_name",
        "example",
        "number_of_words",
        "depth",
        "qwen_runs",
        "qwen_recall",
        "mistral_runs",
        "mistral_recall",
    ]
    assert len(summary) == 2
    first_row = summary.iloc[0]
    assert first_row["algorithm"] == "algo3"
    assert first_row["graph_source"] == "babs_johnson"
    assert first_row["pair_name"] == "subgraph_1_to_subgraph_3"
    assert bool(first_row["example"]) is False
    assert first_row["number_of_words"] == 3
    assert first_row["depth"] == 1
    assert first_row["qwen_runs"] == 1
    assert first_row["mistral_runs"] == 1
    assert first_row["qwen_recall"] == 0.25
    assert first_row["mistral_recall"] == 0.5


def test_build_open_weight_map_extension_summary_uses_stable_nan_safe_grouping(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "algorithm": ["algo3", "algo3"],
            "condition_label": ["greedy", "greedy"],
            "graph_source": ["b", None],
            "pair_name": ["p1", "p1"],
            "Example": [-1, -1],
            "Number of Words": [3, 3],
            "Depth": [1, 1],
            "model": [QWEN, MISTRAL],
            "recall": [0.2, 0.4],
        }
    )
    groupby_calls = []
    sort_calls = []
    real_groupby = pd.DataFrame.groupby
    real_sort_values = pd.DataFrame.sort_values

    def recording_groupby(self, *args, **kwargs):
        if args and isinstance(args[0], list) and "algorithm" in args[0]:
            groupby_calls.append(kwargs)
        return real_groupby(self, *args, **kwargs)

    def recording_sort_values(self, *args, **kwargs):
        if kwargs.get("by") == [
            "algorithm",
            "condition_label",
            "graph_source",
            "pair_name",
            "example",
            "number_of_words",
            "depth",
        ]:
            sort_calls.append(kwargs)
        return real_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", recording_groupby)
    monkeypatch.setattr(pd.DataFrame, "sort_values", recording_sort_values)

    summary = build_open_weight_map_extension_summary(frame)

    assert not summary.empty
    assert any(
        call.get("sort") is True and call.get("dropna") is False for call in groupby_calls
    )
    assert sort_calls == [
        {
            "by": [
                "algorithm",
                "condition_label",
                "graph_source",
                "pair_name",
                "example",
                "number_of_words",
                "depth",
            ],
            "kind": "stable",
        }
    ]


def test_generate_variance_decomposition_bundle_uses_map_extension_evaluated_rows(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    output_root = results_root / "variance_decomposition"
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "ledger.json").write_text(
        json.dumps({"records": []}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_synthetic_map_extension_results(results_root, raw_root=tmp_path / "raw_rows")

    bundle = generate_variance_decomposition_bundle(results_root, output_root)

    assert (output_root / "open_weight_map_extension_summary.csv").exists()
    assert (output_root / "variance_decomposition.csv").exists()
    assert (output_root / "variance_decomposition_babs_johnson.csv").exists()
    assert (output_root / "variance_decomposition_clarice_starling.csv").exists()
    assert (output_root / "variance_decomposition_philip_marlowe.csv").exists()
    assert (output_root / "map_recall_summary.csv").exists()
    assert bundle["summary_csv"] == output_root / "open_weight_map_extension_summary.csv"
    assert set(bundle["map_decomposition_csvs"]) == {
        "babs_johnson",
        "clarice_starling",
        "philip_marlowe",
    }
    assert bundle["map_recall_summary_csv"] == output_root / "map_recall_summary.csv"
    summary = pd.read_csv(output_root / "open_weight_map_extension_summary.csv")
    assert set(summary["graph_source"]) == {
        "babs_johnson",
        "clarice_starling",
        "philip_marlowe",
    }
    assert set(summary["number_of_words"]) == {3, 5}
    assert set(summary["depth"]) == {1, 2}
    assert set(summary["example"]) == {False, True}
    assert summary["qwen_runs"].min() == 2
    assert summary["mistral_runs"].min() == 2
    assert summary["qwen_recall"].max() > 0.0
    assert summary["mistral_recall"].max() > 0.0
    decomposition = pd.read_csv(output_root / "variance_decomposition.csv")
    assert {
        "algorithm",
        "model",
        "feature",
        "metric",
        "pct_with_error",
        "pct_without_error",
        "ss",
    }.issubset(set(decomposition.columns))
    for _, group in decomposition.groupby(["algorithm", "model", "metric"]):
        assert pytest.approx(group["pct_with_error"].sum(), abs=1e-8) == 100.0
        non_error = group[group["feature"] != "Error"]
        assert pytest.approx(non_error["pct_without_error"].sum(), abs=1e-8) == 100.0
        assert set(group["feature"]).issuperset(
            {"graph_source", "pair_name", "Example", "Number of Words", "Depth", "Error"}
        )
    for graph_source in ("babs_johnson", "clarice_starling", "philip_marlowe"):
        map_decomposition = pd.read_csv(output_root / f"variance_decomposition_{graph_source}.csv")
        assert set(map_decomposition["graph_source"]) == {graph_source}
        for _, group in map_decomposition.groupby(["algorithm", "model", "metric"]):
            assert pytest.approx(group["pct_with_error"].sum(), abs=1e-8) == 100.0
            non_error = group[group["feature"] != "Error"]
            assert pytest.approx(non_error["pct_without_error"].sum(), abs=1e-8) == 100.0

    map_summary = pd.read_csv(output_root / "map_recall_summary.csv")
    assert list(map_summary.columns) == [
        "graph_source",
        "pair_count",
        "prompt_cell_count",
        "qwen_runs",
        "qwen_mean_recall",
        "mistral_runs",
        "mistral_mean_recall",
        "overall_runs",
        "overall_mean_recall",
    ]
    assert set(map_summary["graph_source"]) == {
        "babs_johnson",
        "clarice_starling",
        "philip_marlowe",
    }
    assert (map_summary["pair_count"] == 3).all()
    assert (map_summary["prompt_cell_count"] == 24).all()
    assert (map_summary["qwen_runs"] == 48).all()
    assert (map_summary["mistral_runs"] == 48).all()
    assert (map_summary["overall_runs"] == 96).all()


def test_generate_variance_decomposition_bundle_rejects_batch_summary_recall_mismatches(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "ledger.json").write_text(
        json.dumps({"records": []}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_synthetic_map_extension_results(
        results_root,
        raw_root=tmp_path / "raw_rows",
        mismatched_batch_summary_recall=True,
    )

    with pytest.raises(ValueError, match="recall mismatch"):
        generate_variance_decomposition_bundle(
            results_root,
            results_root / "variance_decomposition",
        )


def test_binary_contrast_levels_maps_numeric_levels_and_preserves_integer_dtype() -> None:
    already_contrasted = _binary_contrast_levels(pd.Series([-1.0, 1.0]))
    remapped = _binary_contrast_levels(pd.Series([10.0, 20.0]))

    assert already_contrasted.tolist() == [-1, 1]
    assert already_contrasted.dtype.kind == "i"
    assert remapped.tolist() == [-1, 1]
    assert remapped.dtype.kind == "i"
    assert _binary_contrast_levels(pd.Series([-2, 1])).tolist() == [-1, 1]
    assert _binary_contrast_levels(pd.Series([-1, 2])).tolist() == [-1, 1]
    with pytest.raises(ValueError, match="Expected exactly two levels for contrast coding"):
        _binary_contrast_levels(pd.Series([1, 2, 3]))
    with pytest.raises(ValueError, match="Expected exactly two levels for contrast coding"):
        _binary_contrast_levels(pd.Series([1, 1]))


def test_prepare_open_weight_summary_frame_casts_fields_and_reports_unknown_models() -> None:
    frame = pd.DataFrame(
        [
            {
                "algorithm": "algo3",
                "condition_label": "greedy",
                "graph_source": "default",
                "pair_name": "p1",
                "Example": "1",
                "Number of Words": "5",
                "Depth": "2",
                "model": QWEN,
                "recall": "0.5",
            }
        ]
    )

    prepared = _prepare_open_weight_summary_frame(frame)

    assert prepared.loc[0, "model_prefix"] == "qwen"
    assert bool(prepared.loc[0, "example"]) is True
    assert prepared.loc[0, "number_of_words"] == 5
    assert prepared.loc[0, "depth"] == 2
    assert prepared["number_of_words"].dtype.kind == "i"
    assert prepared["depth"].dtype.kind == "i"

    unknown = frame.assign(model=["z-model"])
    with pytest.raises(
        ValueError,
        match=r"Unsupported model\(s\) in open-weight summary frame: z-model",
    ):
        _prepare_open_weight_summary_frame(unknown)

    two_unknown = pd.concat([unknown, unknown.assign(model="a-model")])
    with pytest.raises(ValueError) as open_weight_unknown_error:
        _prepare_open_weight_summary_frame(two_unknown)
    assert str(open_weight_unknown_error.value) == (
        "Unsupported model(s) in open-weight summary frame: a-model, z-model"
    )


def test_prepare_map_recall_frame_reports_missing_and_unknown_models() -> None:
    with pytest.raises(
        ValueError,
        match="Missing required map recall summary columns: Depth, Example",
    ):
        _prepare_map_recall_frame(pd.DataFrame({"model": [QWEN]}))

    frame = pd.DataFrame(
        {
            "graph_source": ["default"],
            "pair_name": ["p1"],
            "Example": [1],
            "Number of Words": [5],
            "Depth": [2],
            "model": ["z-model"],
            "recall": [0.5],
        }
    )
    with pytest.raises(
        ValueError,
        match=r"Unsupported model\(s\) in map recall summary frame: z-model",
    ):
        _prepare_map_recall_frame(frame)

    two_unknown_map = pd.concat([frame, frame.assign(model="a-model")])
    with pytest.raises(ValueError) as map_unknown_error:
        _prepare_map_recall_frame(two_unknown_map)
    assert str(map_unknown_error.value) == (
        "Unsupported model(s) in map recall summary frame: a-model, z-model"
    )


def test_build_open_weight_summary_row_fills_missing_model_family_with_zeros() -> None:
    group = pd.DataFrame({"model_prefix": ["qwen"], "runs": [2], "recall": [0.75]})

    row = _build_open_weight_summary_row(
        ("algo3", "greedy", "default", "p1", False, 3, 1),
        group,
    )

    assert row == {
        "algorithm": "algo3",
        "condition_label": "greedy",
        "graph_source": "default",
        "pair_name": "p1",
        "example": False,
        "number_of_words": 3,
        "depth": 1,
        "qwen_runs": 2,
        "qwen_recall": 0.75,
        "mistral_runs": 0,
        "mistral_recall": 0.0,
    }

    mistral_only = _build_open_weight_summary_row(
        ("algo3", "greedy", "default", "p1", False, 3, 1),
        pd.DataFrame({"model_prefix": ["mistral"], "runs": [3], "recall": [0.25]}),
    )
    assert mistral_only["qwen_runs"] == 0
    assert mistral_only["mistral_runs"] == 3
    assert mistral_only["mistral_recall"] == 0.25

    with pytest.raises(ValueError) as invalid_level_error:
        _build_open_weight_summary_row(
            ("algo3", "greedy", "default", "p1", False, "3", 1),
            group,
        )
    assert str(invalid_level_error.value) == "Summary group levels must be integers."


def test_group_open_weight_summary_frame_preserves_nan_model_groups(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "algorithm": ["algo3", "algo3"],
            "condition_label": ["greedy", "greedy"],
            "graph_source": ["z", "a"],
            "pair_name": ["p1", "p1"],
            "example": [False, False],
            "number_of_words": [3, 3],
            "depth": [1, 1],
            "model_prefix": ["qwen", float("nan")],
            "recall": [0.2, 0.4],
        }
    )

    groupby_calls = []
    sort_calls = []
    real_groupby = pd.DataFrame.groupby
    real_sort_values = pd.DataFrame.sort_values

    def recording_groupby(self, *args, **kwargs):
        if args and isinstance(args[0], list) and "model_prefix" in args[0]:
            groupby_calls.append(kwargs)
        return real_groupby(self, *args, **kwargs)

    def recording_sort_values(self, *args, **kwargs):
        if kwargs.get("by") and "model_prefix" in kwargs["by"]:
            sort_calls.append(kwargs)
        return real_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", recording_groupby)
    monkeypatch.setattr(pd.DataFrame, "sort_values", recording_sort_values)

    grouped = _group_open_weight_summary_frame(frame)

    assert len(grouped) == 2
    assert grouped["graph_source"].tolist() == ["a", "z"]
    assert grouped.loc[grouped["graph_source"] == "a", "model_prefix"].isna().all()
    assert groupby_calls == [{"dropna": False}]
    assert sort_calls == [
        {
            "by": [
                "algorithm",
                "condition_label",
                "graph_source",
                "pair_name",
                "example",
                "number_of_words",
                "depth",
                "model_prefix",
            ],
            "kind": "stable",
        }
    ]


def test_build_graph_recall_summary_row_reports_family_and_overall_statistics() -> None:
    group = pd.DataFrame(
        {
            "pair_name": ["p1", "p2"],
            "Example": [-1, 1],
            "Number of Words": [3, 5],
            "Depth": [1, 2],
            "model_prefix": ["qwen", "qwen"],
            "recall": [0.25, 0.75],
        }
    )

    row = _build_graph_recall_summary_row("default", group)

    assert row == {
        "graph_source": "default",
        "pair_count": 2,
        "prompt_cell_count": 2,
        "overall_runs": 2,
        "overall_mean_recall": 0.5,
        "qwen_runs": 2,
        "qwen_mean_recall": 0.5,
        "mistral_runs": 0,
        "mistral_mean_recall": 0.0,
    }


def test_build_map_recall_summary_sorts_and_preserves_missing_graph_source(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "graph_source": ["z", "a", None],
            "pair_name": ["p1", "p1", "p1"],
            "Example": [-1, -1, -1],
            "Number of Words": [3, 3, 3],
            "Depth": [1, 1, 1],
            "model": [QWEN, MISTRAL, QWEN],
            "recall": [0.2, 0.4, 0.6],
        }
    )

    groupby_calls = []
    sort_calls = []
    reset_calls = []
    real_groupby = pd.DataFrame.groupby
    real_sort_values = pd.DataFrame.sort_values
    real_reset_index = pd.DataFrame.reset_index

    def recording_groupby(self, *args, **kwargs):
        if args and args[0] == "graph_source":
            groupby_calls.append(kwargs)
        return real_groupby(self, *args, **kwargs)

    def recording_sort_values(self, *args, **kwargs):
        sort_calls.append(kwargs)
        return real_sort_values(self, *args, **kwargs)

    def recording_reset_index(self, *args, **kwargs):
        reset_calls.append(kwargs)
        return real_reset_index(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", recording_groupby)
    monkeypatch.setattr(pd.DataFrame, "sort_values", recording_sort_values)
    monkeypatch.setattr(pd.DataFrame, "reset_index", recording_reset_index)

    summary = build_map_recall_summary(frame)

    assert list(summary.columns) == [
        "graph_source",
        "pair_count",
        "prompt_cell_count",
        "qwen_runs",
        "qwen_mean_recall",
        "mistral_runs",
        "mistral_mean_recall",
        "overall_runs",
        "overall_mean_recall",
    ]
    assert summary["graph_source"].dropna().tolist() == ["a", "z"]
    assert summary["graph_source"].isna().sum() == 1
    assert summary.index.tolist() == [0, 1, 2]
    assert groupby_calls == [{"sort": True, "dropna": False}]
    assert any(call.get("kind") == "stable" for call in sort_calls)
    assert reset_calls == [{"drop": True}]


def test_center_metric_values_coerces_strings_and_handles_constant_metrics() -> None:
    centered, total_ss = _center_metric_values(pd.DataFrame({"recall": ["1", "3"]}), "recall")
    constant, constant_total_ss = _center_metric_values(
        pd.DataFrame({"recall": ["2", "2"]}),
        "recall",
    )

    assert centered.dtype.kind == "f"
    assert centered.tolist() == [-1.0, 1.0]
    assert total_ss == 2.0
    assert constant.tolist() == [0.0, 0.0]
    assert constant_total_ss == 1.0


def test_compute_metric_decomposition_rows_preserves_algorithm_and_model() -> None:
    rows = _compute_metric_decomposition_rows(
        pd.DataFrame({"recall": [1.0, 3.0]}),
        algorithm="algo3",
        model="Mistral",
        metric="recall",
        term_columns=[],
    )

    assert rows == [
        {
            "algorithm": "algo3",
            "model": "Mistral",
            "feature": "Error",
            "metric": "recall",
            "ss": 2.0,
            "pct_with_error": 100.0,
            "pct_without_error": 0.0,
        }
    ]


def test_compute_decomposition_rows_sorts_stably_and_resets_output_index(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "factor": [-1, 1],
            "recall": [1.0, 3.0],
        }
    )

    sort_calls = []
    real_sort_values = pd.DataFrame.sort_values

    def recording_sort_values(self, *args, **kwargs):
        sort_calls.append(kwargs)
        return real_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", recording_sort_values)
    decomposition = _compute_decomposition_rows(
        frame,
        algorithm="algo3",
        model="Mistral",
        factor_order=("factor",),
        metrics=("recall",),
    )

    assert decomposition.index.tolist() == list(range(len(decomposition)))
    assert "index" not in decomposition.columns
    assert any(call.get("kind") == "stable" for call in sort_calls)
    assert decomposition["model"].eq("Mistral").all()
    assert "Error" in set(decomposition["feature"])


def test_build_variance_row_emits_identity_and_decoding_columns() -> None:
    context = _VarianceRecordContext(
        record={"graph_source": "record-graph"},
        identity={
            "pair_name": "p1",
            "condition_bits": "10100",
            "graph_source": "identity-graph",
        },
        metric_values={},
        algorithm="algo1",
        model_label="Qwen",
        metrics=("accuracy", "precision", "recall"),
        condition_label="beam_num_beams_6",
        replication=2,
    )

    row = _build_variance_row(context)

    assert row["algorithm"] == "algo1"
    assert row["model"] == "Qwen"
    assert row["pair_name"] == "p1"
    assert row["condition_bits"] == "10100"
    assert row["condition_label"] == "beam_num_beams_6"
    assert row["replication"] == 2
    assert row["graph_source"] == "identity-graph"
    assert row["Example"] == -1
    assert row["Explanation"] == 1
    assert row["Counterexample"] == 1
    assert row["Decoding Family"] == "beam"

    record_graph_context = context.__class__(
        **{
            **context.__dict__,
            "identity": {
                key: value
                for key, value in context.identity.items()
                if key != "graph_source"
            },
        }
    )
    assert _build_variance_row(record_graph_context)["graph_source"] == "record-graph"

    identity_graph_context = context.__class__(
        **{**context.__dict__, "record": {}, "identity": {"graph_source": "identity-only"}}
    )
    assert _build_variance_row(identity_graph_context)["graph_source"] == "identity-only"

    missing_defaults_context = context.__class__(
        **{
            **context.__dict__,
            "record": {},
            "identity": {},
        }
    )
    missing_defaults_row = _build_variance_row(missing_defaults_context)
    assert missing_defaults_row["pair_name"] == ""
    assert missing_defaults_row["condition_bits"] == ""


def test_extract_metric_row_falls_back_to_flattened_record_metrics() -> None:
    context = _VarianceRecordContext(
        record={"accuracy": "0.8", "recall": 0.3},
        identity={},
        metric_values={},
        algorithm="algo1",
        model_label="Qwen",
        metrics=("accuracy", "recall"),
        condition_label="greedy",
        replication=0,
    )

    assert _extract_metric_row(context) == {"accuracy": 0.8, "recall": 0.3}


def test_extract_variance_rows_skips_invalid_records_and_continues() -> None:
    invalid = _finished_record(
        algorithm="algo3",
        model=QWEN,
        condition_bits="1111",
        condition_label="greedy",
        pair_name="invalid",
        replication=0,
        recall=0.1,
    )
    del invalid["winner"]["metrics"]["recall"]
    valid = _finished_record(
        algorithm="algo3",
        model=QWEN,
        condition_bits="1111",
        condition_label="greedy",
        pair_name="valid",
        replication=0,
        recall=0.9,
    )

    rows = extract_variance_rows_by_algorithm_and_model({"records": [invalid, valid]})

    assert list(rows) == [("algo3", "Qwen")]
    assert len(rows[("algo3", "Qwen")]) == 1
    assert rows[("algo3", "Qwen")][0]["pair_name"] == "valid"


def test_variance_record_context_accepts_implicit_finished_status() -> None:
    record = _finished_record(
        algorithm="algo3",
        model=QWEN,
        condition_bits="1111",
        condition_label="greedy",
        pair_name="p1",
        replication=3,
        recall=0.9,
    )
    del record["status"]

    context = _variance_record_context(record)

    assert context is not None
    assert context.algorithm == "algo3"
    assert context.model_label == "Qwen"
    assert context.replication == 3


def test_variance_record_context_rejects_non_finished_and_defaults_replication() -> None:
    rejected = _finished_record(
        algorithm="algo3",
        model=QWEN,
        condition_bits="1111",
        condition_label="greedy",
        pair_name="p1",
        replication=1,
        recall=0.9,
    )
    rejected["status"] = "retryable_failed"
    assert _variance_record_context(rejected) is None

    missing_replication = _finished_record(
        algorithm="algo3",
        model=QWEN,
        condition_bits="1111",
        condition_label="greedy",
        pair_name="p1",
        replication=1,
        recall=0.9,
    )
    del missing_replication["identity"]["replication"]
    context = _variance_record_context(missing_replication)

    assert context is not None
    assert context.replication == 0


def test_resolve_algorithm_model_and_condition_label_accept_supported_values() -> None:
    resolved = _resolve_algorithm_and_model({"algorithm": "algo1", "model": QWEN})
    fallback_model = _resolve_algorithm_and_model({"algorithm": "algo1", "model": "Qwen"})

    assert resolved is not None
    assert resolved[0] == "algo1"
    assert resolved[1] == "Qwen"
    assert "accuracy" in resolved[2]
    assert fallback_model is not None and fallback_model[1] == "Qwen"
    assert _resolve_algorithm_and_model({"algorithm": "unknown", "model": QWEN}) is None
    assert _resolve_algorithm_and_model({"algorithm": "algo1", "model": "unknown"}) is None
    assert _resolve_condition_label({"condition_label": "greedy"}) == "greedy"
    assert _resolve_condition_label({"condition_label": "unknown"}) is None


def test_group_key_tuple_validates_type_size_and_error_message() -> None:
    assert _group_key_tuple(("algo1", "Qwen"), expected_size=2) == ("algo1", "Qwen")

    with pytest.raises(ValueError, match="Expected grouped key with 2 values"):
        _group_key_tuple(("algo1",), expected_size=2)
    with pytest.raises(ValueError, match="Expected grouped key with 2 values"):
        _group_key_tuple(["algo1", "Qwen"], expected_size=2)


def test_materialized_artifact_candidates_handles_root_prefixed_and_empty_paths(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    candidates = _materialized_artifact_candidates(results_root, Path("results/row.json"))
    empty_candidates = _materialized_artifact_candidates(results_root, Path())

    assert candidates == [
        results_root / "results/row.json",
        results_root / "row.json",
    ]
    assert empty_candidates == [results_root]


def test_batch_summary_record_loaders_resolve_relative_raw_rows(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    raw_row_path = results_root / "raw" / "row.json"
    raw_row_path.parent.mkdir(parents=True)
    raw_row_path.write_text(json.dumps({"raw_only": "yes", "recall": 0.1}), encoding="utf-8")
    record = {"raw_row_path": "raw/row.json", "recall": 0.9}

    loaded = _load_batch_summary_record(results_root, record)
    loaded_many = _load_batch_summary_records(results_root, pd.DataFrame([record]))

    assert loaded == {"raw_only": "yes", "recall": 0.9, "raw_row_path": "raw/row.json"}
    assert loaded_many.to_dict(orient="records") == [loaded]
    assert _read_raw_row(results_root / "missing.json") == {}


def test_read_raw_row_reads_utf8_json_with_explicit_encoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_row_path = tmp_path / "row.json"
    raw_row_path.write_text(json.dumps({"recall": 0.5}), encoding="utf-8")
    read_calls = []
    real_read_text = Path.read_text

    def recording_read_text(self, *args, **kwargs):
        if self == raw_row_path:
            read_calls.append((args, kwargs))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    assert _read_raw_row(raw_row_path) == {"recall": 0.5}
    assert read_calls == [((), {"encoding": "utf-8"})]


def test_load_batch_summary_records_requests_records_orientation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    raw_row_path = results_root / "row.json"
    raw_row_path.write_text("{}", encoding="utf-8")
    batch_summary = pd.DataFrame([{"raw_row_path": "row.json"}])
    to_dict_calls = []
    real_to_dict = pd.DataFrame.to_dict

    def recording_to_dict(self, *args, **kwargs):
        to_dict_calls.append((args, kwargs))
        return real_to_dict(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_dict", recording_to_dict)

    _load_batch_summary_records(results_root, batch_summary)

    assert to_dict_calls == [((), {"orient": "records"})]


def test_load_variance_source_frame_prefers_batch_summary_before_ledger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = pd.DataFrame({"algorithm": ["algo1"], "model": [QWEN]})
    monkeypatch.setattr(
        variance_module,
        "_load_map_extension_evaluated_frame",
        lambda results_root: None,
    )
    monkeypatch.setattr(
        variance_module,
        "_load_batch_summary_source_frame",
        lambda results_root: expected,
    )
    monkeypatch.setattr(
        variance_module,
        "_ledger_to_variance_source_frame",
        lambda ledger: pytest.fail("ledger fallback should not be used"),
    )

    assert _load_variance_source_frame(tmp_path) is expected


def test_load_variance_source_frame_reads_utf8_ledger_at_canonical_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    ledger_path = results_root / "ledger.json"
    ledger_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    read_calls = []
    real_read_text = Path.read_text
    expected = pd.DataFrame({"algorithm": ["algo1"]})

    def recording_read_text(self, *args, **kwargs):
        if self == ledger_path:
            read_calls.append((self, args, kwargs))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)
    monkeypatch.setattr(variance_module, "_load_map_extension_evaluated_frame", lambda root: None)
    monkeypatch.setattr(variance_module, "_load_batch_summary_source_frame", lambda root: None)
    monkeypatch.setattr(
        variance_module,
        "_ledger_to_variance_source_frame",
        lambda ledger: expected,
    )

    assert _load_variance_source_frame(results_root) is expected
    assert read_calls == [(ledger_path, (), {"encoding": "utf-8"})]


def test_normalize_map_extension_frame_renames_columns_and_sets_algorithm() -> None:
    frame = pd.DataFrame(
        {
            "Counter-Example": [-1],
            "Recall": [0.5],
            "Repetition": [2],
            "decoding_condition": ["greedy"],
        }
    )

    normalized = _normalize_map_extension_frame(frame, algorithm="algo3")

    assert normalized.to_dict(orient="records") == [
        {
            "Counterexample": -1,
            "recall": 0.5,
            "replication": 2,
            "condition_label": "greedy",
            "algorithm": "algo3",
        }
    ]


def test_load_map_extension_batch_summary_frame_decodes_all_bit_width_fields(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    pd.DataFrame({"condition_bits": ["0101", "1010"]}).to_csv(
        results_root / "batch_summary.csv",
        index=False,
    )

    loaded = _load_map_extension_batch_summary_frame(results_root)

    assert loaded is not None
    assert loaded["Example"].tolist() == [-1, 1]
    assert loaded["Counterexample"].tolist() == [1, -1]
    assert loaded["Number of Words"].tolist() == [3, 5]
    assert loaded["Depth"].tolist() == [2, 1]
    assert loaded["Example"].dtype.kind == "i"
    assert loaded["Counterexample"].dtype.kind == "i"
    assert loaded["Number of Words"].dtype.kind == "i"
    assert loaded["Depth"].dtype.kind == "i"


def test_load_map_extension_batch_summary_frame_decodes_three_bit_legacy_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    pd.DataFrame({"condition_bits": ["010", "101"]}).to_csv(
        results_root / "batch_summary.csv",
        index=False,
    )

    astype_calls = []
    real_astype = pd.Series.astype

    def recording_astype(self, dtype, *args, **kwargs):
        if dtype is int:
            astype_calls.append(dtype)
        return real_astype(self, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "astype", recording_astype)

    loaded = _load_map_extension_batch_summary_frame(results_root)

    assert loaded is not None
    assert loaded["Example"].tolist() == [-1, 1]
    assert loaded["Counterexample"].tolist() == [-1, -1]
    assert loaded["Number of Words"].tolist() == [5, 3]
    assert loaded["Depth"].tolist() == [1, 2]
    assert astype_calls == [int, int, int]


def test_load_batch_summary_source_frame_uses_canonical_path_and_results_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    pd.DataFrame(
        {"graph_source": ["default"], "raw_row_path": ["row.json"]}
    ).to_csv(results_root / "batch_summary.csv", index=False)
    read_csv_calls = []
    records_calls = []
    real_read_csv = pd.read_csv
    expected = pd.DataFrame({"graph_source": ["default"]})

    def recording_read_csv(path, *args, **kwargs):
        read_csv_calls.append(path)
        return real_read_csv(path, *args, **kwargs)

    def recording_records(root, frame):
        records_calls.append((root, frame))
        return expected

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    monkeypatch.setattr(variance_module, "_load_batch_summary_records", recording_records)

    assert _load_batch_summary_source_frame(results_root) is expected
    assert read_csv_calls == [results_root / "batch_summary.csv"]
    assert records_calls[0][0] == results_root


def test_load_map_extension_batch_summary_frame_uses_canonical_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    pd.DataFrame({"condition_bits": ["010", "101"]}).to_csv(
        results_root / "batch_summary.csv",
        index=False,
    )
    read_csv_calls = []
    real_read_csv = pd.read_csv

    def recording_read_csv(path, *args, **kwargs):
        read_csv_calls.append(path)
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)

    assert _load_map_extension_batch_summary_frame(results_root) is not None
    assert read_csv_calls == [results_root / "batch_summary.csv"]


def test_load_map_extension_evaluated_frame_resets_combined_index(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    for model_name, recall in (("qwen", 0.2), ("mistral", 0.8)):
        evaluated_root = results_root / "aggregated" / "algo3" / model_name / "combined"
        evaluated_root.mkdir(parents=True)
        pd.DataFrame(
            {
                "graph_source": ["default"],
                "Recall": [recall],
                "model": [model_name],
            }
        ).to_csv(evaluated_root / "evaluated.csv", index=False)

    loaded = _load_map_extension_evaluated_frame(results_root)

    assert loaded is not None
    assert loaded.index.tolist() == [0, 1]
    assert loaded["algorithm"].tolist() == ["algo3", "algo3"]
    assert loaded["recall"].tolist() == [0.8, 0.2]


def test_assert_map_extension_recall_matches_reports_identity_and_value_mismatches(
    monkeypatch,
) -> None:
    identity = {
        "algorithm": "algo3",
        "model": QWEN,
        "graph_source": "default",
        "pair_name": "p1",
        "condition_label": "greedy",
        "Example": -1,
        "Counterexample": -1,
        "Number of Words": 3,
        "Depth": 1,
        "replication": 0,
    }
    evaluated = pd.DataFrame(
        [
            {**identity, "recall": 0.5},
            {**{**identity, "pair_name": "p2"}, "recall": 0.5},
        ]
    )
    different_identity = pd.DataFrame(
        [
            {**identity, "recall": 0.5},
            {**{**identity, "pair_name": "p3"}, "recall": 0.5},
        ]
    )
    merge_calls = []
    astype_calls = []
    real_merge = pd.DataFrame.merge
    real_astype = pd.Series.astype

    def recording_merge(self, *args, **kwargs):
        merge_calls.append((args, kwargs))
        return real_merge(self, *args, **kwargs)

    def recording_astype(self, dtype, *args, **kwargs):
        if self.name in {"evaluated_recall", "batch_summary_recall"}:
            astype_calls.append((self.name, dtype, args, kwargs))
        return real_astype(self, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", recording_merge)
    monkeypatch.setattr(pd.Series, "astype", recording_astype)

    with pytest.raises(ValueError) as identity_error:
        _assert_map_extension_recall_matches_batch_summary(evaluated, different_identity)
    assert str(identity_error.value) == (
        "Map-extension evaluated/batch-summary identity mismatch for 2 run(s)."
    )

    astype_calls.clear()
    evaluated_value = pd.DataFrame([{**identity, "recall": "0.5"}])
    different_value = pd.DataFrame([{**identity, "recall": "0.6"}])
    with pytest.raises(ValueError) as recall_error:
        _assert_map_extension_recall_matches_batch_summary(evaluated_value, different_value)
    assert str(recall_error.value) == (
        "Map-extension recall mismatch between evaluated artifacts and batch summary "
        "for 1 run(s)."
    )
    assert astype_calls == [
        ("evaluated_recall", float, (), {}),
        ("batch_summary_recall", float, (), {}),
    ]

    assert len(merge_calls) == 2
    for _args, kwargs in merge_calls:
        assert kwargs["on"] == [
            "algorithm",
            "model",
            "graph_source",
            "pair_name",
            "condition_label",
            "Example",
            "Counterexample",
            "Number of Words",
            "Depth",
            "replication",
        ]
        assert kwargs["how"] == "outer"
        assert kwargs["indicator"] is True


def test_write_map_specific_decompositions_preserves_group_order_and_output_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_rows = []
    for graph_source in ("b", "a", None):
        for model in (QWEN, MISTRAL):
            for level in (-1, 1):
                source_rows.append(
                    {
                        "graph_source": graph_source,
                        "model": model,
                        "pair_name": "p1",
                        "Example": level,
                        "Number of Words": level,
                        "Depth": level,
                        "recall": float(level),
                    }
                )
    for level in (-1, 1):
        source_rows.append(
            {
                "graph_source": "a",
                "model": None,
                "pair_name": "p1",
                "Example": level,
                "Number of Words": level,
                "Depth": level,
                "recall": float(level),
            }
        )
    normalized = pd.DataFrame(source_rows)
    compute_calls = []
    concat_calls = []
    groupby_calls = []
    to_csv_calls = []
    real_concat = variance_module.pd.concat
    real_groupby = pd.DataFrame.groupby
    real_to_csv = pd.DataFrame.to_csv

    def fake_compute(frame, *, algorithm, model, factor_order, metrics):
        compute_calls.append(
            (
                algorithm,
                model,
                tuple(factor_order),
                tuple(metrics),
                str(frame["graph_source"].iloc[0]),
            )
        )
        return pd.DataFrame(
            {
                "algorithm": [algorithm],
                "model": [model],
                "feature": ["Error"],
                "metric": ["recall"],
                "ss": [0.0],
            },
            index=[7],
        )

    def recording_concat(*args, **kwargs):
        concat_calls.append(kwargs)
        return real_concat(*args, **kwargs)

    def recording_groupby(self, *args, **kwargs):
        if args and args[0] in {"graph_source", "model"}:
            groupby_calls.append((args[0], kwargs))
        return real_groupby(self, *args, **kwargs)

    def recording_to_csv(self, path, *args, **kwargs):
        to_csv_calls.append((path, kwargs))
        return real_to_csv(self, path, *args, **kwargs)

    monkeypatch.setattr(variance_module, "_compute_decomposition_rows", fake_compute)
    monkeypatch.setattr(variance_module.pd, "concat", recording_concat)
    monkeypatch.setattr(pd.DataFrame, "groupby", recording_groupby)
    monkeypatch.setattr(pd.DataFrame, "to_csv", recording_to_csv)

    output = _write_map_specific_decompositions(normalized, tmp_path)

    assert set(output) == {"a", "b", "nan"}
    assert len(compute_calls) == 7
    assert [(call[1], call[4]) for call in compute_calls] == [
        ("Qwen", "a"),
        ("Mistral", "a"),
        ("nan", "a"),
        ("Qwen", "b"),
        ("Mistral", "b"),
        ("Qwen", "nan"),
        ("Mistral", "nan"),
    ]
    assert groupby_calls == [
        ("graph_source", {"sort": True, "dropna": False}),
        ("model", {"sort": True, "dropna": False}),
        ("model", {"sort": True, "dropna": False}),
        ("model", {"sort": True, "dropna": False}),
    ]
    assert concat_calls and all(call["ignore_index"] is True for call in concat_calls)
    assert [path.name for path, _kwargs in to_csv_calls] == [
        "variance_decomposition_a.csv",
        "variance_decomposition_b.csv",
        "variance_decomposition_nan.csv",
    ]
    assert all(kwargs == {"index": False} for _path, kwargs in to_csv_calls)
    for graph_source, output_path in output.items():
        written = pd.read_csv(output_path)
        assert written.columns[0] == "graph_source"
        if graph_source == "nan":
            assert written["graph_source"].isna().all()
        else:
            assert set(written["graph_source"].astype(str)) == {graph_source}
        assert set(written["algorithm"]) == {"algo3"}


def test_write_map_extension_variance_decomposition_orchestrates_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_frame = pd.DataFrame({"source": ["raw"]})
    normalized = pd.DataFrame(
        [
            {
                "graph_source": "default",
                "pair_name": "p1",
                "model": QWEN,
                "Example": -1,
                "Number of Words": -1,
                "Depth": -1,
                "recall": 0.2,
            },
            {
                "graph_source": "default",
                "pair_name": "p1",
                "model": QWEN,
                "Example": 1,
                "Number of Words": 1,
                "Depth": 1,
                "recall": 0.8,
            },
            {
                "graph_source": "default",
                "pair_name": "p1",
                "model": None,
                "Example": -1,
                "Number of Words": -1,
                "Depth": -1,
                "recall": 0.3,
            },
            {
                "graph_source": "default",
                "pair_name": "p1",
                "model": None,
                "Example": 1,
                "Number of Words": 1,
                "Depth": 1,
                "recall": 0.7,
            },
        ]
    )
    normalize_calls = []
    compute_calls = []
    concat_calls = []
    groupby_calls = []
    output_calls = []
    map_specific = {"default": tmp_path / "variance_decomposition_default.csv"}
    real_concat = variance_module.pd.concat
    real_groupby = pd.DataFrame.groupby

    def recording_normalize(frame, *, algorithm):
        normalize_calls.append((frame, algorithm))
        return normalized.copy()

    def fake_compute(frame, *, algorithm, model, factor_order, metrics):
        compute_calls.append((algorithm, model, tuple(factor_order), tuple(metrics)))
        return pd.DataFrame(
            {
                "algorithm": [algorithm],
                "model": [model],
                "feature": ["Error"],
                "metric": ["recall"],
                "ss": [0.0],
            },
            index=[9],
        )

    def recording_concat(*args, **kwargs):
        concat_calls.append(kwargs)
        return real_concat(*args, **kwargs)

    def recording_groupby(self, *args, **kwargs):
        if args and args[0] == "model":
            groupby_calls.append(kwargs)
        return real_groupby(self, *args, **kwargs)

    def fake_outputs(**kwargs):
        output_calls.append(kwargs)
        return {"decomposition_csv": tmp_path / "variance_decomposition.csv"}

    monkeypatch.setattr(variance_module, "_normalize_map_extension_frame", recording_normalize)
    monkeypatch.setattr(variance_module, "_compute_decomposition_rows", fake_compute)
    monkeypatch.setattr(
        variance_module,
        "render_variance_decomposition_table",
        lambda algorithm, frame: "table",
    )
    monkeypatch.setattr(variance_module, "write_variance_decomposition_outputs", fake_outputs)
    monkeypatch.setattr(
        variance_module,
        "_write_map_specific_decompositions",
        lambda frame, target: map_specific,
    )
    monkeypatch.setattr(variance_module.pd, "concat", recording_concat)
    monkeypatch.setattr(pd.DataFrame, "groupby", recording_groupby)

    result = _write_map_extension_variance_decomposition(source_frame, tmp_path)

    assert normalize_calls == [(source_frame, "algo3")]
    assert compute_calls == [
        (
            "algo3",
            "Qwen",
            ("graph_source", "pair_name", "Example", "Number of Words", "Depth"),
            ("recall",),
        ),
        (
            "algo3",
            "nan",
            ("graph_source", "pair_name", "Example", "Number of Words", "Depth"),
            ("recall",),
        ),
    ]
    assert groupby_calls == [{"sort": True, "dropna": False}]
    assert concat_calls and concat_calls[0]["ignore_index"] is True
    assert output_calls[0]["algorithm_csvs"] == {
        "algo3": tmp_path / "variance_decomposition_algo3.csv"
    }
    assert output_calls[0]["tables"] == {"algo3": "table"}
    assert result["decomposition_csv"] == tmp_path / "variance_decomposition.csv"
    assert result["map_decomposition_csvs"] == map_specific


def test_generate_map_extension_bundle_writes_summaries_and_merges_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_frame = pd.DataFrame({"graph_source": ["default"]})
    summary = Mock()
    map_recall_summary = Mock()
    output_records = {"decomposition_csv": tmp_path / "variance_decomposition.csv"}

    monkeypatch.setattr(
        variance_module,
        "build_open_weight_map_extension_summary",
        lambda frame: summary,
    )
    monkeypatch.setattr(
        variance_module,
        "build_map_recall_summary",
        lambda frame: map_recall_summary,
    )
    monkeypatch.setattr(
        variance_module,
        "_write_map_extension_variance_decomposition",
        lambda frame, target_dir: output_records,
    )

    result = _generate_map_extension_bundle(source_frame, tmp_path)

    summary.to_csv.assert_called_once_with(
        tmp_path / "open_weight_map_extension_summary.csv",
        index=False,
    )
    map_recall_summary.to_csv.assert_called_once_with(
        tmp_path / "map_recall_summary.csv",
        index=False,
    )
    assert result == {
        "summary_csv": tmp_path / "open_weight_map_extension_summary.csv",
        "map_recall_summary_csv": tmp_path / "map_recall_summary.csv",
        **output_records,
    }


def test_generate_standard_variance_bundle_groups_deterministically_and_writes_all_algorithms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_frame = pd.DataFrame(
        {
            "algorithm": ["algo3", "algo1", "algo2"],
            "model": [MISTRAL, QWEN, None],
            "marker": [3, 1, 2],
        }
    )
    compute_calls = []
    render_calls = []
    concat_calls = []
    groupby_calls = []
    output_calls = []
    real_concat = variance_module.pd.concat
    real_groupby = pd.DataFrame.groupby

    def fake_compute(frame, algorithm, model):
        compute_calls.append((algorithm, model, frame["marker"].tolist()))
        return pd.DataFrame(
            {
                "algorithm": [algorithm],
                "model": [model],
                "feature": ["Error"],
                "metric": ["recall"],
                "ss": [0.0],
            },
            index=[5],
        )

    def fake_render(algorithm, frame):
        render_calls.append((algorithm, frame["algorithm"].tolist()))
        return f"table-{algorithm}"

    def recording_concat(*args, **kwargs):
        concat_calls.append(kwargs)
        return real_concat(*args, **kwargs)

    def recording_groupby(self, *args, **kwargs):
        if args and args[0] == ["algorithm", "model"]:
            groupby_calls.append(kwargs)
        return real_groupby(self, *args, **kwargs)

    def fake_outputs(**kwargs):
        output_calls.append(kwargs)
        return {"combined_table": tmp_path / "combined.tex"}

    monkeypatch.setattr(variance_module, "compute_variance_decomposition", fake_compute)
    monkeypatch.setattr(variance_module, "render_variance_decomposition_table", fake_render)
    monkeypatch.setattr(variance_module, "write_variance_decomposition_outputs", fake_outputs)
    monkeypatch.setattr(variance_module.pd, "concat", recording_concat)
    monkeypatch.setattr(pd.DataFrame, "groupby", recording_groupby)

    result = _generate_standard_variance_bundle(source_frame, tmp_path)

    assert compute_calls == [
        ("algo1", QWEN, [1]),
        ("algo2", "nan", [2]),
        ("algo3", MISTRAL, [3]),
    ]
    assert concat_calls == [{"ignore_index": True}]
    assert groupby_calls == [{"sort": True, "dropna": False}]
    assert dict(render_calls) == {
        "algo1": ["algo1"],
        "algo2": ["algo2"],
        "algo3": ["algo3"],
    }
    assert output_calls[0]["algorithm_csvs"] == {
        algorithm: tmp_path / f"variance_decomposition_{algorithm}.csv"
        for algorithm in ("algo1", "algo2", "algo3")
    }
    assert output_calls[0]["tables"] == {
        "algo1": "table-algo1",
        "algo2": "table-algo2",
        "algo3": "table-algo3",
    }
    assert result["decomposition"].index.tolist() == [0, 1, 2]
    assert result["combined_table"] == tmp_path / "combined.tex"


def test_generate_variance_decomposition_bundle_creates_nested_output_dir_and_reuses_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "nested" / "variance"
    source_frame = pd.DataFrame({"algorithm": ["algo1"]})
    bundle = {"sentinel": True}
    calls = []

    monkeypatch.setattr(
        variance_module,
        "_load_variance_source_frame",
        lambda results_root: source_frame,
    )

    def fake_generate(frame, target_dir):
        calls.append((frame, target_dir))
        return bundle

    monkeypatch.setattr(variance_module, "_generate_standard_variance_bundle", fake_generate)

    assert generate_variance_decomposition_bundle(tmp_path / "results", output_dir) == bundle
    assert generate_variance_decomposition_bundle(tmp_path / "results", output_dir) == bundle
    assert output_dir.is_dir()
    assert calls == [(source_frame, output_dir), (source_frame, output_dir)]


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


def _write_synthetic_map_extension_results(
    results_root: Path,
    *,
    raw_root: Path,
    mismatched_batch_summary_recall: bool = False,
) -> None:
    records: list[dict[str, object]] = []
    for model in (QWEN, MISTRAL):
        evaluated_rows: list[dict[str, object]] = []
        model_dir = results_root / "aggregated" / "algo3" / model.replace("/", "__") / "combined"
        model_dir.mkdir(parents=True, exist_ok=True)
        factor_space = product(
            ("babs_johnson", "clarice_starling", "philip_marlowe"),
            (
                "subgraph_1_to_subgraph_3",
                "subgraph_2_to_subgraph_1",
                "subgraph_2_to_subgraph_3",
            ),
            (-1, 1),
            (3, 5),
            (1, 2),
            (0, 1),
        )
        for index, (
            graph_source,
            pair_name,
            example,
            number_of_words,
            depth,
            replication,
        ) in enumerate(factor_space):
            recall = _map_extension_recall(
                model=model,
                graph_source=graph_source,
                pair_name=pair_name,
                example=example,
                number_of_words=number_of_words,
                depth=depth,
                replication=replication,
            )
            raw_row_path = raw_root / model.replace("/", "__") / f"raw_row_{index}.json"
            raw_row_path.parent.mkdir(parents=True, exist_ok=True)
            raw_row_path.write_text(
                json.dumps(
                    {
                        "Counter-Example": -1,
                        "Depth": depth,
                        "Example": example,
                        "Number of Words": number_of_words,
                        "Recall": recall,
                        "Repetition": replication,
                        "decoding_algorithm": "beam",
                        "decoding_condition": "beam_num_beams_6",
                        "embedding_model": model,
                        "graph_source": graph_source,
                        "model": model,
                        "pair_name": pair_name,
                        "provider": "hf",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            records.append(
                {
                    "algorithm": "algo3",
                    "condition_label": "beam_num_beams_6",
                    "condition_bits": "".join(
                        (
                            "1" if example == 1 else "0",
                            "0",
                            "1" if number_of_words == 5 else "0",
                            "1" if depth == 2 else "0",
                        )
                    ),
                    "graph_source": graph_source,
                    "model": model,
                    "pair_name": pair_name,
                    "raw_row_path": str(raw_row_path),
                    "recall": (
                        1.0 - recall
                        if mismatched_batch_summary_recall and model == QWEN and index == 0
                        else recall
                    ),
                    "replication": replication,
                    "status": "finished",
                }
            )
            evaluated_rows.append(
                {
                    "Counter-Example": -1,
                    "Depth": depth,
                    "Example": example,
                    "Number of Words": number_of_words,
                    "Recall": recall,
                    "Repetition": replication,
                    "decoding_algorithm": "beam",
                    "decoding_condition": "beam_num_beams_6",
                    "embedding_model": model,
                    "graph_source": graph_source,
                    "model": model,
                    "pair_name": pair_name,
                    "provider": "hf",
                }
            )
        pd.DataFrame.from_records(evaluated_rows).to_csv(
            model_dir / "evaluated.csv",
            index=False,
        )
    pd.DataFrame.from_records(records).to_csv(results_root / "batch_summary.csv", index=False)


def _map_extension_recall(
    *,
    model: str,
    graph_source: str,
    pair_name: str,
    example: int,
    number_of_words: int,
    depth: int,
    replication: int,
) -> float:
    base = 0
    if model == QWEN:
        base += 1 if graph_source != "babs_johnson" else 0
        base += 1 if pair_name == "subgraph_2_to_subgraph_3" else 0
        base += 1 if example == 1 else 0
        base += 1 if number_of_words == 5 else 0
        base += 1 if depth == 2 else 0
        return 1.0 if base >= 4 else 0.0
    base += 1 if graph_source == "philip_marlowe" else 0
    base += 1 if pair_name != "subgraph_2_to_subgraph_1" else 0
    base += 1 if example == -1 else 0
    base += 1 if number_of_words == 5 else 0
    base += 1 if depth == 2 else 0
    base += 1 if replication == 1 else 0
    return 1.0 if base >= 5 else 0.0


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
