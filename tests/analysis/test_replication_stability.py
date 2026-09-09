from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.analysis.replication_stability import (
    ReplicationStabilityResult,
    _as_tuple,
    _coefficient_of_variation,
    _condition_budget,
    _cumulative_means,
    _model_summary,
    _prepare_input_frame,
    _read_manifest_frames,
    _reject_duplicate_repetitions,
    _requirement_status,
    _requires_more_runs,
    _sample_std,
    _source_label,
    _write_result,
    analyze_replication_stability,
    estimate_required_runs,
)
from llm_conceptual_modeling.cli import main

_FACTOR_VALUES = {
    "Example": -1,
    "Counter-Example": 1,
    "Number of Words": 3,
    "Depth": 1,
}


def _write_input(path: Path, *, pair: str, values: list[float]) -> None:
    rows = []
    for repetition, recall in enumerate(values):
        rows.append(
            {
                "Repetition": repetition,
                **_FACTOR_VALUES,
                "Source Subgraph Name": pair,
                "Target Subgraph Name": "target",
                "Recall": recall,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_manifest(path: Path, inputs: list[tuple[str, Path]]) -> None:
    pd.DataFrame(
        [{"model": model, "input_path": str(input_path)} for model, input_path in inputs]
    ).to_csv(path, index=False)


def test_analysis_keeps_pairs_separate_and_labels_zero_mean_cells(tmp_path: Path) -> None:
    input_path = tmp_path / "model_a.csv"
    _write_input(input_path, pair="source_a", values=[0.1, 0.2, 0.3])
    zero_input_path = tmp_path / "model_a_zero.csv"
    _write_input(zero_input_path, pair="source_b", values=[0.0, 0.0, 0.0])
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [("model-a", input_path), ("model-a", zero_input_path)],
    )

    result = analyze_replication_stability(manifest_path, tmp_path / "out")

    assert set(result.condition_budget["Source Subgraph Name"]) == {"source_a", "source_b"}
    varying = result.condition_budget[
        result.condition_budget["Source Subgraph Name"] == "source_a"
    ].iloc[0]
    assert list(result.condition_budget.columns) == [
        "source_input",
        "model",
        "Example",
        "Counter-Example",
        "Number of Words",
        "Depth",
        "Source Subgraph Name",
        "Target Subgraph Name",
        "metric",
        "observed_runs",
        "mean",
        "sample_std",
        "min",
        "max",
        "range_width",
        "coefficient_of_variation",
        "required_total_runs",
        "additional_runs_needed",
        "requirement_status",
    ]
    assert varying["source_input"] == str(input_path)
    assert varying["model"] == "model-a"
    assert varying["Example"] == -1
    assert varying["Counter-Example"] == 1
    assert varying["Number of Words"] == 3
    assert varying["Depth"] == 1
    assert varying["Target Subgraph Name"] == "target"
    assert varying["metric"] == "Recall"
    assert varying["observed_runs"] == 3
    assert varying["mean"] == pytest.approx(0.2)
    assert varying["sample_std"] == pytest.approx(0.1)
    assert varying["min"] == pytest.approx(0.1)
    assert varying["max"] == pytest.approx(0.3)
    assert varying["range_width"] == pytest.approx(0.2)
    assert varying["coefficient_of_variation"] == pytest.approx(0.5)
    assert varying["required_total_runs"] == 385
    assert varying["additional_runs_needed"] == 382
    assert varying["requirement_status"] == "requires_more_runs"

    zero_mean = result.condition_budget[
        result.condition_budget["Source Subgraph Name"] == "source_b"
    ].iloc[0]
    assert zero_mean["source_input"] == str(zero_input_path)
    assert zero_mean["metric"] == "Recall"
    assert pd.isna(zero_mean["coefficient_of_variation"])
    assert pd.isna(zero_mean["additional_runs_needed"])
    assert pd.isna(zero_mean["required_total_runs"])
    assert zero_mean["requirement_status"] == "zero_mean_not_estimable"


def test_analysis_writes_cumulative_means_and_model_summary(tmp_path: Path) -> None:
    varying_input = tmp_path / "model_a.csv"
    _write_input(varying_input, pair="source_a", values=[0.1, 0.2, 0.3])
    constant_input = tmp_path / "model_b.csv"
    _write_input(constant_input, pair="source_b", values=[0.4, 0.4, 0.4])
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [("model-a", varying_input), ("model-b", constant_input)])

    output_dir = tmp_path / "nested" / "deeper" / "out"
    result = analyze_replication_stability(manifest_path, output_dir)

    cumulative = result.cumulative_means
    model_a_means = cumulative[cumulative["model"] == "model-a"]["cumulative_mean"].tolist()
    assert list(cumulative.columns) == [
        "source_input",
        "model",
        "Example",
        "Counter-Example",
        "Number of Words",
        "Depth",
        "Source Subgraph Name",
        "Target Subgraph Name",
        "metric",
        "prefix_runs",
        "cumulative_mean",
    ]
    assert cumulative[cumulative["model"] == "model-a"]["prefix_runs"].tolist() == [1, 2, 3]
    assert cumulative[cumulative["model"] == "model-a"]["source_input"].tolist() == [
        str(varying_input)
    ] * 3
    assert cumulative[cumulative["model"] == "model-a"]["metric"].tolist() == [
        "Recall"
    ] * 3
    assert model_a_means == pytest.approx([0.1, 0.15, 0.2])

    summary = result.model_summary.set_index("model")
    assert list(result.model_summary.columns) == [
        "model",
        "metric",
        "condition_count",
        "observed_runs",
        "zero_mean_conditions",
        "nonzero_mean_conditions",
        "varying_conditions",
        "varying_condition_share",
        "varying_nonzero_conditions",
        "mean_cv_nonzero",
        "median_cv_nonzero",
        "max_cv_nonzero",
        "conditions_requiring_more_runs",
        "median_required_total_runs",
        "max_required_total_runs",
    ]
    assert summary.loc["model-a", "condition_count"] == 1
    assert summary.loc["model-a", "metric"] == "Recall"
    assert summary.loc["model-a", "observed_runs"] == 3
    assert summary.loc["model-a", "zero_mean_conditions"] == 0
    assert summary.loc["model-a", "nonzero_mean_conditions"] == 1
    assert summary.loc["model-a", "varying_conditions"] == 1
    assert summary.loc["model-a", "varying_condition_share"] == 1
    assert summary.loc["model-a", "varying_nonzero_conditions"] == 1
    assert summary.loc["model-a", "mean_cv_nonzero"] == pytest.approx(0.5)
    assert summary.loc["model-a", "median_cv_nonzero"] == pytest.approx(0.5)
    assert summary.loc["model-a", "max_cv_nonzero"] == pytest.approx(0.5)
    assert summary.loc["model-a", "conditions_requiring_more_runs"] == 1
    assert summary.loc["model-a", "median_required_total_runs"] == 385
    assert summary.loc["model-a", "max_required_total_runs"] == 385
    assert summary.loc["model-b", "metric"] == "Recall"
    assert summary.loc["model-b", "observed_runs"] == 3
    assert summary.loc["model-b", "zero_mean_conditions"] == 0
    assert summary.loc["model-b", "nonzero_mean_conditions"] == 1
    assert summary.loc["model-b", "varying_conditions"] == 0
    assert summary.loc["model-b", "varying_condition_share"] == 0
    assert summary.loc["model-b", "varying_nonzero_conditions"] == 0
    assert summary.loc["model-b", "mean_cv_nonzero"] == pytest.approx(0)
    assert summary.loc["model-b", "median_cv_nonzero"] == pytest.approx(0)
    assert summary.loc["model-b", "max_cv_nonzero"] == pytest.approx(0)
    assert summary.loc["model-b", "conditions_requiring_more_runs"] == 0
    assert summary.loc["model-b", "median_required_total_runs"] == 3
    assert summary.loc["model-b", "max_required_total_runs"] == 3

    for filename, expected_columns in (
        (
            "condition_budget.csv",
            list(result.condition_budget.columns),
        ),
        (
            "cumulative_means.csv",
            list(result.cumulative_means.columns),
        ),
        ("model_summary.csv", list(result.model_summary.columns)),
    ):
        written = pd.read_csv(tmp_path / "nested" / "deeper" / "out" / filename)
        assert list(written.columns) == expected_columns
        assert "Unnamed: 0" not in written.columns
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "condition_budget.csv",
        "cumulative_means.csv",
        "model_summary.csv",
    ]
    analyze_replication_stability(manifest_path, output_dir)


def test_cli_analyze_replication_stability_writes_three_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "model.csv"
    _write_input(input_path, pair="source", values=[0.1, 0.2, 0.3])
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [("model-a", input_path)])
    output_dir = tmp_path / "nested" / "out"

    exit_code = main(
        [
            "analyze",
            "replication-stability",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "condition_budget.csv").exists()
    assert (output_dir / "cumulative_means.csv").exists()
    assert (output_dir / "model_summary.csv").exists()
    assert pd.read_csv(output_dir / "condition_budget.csv")["source_input"].tolist() == [
        str(input_path)
    ]


def test_write_result_explicitly_omits_dataframe_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []

    def capture_to_csv(
        _frame: pd.DataFrame, path: Path, *, index: bool
    ) -> None:
        calls.append((Path(path).name, index))

    monkeypatch.setattr(pd.DataFrame, "to_csv", capture_to_csv)
    empty_frame = pd.DataFrame()

    _write_result(
        ReplicationStabilityResult(empty_frame, empty_frame, empty_frame),
        tmp_path,
    )

    assert calls == [
        ("condition_budget.csv", False),
        ("cumulative_means.csv", False),
        ("model_summary.csv", False),
    ]


def test_estimate_required_runs_handles_degenerate_cells_and_invalid_parameters() -> None:
    assert (
        estimate_required_runs(
            observed_runs=5,
            mean=0.0,
            sample_std=0.2,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        is None
    )
    assert (
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=0.0,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        == 5
    )
    assert (
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=1.0,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        == 38416
    )
    assert (
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=0.1,
            relative_half_width_target=0.05,
            z_score=0.5,
        )
        == 25
    )
    assert (
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=None,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        is None
    )
    with pytest.raises(ValueError) as error:
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=0.1,
            relative_half_width_target=0.0,
            z_score=1.96,
        )
    assert str(error.value) == "relative_half_width_target must be positive."
    with pytest.raises(ValueError) as error:
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=0.1,
            relative_half_width_target=0.05,
            z_score=0.0,
        )
    assert str(error.value) == "z_score must be positive."
    with pytest.raises(ValueError) as error:
        estimate_required_runs(
            observed_runs=0,
            mean=0.2,
            sample_std=0.1,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
    assert str(error.value) == "observed_runs must be positive."
    with pytest.raises(ValueError) as error:
        estimate_required_runs(
            observed_runs=5,
            mean=float("nan"),
            sample_std=0.1,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
    assert str(error.value) == "mean must be finite."
    with pytest.raises(ValueError) as error:
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=-0.1,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
    assert str(error.value) == "sample_std must be non-negative."
    assert (
        estimate_required_runs(
            observed_runs=5,
            mean=0.2,
            sample_std=float("nan"),
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        is None
    )


def test_analysis_labels_insufficient_observations_and_rejects_duplicate_repetitions(
    tmp_path: Path,
) -> None:
    one_run_path = tmp_path / "one_run.csv"
    _write_input(one_run_path, pair="source", values=[0.2])
    one_run_manifest = tmp_path / "one_run_manifest.csv"
    _write_manifest(one_run_manifest, [("model-a", one_run_path)])

    result = analyze_replication_stability(one_run_manifest, tmp_path / "one_run_out")

    row = result.condition_budget.iloc[0]
    assert row["requirement_status"] == "insufficient_observations"
    assert pd.isna(row["required_total_runs"])

    duplicate_path = tmp_path / "duplicate.csv"
    duplicate = pd.read_csv(one_run_path)
    pd.concat([duplicate, duplicate], ignore_index=True).to_csv(duplicate_path, index=False)
    duplicate_manifest = tmp_path / "duplicate_manifest.csv"
    _write_manifest(duplicate_manifest, [("model-a", duplicate_path)])

    with pytest.raises(ValueError) as error:
        analyze_replication_stability(duplicate_manifest, tmp_path / "duplicate_out")
    assert str(error.value) == (
        "Each exact condition must contain at most one row per repetition."
    )


def test_analysis_rejects_missing_input_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.csv"
    pd.DataFrame({"Repetition": [0], "Recall": [0.2]}).to_csv(input_path, index=False)
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [("model-a", input_path)])

    with pytest.raises(ValueError, match="Missing required replication-stability input columns"):
        analyze_replication_stability(manifest_path, tmp_path / "out")


def test_analysis_rejects_invalid_manifest_and_input_values(tmp_path: Path) -> None:
    empty_manifest_path = tmp_path / "empty_manifest.csv"
    pd.DataFrame(columns=["model", "input_path"]).to_csv(empty_manifest_path, index=False)
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(empty_manifest_path, tmp_path / "empty_out")
    assert str(error.value) == "The replication-stability manifest is empty."

    missing_input_manifest = tmp_path / "missing_input_manifest.csv"
    _write_manifest(missing_input_manifest, [("model-a", tmp_path / "missing.csv")])
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(missing_input_manifest, tmp_path / "missing_out")
    assert str(error.value) == (
        f"Manifest input does not exist: {tmp_path / 'missing.csv'}"
    )

    missing_manifest_column = tmp_path / "missing_manifest_column.csv"
    pd.DataFrame([{"model": "model-a"}]).to_csv(missing_manifest_column, index=False)
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(missing_manifest_column, tmp_path / "bad_manifest")
    assert str(error.value) == "Missing required manifest columns: ['input_path']"

    relative_input = tmp_path / "relative.csv"
    _write_input(relative_input, pair="source", values=[0.2, 0.2])
    relative_manifest = tmp_path / "relative_manifest.csv"
    pd.DataFrame([{"model": "model-a", "input_path": "relative.csv"}]).to_csv(
        relative_manifest,
        index=False,
    )
    analyze_replication_stability(relative_manifest, tmp_path / "relative_out")

    for field, value, _message in (
        ("model", " ", "must not be empty"),
        ("input_path", "", "must not be missing"),
    ):
        invalid_manifest = tmp_path / f"invalid_{field}.csv"
        row = {"model": "model-a", "input_path": str(relative_input)}
        row[field] = value
        pd.DataFrame([row]).to_csv(invalid_manifest, index=False)
        with pytest.raises(ValueError) as error:
            analyze_replication_stability(invalid_manifest, tmp_path / f"{field}_out")
        expected = (
            f"Manifest {field} values must not be empty."
            if field == "model"
            else f"Manifest {field} values must not be missing."
        )
        assert str(error.value) == expected

    missing_value_input = tmp_path / "missing_value.csv"
    missing_value = pd.read_csv(relative_input)
    missing_value.loc[0, "Recall"] = None
    missing_value.to_csv(missing_value_input, index=False)
    missing_value_manifest = tmp_path / "missing_value_manifest.csv"
    _write_manifest(missing_value_manifest, [("model-a", missing_value_input)])
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(missing_value_manifest, tmp_path / "missing_value_out")
    assert str(error.value) == (
        "Replication-stability input contains missing required values: "
        f"{missing_value_input}"
    )

    non_integer_input = tmp_path / "non_integer.csv"
    non_integer = pd.read_csv(relative_input)
    non_integer["Repetition"] = non_integer["Repetition"].astype(float)
    non_integer.loc[0, "Repetition"] = 0.5
    non_integer.to_csv(non_integer_input, index=False)
    non_integer_manifest = tmp_path / "non_integer_manifest.csv"
    _write_manifest(non_integer_manifest, [("model-a", non_integer_input)])
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(non_integer_manifest, tmp_path / "non_integer_out")
    assert str(error.value) == f"Repetition values must be integer-valued: {non_integer_input}"

    non_finite_input = tmp_path / "non_finite.csv"
    non_finite = pd.read_csv(relative_input)
    non_finite.loc[0, "Recall"] = float("inf")
    non_finite.to_csv(non_finite_input, index=False)
    non_finite_manifest = tmp_path / "non_finite_manifest.csv"
    _write_manifest(non_finite_manifest, [("model-a", non_finite_input)])
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(non_finite_manifest, tmp_path / "non_finite_out")
    assert str(error.value) == f"Metric values must be finite: {non_finite_input}"

    malformed_repetition = pd.read_csv(relative_input)
    malformed_repetition["Repetition"] = malformed_repetition["Repetition"].astype(
        object
    )
    malformed_repetition.loc[0, "Repetition"] = "not-a-number"
    malformed_repetition_path = tmp_path / "malformed_repetition.csv"
    malformed_repetition.to_csv(malformed_repetition_path, index=False)
    malformed_repetition_manifest = tmp_path / "malformed_repetition_manifest.csv"
    _write_manifest(malformed_repetition_manifest, [("model-a", malformed_repetition_path)])
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(
            malformed_repetition_manifest,
            tmp_path / "malformed_repetition_out",
        )
    assert str(error.value) == (
        f"Repetition values must be integer-valued: {malformed_repetition_path}"
    )

    malformed_metric = pd.read_csv(relative_input)
    malformed_metric["Recall"] = malformed_metric["Recall"].astype(object)
    malformed_metric.loc[0, "Recall"] = "not-a-number"
    malformed_metric_path = tmp_path / "malformed_metric.csv"
    malformed_metric.to_csv(malformed_metric_path, index=False)
    malformed_metric_manifest = tmp_path / "malformed_metric_manifest.csv"
    _write_manifest(malformed_metric_manifest, [("model-a", malformed_metric_path)])
    with pytest.raises(ValueError) as error:
        analyze_replication_stability(
            malformed_metric_manifest,
            tmp_path / "malformed_metric_out",
        )
    assert str(error.value) == f"Metric values must be finite: {malformed_metric_path}"

    with pytest.raises(ValueError, match="Expected 2 grouped values"):
        _as_tuple("one", 2)


def test_internal_helpers_define_boundary_and_label_contracts() -> None:
    assert _sample_std(pd.Series([0.1, 0.3])) == pytest.approx(0.1414213562)
    assert _sample_std(pd.Series([0.1])) is None
    assert _coefficient_of_variation(0.2, 0.1) == pytest.approx(0.5)
    assert _coefficient_of_variation(0.0, 0.1) is None
    assert _coefficient_of_variation(0.2, None) is None
    assert _requires_more_runs(None, 5) is False
    assert _requires_more_runs(5, 5) is False
    assert _requires_more_runs(6, 5) is True
    assert (
        _requirement_status(
            observed_runs=1,
            mean=0.2,
            sample_std=0.1,
            required=1,
        )
        == "insufficient_observations"
    )
    assert (
        _requirement_status(
            observed_runs=2,
            mean=0.2,
            sample_std=0.1,
            required=2,
        )
        == "stable_at_observed_runs"
    )
    assert _source_label(pd.Series(["source_b", "source_a", "source_b"])) == (
        "source_a|source_b"
    )

    summary = _model_summary(
        pd.DataFrame(
            [
                {
                    "model": "model",
                    "mean": 1.0,
                    "range_width": 1.0,
                    "coefficient_of_variation": 0.5,
                    "observed_runs": 3,
                    "required_total_runs": 5,
                    "requirement_status": "requires_more_runs",
                },
                {
                    "model": "model",
                    "mean": 2.0,
                    "range_width": 0.0,
                    "coefficient_of_variation": 0.0,
                    "observed_runs": 3,
                    "required_total_runs": 3,
                    "requirement_status": "stable_at_observed_runs",
                },
            ]
        ),
        metric="Recall",
    )
    assert summary.loc[0, "zero_mean_conditions"] == 0
    assert summary.loc[0, "nonzero_mean_conditions"] == 2
    assert summary.loc[0, "varying_conditions"] == 1
    assert summary.loc[0, "varying_condition_share"] == pytest.approx(0.5)
    assert summary.loc[0, "varying_nonzero_conditions"] == 1
    assert summary.loc[0, "conditions_requiring_more_runs"] == 1


def test_duplicate_detection_uses_exact_condition_keys() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "model-a",
                **_FACTOR_VALUES,
                "Source Subgraph Name": "source",
                "Target Subgraph Name": "target",
                "Repetition": 0,
                "source_input": "first.csv",
            },
            {
                "model": "model-a",
                **_FACTOR_VALUES,
                "Source Subgraph Name": "source",
                "Target Subgraph Name": "target",
                "Repetition": 0,
                "source_input": "second.csv",
            },
        ]
    )

    with pytest.raises(ValueError) as error:
        _reject_duplicate_repetitions(frame)
    assert str(error.value) == (
        "Each exact condition must contain at most one row per repetition."
    )


def test_input_preparation_normalizes_numeric_types_and_preserves_source(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "Repetition": "0.0",
                **_FACTOR_VALUES,
                "Source Subgraph Name": "source",
                "Target Subgraph Name": "target",
                "Recall": 1.0,
            },
            {
                "Repetition": "1.0",
                **_FACTOR_VALUES,
                "Source Subgraph Name": "source",
                "Target Subgraph Name": "target",
                "Recall": 2.0,
            },
        ]
    )

    prepared = _prepare_input_frame(
        frame,
        model="model-a",
        source_input="input.csv",
        metric_column="Recall",
    )

    assert pd.api.types.is_integer_dtype(prepared["Repetition"])
    assert pd.api.types.is_float_dtype(prepared["Recall"])
    assert prepared["model"].tolist() == ["model-a", "model-a"]
    assert prepared["source_input"].tolist() == ["input.csv", "input.csv"]


def test_manifest_frames_reset_indexes_and_join_source_labels(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    _write_input(first, pair="source", values=[0.1, 0.2, 0.3])
    second = tmp_path / "second.csv"
    second_frame = pd.read_csv(first)
    second_frame["Repetition"] += 3
    second_frame.to_csv(second, index=False)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [("model-a", first), ("model-a", second)])

    frames = _read_manifest_frames(manifest, metric_column="Recall")

    assert len(frames) == 1
    assert frames[0].index.tolist() == list(range(6))
    assert set(frames[0]["source_input"]) == {str(first), str(second)}
    result = analyze_replication_stability(manifest, tmp_path / "out")
    assert result.condition_budget.iloc[0]["source_input"] == f"{first}|{second}"
    assert result.condition_budget.iloc[0]["observed_runs"] == 6


def test_empty_internal_frames_keep_output_schemas() -> None:
    columns = ["model", *_FACTOR_VALUES, "Source Subgraph Name", "Target Subgraph Name"]
    empty = pd.DataFrame(
        columns=["source_input", "Repetition", *columns, "Recall"]
    )
    condition = _condition_budget(
        [empty],
        metric_column="Recall",
        relative_half_width_target=0.05,
        z_score=1.96,
    )
    cumulative = _cumulative_means([empty], metric_column="Recall")
    summary = _model_summary(condition, metric="Recall")

    assert list(condition.columns) == [
        "source_input",
        "model",
        "Example",
        "Counter-Example",
        "Number of Words",
        "Depth",
        "Source Subgraph Name",
        "Target Subgraph Name",
        "metric",
        "observed_runs",
        "mean",
        "sample_std",
        "min",
        "max",
        "range_width",
        "coefficient_of_variation",
        "required_total_runs",
        "additional_runs_needed",
        "requirement_status",
    ]
    assert list(cumulative.columns) == [
        "source_input",
        "model",
        "Example",
        "Counter-Example",
        "Number of Words",
        "Depth",
        "Source Subgraph Name",
        "Target Subgraph Name",
        "metric",
        "prefix_runs",
        "cumulative_mean",
    ]
    assert list(summary.columns) == [
        "model",
        "metric",
        "condition_count",
        "observed_runs",
        "zero_mean_conditions",
        "nonzero_mean_conditions",
        "varying_conditions",
        "varying_condition_share",
        "varying_nonzero_conditions",
        "mean_cv_nonzero",
        "median_cv_nonzero",
        "max_cv_nonzero",
        "conditions_requiring_more_runs",
        "median_required_total_runs",
        "max_required_total_runs",
    ]
