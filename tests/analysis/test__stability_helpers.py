"""Tests for _stability_helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from llm_conceptual_modeling.analysis._stability_helpers import (
    frame_to_overview_records,
    patch_algorithm_rows,
    slugify,
)


class TestSlugify:
    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            ("Explanation", "explanation"),
            ("Array/List(1/-1)", "array_list1_1"),
            ("Tag/Adjacency(1/-1)", "tag_adjacency1_1"),
            ("Number of Words", "number_of_words"),
            ("Counter-Example", "counter_example"),
            ("Depth", "depth"),
            ("a b", "a_b"),
            ("a/b", "a_b"),
            ("(test)", "test"),
            ("a-b-c", "a_b_c"),
            ("__all__", "all"),
        ],
    )
    def test_slugify(self, input_value: str, expected: str) -> None:
        assert slugify(input_value) == expected


class TestFrameToOverviewRecords:
    def test_empty_frame(self) -> None:
        assert frame_to_overview_records(pd.DataFrame()) == []

    def test_single_row(self) -> None:
        frame = pd.DataFrame(
            {
                "algorithm": ["algo1"],
                "metric": ["accuracy"],
                "condition_count": [576],
                "mean_cv": [0.00002],
                "median_cv": [0.0],
                "max_cv": [0.012],
                "mean_range_width": [0.00005],
                "max_range_width": [0.026],
            }
        )
        records = frame_to_overview_records(frame)
        assert len(records) == 1
        assert records[0]["algorithm"] == "algo1"
        assert records[0]["condition_count"] == 576
        assert records[0]["mean_cv"] == pytest.approx(0.00002)


class TestPatchAlgorithmRows:
    def test_empty_frame(self) -> None:
        result = patch_algorithm_rows(pd.DataFrame(), {"algorithm": "algo1", "metric": "accuracy"})
        assert result.empty

    def test_replaces_existing_row(self) -> None:
        frame = pd.DataFrame(
            [
                {"algorithm": "algo1", "metric": "accuracy", "condition_count": 100},
                {"algorithm": "algo3", "metric": "Recall", "condition_count": 50},
            ]
        )
        replacement = {"algorithm": "algo1", "metric": "accuracy", "condition_count": 576}
        result = patch_algorithm_rows(frame, replacement)
        assert len(result) == 2
        algo1_row = result[(result["algorithm"] == "algo1") & (result["metric"] == "accuracy")]
        assert len(algo1_row) == 1
        assert int(algo1_row.iloc[0]["condition_count"]) == 576

    def test_appends_new_row(self) -> None:
        frame = pd.DataFrame(
            [
                {"algorithm": "algo1", "metric": "accuracy", "condition_count": 100},
            ]
        )
        replacement = {"algorithm": "algo2", "metric": "precision", "condition_count": 200}
        result = patch_algorithm_rows(frame, replacement)
        assert len(result) == 2  # original algo1 row + appended algo2 row
        assert set(result["algorithm"]) == {"algo1", "algo2"}
        assert int(result[result["algorithm"] == "algo2"].iloc[0]["condition_count"]) == 200

    def test_missing_columns_passes_through(self) -> None:
        frame = pd.DataFrame([{"algorithm": "algo1", "metric": "accuracy", "condition_count": 100}])
        result = patch_algorithm_rows(frame, {"algorithm": "algo1", "metric": "accuracy"})
        assert len(result) == 1
