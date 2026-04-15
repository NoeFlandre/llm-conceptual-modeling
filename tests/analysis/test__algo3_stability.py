"""Tests for _algo3_stability module."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.analysis._algo3_stability import (
    algo3_level_stability_frame,
    algo3_level_variability_frame,
    algo3_metric_overview_row,
    algo3_metric_summary,
    algo3_metric_variability_row,
    algo3_pair_aware_condition_frame,
    algo3_raw_input_paths,
    resolve_source_input_path,
)


class TestAlgo3PairAwareConditionFrame:
    def test_produces_pair_name_column(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw.csv"
        raw_path.write_text(
            "Example,Counter-Example,Number of Words,Depth,Recall,"
            "Source Subgraph Name,Target Subgraph Name\n"
            "-1,-1,3,1,0.1,sg1,sg2\n"
            "1,-1,3,1,0.2,sg1,sg2\n"
            "-1,1,5,2,0.3,sg1,sg2\n"
            "1,1,5,2,0.4,sg1,sg2\n",
            encoding="utf-8",
        )
        frame = algo3_pair_aware_condition_frame([raw_path])
        assert "pair_name" in frame.columns
        assert set(frame["pair_name"]) == {"sg1_to_sg2"}

    def test_uses_existing_pair_name_column(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw.csv"
        raw_path.write_text(
            "Example,Counter-Example,Number of Words,Depth,Recall,pair_name\n"
            "-1,-1,3,1,0.1,sg1_to_sg2\n"
            "1,-1,3,1,0.2,sg1_to_sg2\n",
            encoding="utf-8",
        )
        frame = algo3_pair_aware_condition_frame([raw_path])
        assert "pair_name" in frame.columns
        assert len(frame) == 2

    def test_computes_cv_and_range_width(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw.csv"
        raw_path.write_text(
            "Example,Counter-Example,Number of Words,Depth,Recall,pair_name\n"
            "-1,-1,3,1,0.10,sg1_to_sg2\n"
            "1,-1,3,1,0.20,sg1_to_sg2\n",
            encoding="utf-8",
        )
        frame = algo3_pair_aware_condition_frame([raw_path])
        assert "coefficient_of_variation" in frame.columns
        assert "range_width" in frame.columns
        assert "metric" in frame.columns
        assert frame.iloc[0]["metric"] == "Recall"

    def test_ordered_columns(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw.csv"
        raw_path.write_text(
            "Example,Counter-Example,Number of Words,Depth,Recall,pair_name\n"
            "-1,-1,3,1,0.10,sg1_to_sg2\n",
            encoding="utf-8",
        )
        frame = algo3_pair_aware_condition_frame([raw_path])
        expected = [
            "source_input",
            "pair_name",
            "Depth",
            "Number of Words",
            "Example",
            "Counter-Example",
            "metric",
            "n",
            "mean",
            "sample_std",
            "min",
            "max",
            "range_width",
            "coefficient_of_variation",
        ]
        assert list(frame.columns) == expected


class TestAlgo3LevelStabilityFrame:
    def test_groups_by_level_column(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "Depth": [1, 1, 2, 2],
                "metric": ["Recall", "Recall", "Recall", "Recall"],
                "coefficient_of_variation": [0.1, 0.2, 0.3, 0.4],
                "range_width": [0.5, 0.6, 0.7, 0.8],
            }
        )
        result = algo3_level_stability_frame(frame, "Depth")
        assert set(result["Depth"]) == {1, 2}
        assert "mean_cv" in result.columns
        assert "median_cv" in result.columns
        assert "mean_range_width" in result.columns
        assert "max_range_width" in result.columns


class TestAlgo3LevelVariabilityFrame:
    def test_computes_varying_condition_share(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "Depth": [1, 1, 2],
                "metric": ["Recall", "Recall", "Recall"],
                "sample_std": [0.0, 0.1, 0.5],  # 1st not varying, others are
            }
        )
        result = algo3_level_variability_frame(frame, "Depth")
        depth1 = result[result["Depth"] == 1].iloc[0]
        assert depth1["varying_condition_count"] == 1
        assert depth1["varying_condition_share"] == pytest.approx(0.5)


class TestAlgo3MetricSummary:
    def test_returns_single_row(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "metric": ["Recall", "Recall"],
                "coefficient_of_variation": [0.1, 0.2],
                "range_width": [0.5, 0.6],
            }
        )
        result = algo3_metric_summary(frame)
        assert len(result) == 1
        assert "mean_cv" in result.columns
        assert "max_cv" in result.columns


class TestAlgo3MetricOverviewRow:
    def test_raises_on_empty_summary(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(columns=["metric", "coefficient_of_variation", "range_width"])
        with pytest.raises(ValueError, match="Expected exactly one"):
            algo3_metric_overview_row(frame)

    def test_returns_dict_with_expected_keys(self, tmp_path: Path) -> None:
        # algo3_metric_overview_row calls algo3_metric_summary internally, which
        # expects a condition_frame with metric, coefficient_of_variation, range_width columns
        condition_frame = pd.DataFrame(
            {
                "metric": ["Recall", "Recall"],
                "coefficient_of_variation": [0.1, 0.2],
                "range_width": [0.5, 0.6],
            }
        )
        result = algo3_metric_overview_row(condition_frame)
        assert result["algorithm"] == "algo3"
        assert result["metric"] == "Recall"
        assert "condition_count" in result


class TestAlgo3MetricVariabilityRow:
    def test_raises_on_empty_summary(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(columns=["metric", "sample_std"])
        with pytest.raises(ValueError, match="Expected exactly one"):
            algo3_metric_variability_row(frame)

    def test_computes_varying_condition_share(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "metric": ["Recall", "Recall"],
                "sample_std": [0.0, 0.1],  # one varying, one not
            }
        )
        result = algo3_metric_variability_row(frame)
        assert result["varying_condition_count"] == 1
        assert result["varying_condition_share"] == pytest.approx(0.5)


class TestResolveSourceInputPath:
    def test_returns_absolute_path_when_it_exists(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.csv"
        file_path.touch()
        result = resolve_source_input_path(str(file_path), tmp_path)
        assert result == file_path

    def test_resolves_relative_path_in_results_root(self, tmp_path: Path) -> None:
        results_root = tmp_path / "results"
        results_root.mkdir()
        file_path = results_root / "file.csv"
        file_path.touch()
        result = resolve_source_input_path("file.csv", results_root)
        assert result == file_path

    def test_strips_workspace_prefix(self, tmp_path: Path) -> None:
        results_root = tmp_path / "results"
        results_root.mkdir()
        file_path = results_root / "subdir" / "file.csv"
        file_path.parent.mkdir()
        file_path.touch()
        result = resolve_source_input_path(
            "/workspace/results/subdir/file.csv",
            results_root,
        )
        assert result == file_path

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        result = resolve_source_input_path("nonexistent.csv", tmp_path)
        assert result is None


class TestAlgo3RawInputPaths:
    def test_returns_empty_when_source_condition_lacks_source_input(self, tmp_path: Path) -> None:
        cond_path = tmp_path / "condition.csv"
        cond_path.write_text("metric,n,mean\nRecall,5,0.1", encoding="utf-8")
        result = algo3_raw_input_paths(cond_path, tmp_path)
        assert result == []

    def test_returns_empty_when_raw_lacks_pair_columns(self, tmp_path: Path) -> None:
        cond_path = tmp_path / "condition.csv"
        results_root = tmp_path / "results"
        results_root.mkdir()
        raw_path = results_root / "raw.csv"
        raw_path.write_text("metric,value\nRecall,0.1", encoding="utf-8")
        cond_path.write_text(f"source_input\n{raw_path}", encoding="utf-8")
        result = algo3_raw_input_paths(cond_path, results_root)
        assert result == []

    def test_returns_resolved_paths_when_pair_columns_present(self, tmp_path: Path) -> None:
        cond_path = tmp_path / "condition.csv"
        results_root = tmp_path / "results"
        results_root.mkdir()
        raw_path = results_root / "raw.csv"
        raw_path.write_text(
            "Example,Counter-Example,Number of Words,Depth,Recall,pair_name\n"
            "-1,-1,3,1,0.1,sg1_to_sg2\n",
            encoding="utf-8",
        )
        cond_path.write_text(f"source_input\n{raw_path}", encoding="utf-8")
        result = algo3_raw_input_paths(cond_path, results_root)
        assert result == [raw_path]
