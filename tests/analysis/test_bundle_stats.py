"""Tests for _bundle_stats pure DataFrame transformers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._bundle_stats import (
    _annotate,
    _build_breadth_distribution,
    _build_failure_rates,
    _build_parsed_edge_counts,
    _build_parsed_edge_quartiles,
    _build_validity_summary,
    _extract_model,
)


class TestAnnotate:
    def test_adds_summary_type_column(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _annotate(df, "failure_rates")
        assert "summary_type" in result.columns
        assert (result["summary_type"] == "failure_rates").all()


class TestExtractModel:
    def test_extracts_model_from_legacy_and_results_paths(self) -> None:
        assert (
            _extract_model(
                "/algo1/gpt-5/raw/x.csv",
            )
            == "gpt-5"
        )
        assert (
            _extract_model(
                "/private/var/tmp/results/algo1/gpt-5/evaluated/x.csv",
            )
            == "gpt-5"
        )
        assert (
            _extract_model(
                "/private/var/tmp/legacy/algo2/mistral/raw/y.csv",
            )
            == "mistral"
        )

    def test_extracts_model_from_path_objects(self) -> None:
        path = Path("/private/var/tmp/results/algo1/gpt-5/evaluated/x.csv")

        assert _extract_model(path) == "gpt-5"


class TestBuildValiditySummary:
    def test_zero_failure_rate_when_all_valid(self) -> None:
        df = pd.DataFrame({
            "source_input": ["/algo1/gpt-5/raw/x.csv", "/algo1/gpt-5/raw/y.csv"],
            "failure_category": ["valid_output", "valid_output"],
        })
        result = _build_validity_summary(df, "algo1")
        assert result.iloc[0]["failure_rate"] == 0.0
        assert result.iloc[0]["valid_count"] == 2

    def test_empty_output_counted(self) -> None:
        df = pd.DataFrame({
            "source_input": ["/algo1/gpt-5/raw/x.csv"],
            "failure_category": ["empty_output"],
        })
        result = _build_validity_summary(df, "algo1")
        assert result.iloc[0]["failure_rate"] == 1.0
        assert result.iloc[0]["empty_count"] == 1


class TestBuildBreadthDistribution:
    def test_mean_minus_median(self) -> None:
        df = pd.DataFrame({
            "source_input": ["/algo1/gpt-5/raw/x.csv"] * 3,
            "failure_category": ["valid_output"] * 3,
            "parsed_edge_count": [5, 10, 15],
        })
        result = _build_breadth_distribution(df, "algo1")
        assert result.iloc[0]["mean"] == 10.0
        assert result.iloc[0]["median"] == 10.0
        assert result.iloc[0]["mean_minus_median"] == 0.0


class TestBuildParsedEdgeQuartiles:
    def test_iqr_is_q3_minus_q1(self) -> None:
        df = pd.DataFrame({
            "algorithm": ["algo1", "algo1", "algo1", "algo1", "algo1"],
            "model": ["gpt-5", "gpt-5", "gpt-5", "gpt-5", "gpt-5"],
            "failure_category": ["valid_output"] * 5,
            "parsed_edge_count": [1, 2, 3, 4, 5],
        })
        result = _build_parsed_edge_quartiles(df)
        assert "iqr" in result.columns
        assert result.iloc[0]["q1"] == 2.0
        assert result.iloc[0]["q3"] == 4.0


class TestBuildFailureRates:
    def test_failure_rate_fraction(self) -> None:
        df = pd.DataFrame({
            "algorithm": ["algo1"] * 4,
            "model": ["gpt-5"] * 4,
            "failure_category": [
                "valid_output",
                "valid_output",
                "malformed_output",
                "empty_output",
            ],
        })
        result = _build_failure_rates(df)
        assert result.iloc[0]["failure_rate"] == 0.5


class TestBuildParsedEdgeCounts:
    def test_rounds_mean_to_two_decimals(self) -> None:
        df = pd.DataFrame({
            "algorithm": ["algo1"] * 3,
            "model": ["gpt-5"] * 3,
            "failure_category": ["valid_output"] * 3,
            "parsed_edge_count": [3, 4, 5],
        })
        result = _build_parsed_edge_counts(df)
        assert result.iloc[0]["mean"] == 4.0
        assert result.iloc[0]["median"] == 4.0
