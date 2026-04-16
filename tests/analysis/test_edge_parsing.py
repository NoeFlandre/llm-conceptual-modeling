from __future__ import annotations

import pytest

from llm_conceptual_modeling.analysis import _edge_parsing


def test_parse_edges_cached_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_: str) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(_edge_parsing, "parse_python_literal", boom)

    with pytest.raises(RuntimeError, match="boom"):
        _edge_parsing._parse_edges_cached("[('a', 'b')]")


def test_parse_algo3_edge_list_cached_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_: str) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(_edge_parsing.ast, "literal_eval", boom)

    with pytest.raises(RuntimeError, match="boom"):
        _edge_parsing._parse_algo3_edge_list_cached("[('a', 'b')]")
