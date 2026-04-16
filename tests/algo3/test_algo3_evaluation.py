from __future__ import annotations

import pytest

from llm_conceptual_modeling.algo3 import evaluation


def test_parse_edge_list_propagates_unexpected_literal_eval_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_: str) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(evaluation.ast, "literal_eval", boom)

    with pytest.raises(RuntimeError, match="boom"):
        evaluation.parse_edge_list("[('a', 'b')]")
