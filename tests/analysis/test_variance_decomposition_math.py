import pandas as pd
import pytest

from llm_conceptual_modeling.analysis._variance_decomposition_math import (
    _assert_balanced_cells,
    _assert_required_columns,
    _build_term_columns,
)


def test_build_term_columns_includes_main_and_interaction_terms() -> None:
    frame = pd.DataFrame(
        {
            "Greedy vs Beam/Contrastive": [-1.0, -1.0, 1.0, 1.0],
            "Beam Width": [-1.0, 1.0, -1.0, 1.0],
            "Example": [-1.0, 1.0, -1.0, 1.0],
        }
    )

    term_columns = _build_term_columns(
        frame,
        ("Greedy vs Beam/Contrastive", "Beam Width", "Example"),
    )

    assert [name for name, _columns in term_columns] == [
        "Greedy vs Beam/Contrastive",
        "Beam Width",
        "Example",
        "Greedy vs Beam/Contrastive & Example",
        "Beam Width & Example",
    ]


def test_assert_required_columns_raises_on_missing_columns() -> None:
    frame = pd.DataFrame({"Greedy vs Beam/Contrastive": [-1.0, 1.0]})

    with pytest.raises(ValueError, match="Missing required variance decomposition columns"):
        _assert_required_columns(frame, ["Greedy vs Beam/Contrastive", "Beam Width"])


def test_assert_balanced_cells_raises_on_unbalanced_design() -> None:
    frame = pd.DataFrame(
        {
            "Greedy vs Beam/Contrastive": [-1.0, -1.0, 1.0],
            "Beam Width": [-1.0, -1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="balanced design"):
        _assert_balanced_cells(frame, ("Greedy vs Beam/Contrastive", "Beam Width"))


def test_build_term_columns_supports_categorical_factors_with_orthogonal_basis() -> None:
    frame = pd.DataFrame(
        {
            "graph_source": [
                "babs_johnson",
                "babs_johnson",
                "clarice_starling",
                "clarice_starling",
                "philip_marlowe",
                "philip_marlowe",
            ],
            "Example": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        }
    )

    term_columns = _build_term_columns(frame, ("graph_source", "Example"))

    assert [name for name, _columns in term_columns] == [
        "graph_source",
        "Example",
        "graph_source & Example",
    ]
    graph_columns = term_columns[0][1]
    assert len(graph_columns) == 2
    for left_index, left in enumerate(graph_columns):
        for right in graph_columns[left_index + 1 :]:
            assert abs(float(left @ right)) <= 1e-8
