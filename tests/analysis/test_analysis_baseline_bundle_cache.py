from __future__ import annotations

from llm_conceptual_modeling.analysis import baseline_bundle


def test_compute_baseline_counts_reuses_identical_requests(monkeypatch) -> None:
    call_count = 0

    def fake_find_valid_connections(
        mother_edges: list[tuple[str, str]],
        subgraph1_edges: list[tuple[str, str]],
        subgraph2_edges: list[tuple[str, str]],
    ) -> set[tuple[str, str]]:
        nonlocal call_count
        call_count += 1
        return {("a", "x")}

    monkeypatch.setattr(baseline_bundle, "find_valid_connections", fake_find_valid_connections)
    monkeypatch.setattr(
        baseline_bundle,
        "_sample_baseline_edges",
        lambda **_: {("a", "x")},
    )

    kwargs = {
        "baseline_strategy": "random-k",
        "k": 1,
        "mother_edges": [("a", "x")],
        "subgraph1_edges": [("a", "b")],
        "subgraph2_edges": [("x", "y")],
        "ground_truth": {("a", "x")},
    }

    first = baseline_bundle._compute_baseline_counts(**kwargs)
    second = baseline_bundle._compute_baseline_counts(**kwargs)

    assert first == second == {"tp": 1, "fp": 0, "fn": 0}
    assert call_count == 1
