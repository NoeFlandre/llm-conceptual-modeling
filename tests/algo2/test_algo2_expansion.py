from llm_conceptual_modeling.algo2.expansion import (
    ExpansionIteration,
    average_best_match_similarity,
    has_converged,
    run_label_expansion,
)


def test_has_converged_uses_absolute_similarity_delta() -> None:
    assert has_converged(previous_similarity=0.21, current_similarity=0.215, threshold=0.01) is True
    assert has_converged(previous_similarity=0.21, current_similarity=0.24, threshold=0.01) is False


def test_run_label_expansion_stops_when_similarity_delta_reaches_threshold() -> None:
    observed_candidate_sets: list[list[str]] = []
    similarity_values = iter([0.20, 0.205])

    def propose_labels(current_labels: list[str]) -> list[str]:
        if len(current_labels) == 2:
            return ["bridge_a", "bridge_b"]
        return ["bridge_c"]

    def measure_similarity(candidate_labels: list[str], seed_labels: list[str]) -> float:
        observed_candidate_sets.append(candidate_labels)
        return next(similarity_values)

    actual = run_label_expansion(
        seed_labels=["alpha", "beta"],
        propose_labels=propose_labels,
        measure_similarity=measure_similarity,
        threshold=0.01,
    )

    assert actual.expanded_labels == ["bridge_a", "bridge_b", "bridge_c"]
    assert len(actual.iterations) == 2
    assert actual.iterations == [
        ExpansionIteration(
            proposed_labels=["bridge_a", "bridge_b"],
            accumulated_labels=["bridge_a", "bridge_b"],
            similarity=0.20,
        ),
        ExpansionIteration(
            proposed_labels=["bridge_c"],
            accumulated_labels=["bridge_a", "bridge_b", "bridge_c"],
            similarity=0.205,
        ),
    ]
    assert observed_candidate_sets == [
        ["bridge_a", "bridge_b"],
        ["bridge_a", "bridge_b", "bridge_c"],
    ]


def test_average_best_match_similarity_averages_best_seed_match_per_candidate() -> None:
    embeddings = {
        "seed_a": [1.0, 0.0],
        "seed_b": [0.0, 1.0],
        "candidate_a": [1.0, 0.0],
        "candidate_b": [1.0, 1.0],
    }

    actual = average_best_match_similarity(
        candidate_labels=["candidate_a", "candidate_b"],
        seed_labels=["seed_a", "seed_b"],
        embeddings_by_label=embeddings,
    )

    assert round(actual, 6) == round((1.0 + 0.7071067811865475) / 2, 6)


def test_average_best_match_similarity_returns_zero_for_empty_candidate_set() -> None:
    embeddings = {"seed_a": [1.0, 0.0]}

    actual = average_best_match_similarity(
        candidate_labels=[],
        seed_labels=["seed_a"],
        embeddings_by_label=embeddings,
    )

    assert actual == 0.0
