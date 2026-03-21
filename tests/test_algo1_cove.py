from llm_conceptual_modeling.algo1.cove import (
    apply_cove_verification,
    build_cove_prompt,
)


def test_build_cove_prompt_mentions_y_n_output_only() -> None:
    candidate_edges = [
        ("smoking", "cancer"),
        ("ice cream sales", "shark attacks"),
    ]

    actual = build_cove_prompt(candidate_edges)

    assert "Return whether a causal relationship exists" in actual
    assert "return ['Y', 'N'] with no other text" in actual
    assert "smoking" in actual
    assert "shark attacks" in actual


def test_apply_cove_verification_keeps_only_edges_with_y_vote() -> None:
    candidate_edges = [
        ("alpha", "bridge_a"),
        ("alpha", "bridge_b"),
        ("bridge_c", "beta"),
    ]
    verification_votes = ["Y", "N", "Y"]

    actual = apply_cove_verification(candidate_edges, verification_votes)

    assert actual == [
        ("alpha", "bridge_a"),
        ("bridge_c", "beta"),
    ]


def test_apply_cove_verification_treats_missing_votes_as_no() -> None:
    candidate_edges = [
        ("alpha", "bridge_a"),
        ("alpha", "bridge_b"),
        ("bridge_c", "beta"),
    ]
    verification_votes = ["Y"]

    actual = apply_cove_verification(candidate_edges, verification_votes)

    assert actual == [("alpha", "bridge_a")]
