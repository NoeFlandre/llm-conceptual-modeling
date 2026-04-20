from itertools import zip_longest

from llm_conceptual_modeling.common.types import Edge


def build_cove_prompt(candidate_edges: list[Edge]) -> str:
    prompt = (
        "Return whether a causal relationship exists between the source and target "
        "concepts for each pair in a list. "
        "For example, given [('smoking', 'cancer'), ('ice cream sales', 'shark attacks')], "
        "return ['Y', 'N'] with no other text. "
        f"Candidate pairs: {candidate_edges}"
    )
    return prompt


def apply_cove_verification(
    candidate_edges: list[Edge],
    verification_votes: list[str],
) -> list[Edge]:
    verified_edges: list[Edge] = []

    for candidate_edge, verification_vote in zip_longest(
        candidate_edges,
        verification_votes,
        fillvalue=None,
    ):
        if candidate_edge is None:
            break
        vote_is_yes = verification_vote == "Y"
        if not vote_is_yes:
            continue
        verified_edges.append(candidate_edge)

    return verified_edges
