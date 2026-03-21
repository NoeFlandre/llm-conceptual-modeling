from llm_conceptual_modeling.algo1.method import (
    Method1ExecutionResult,
    execute_method1,
)


def test_execute_method1_runs_generation_then_cove_filtering() -> None:
    generation_calls: list[dict[str, object]] = []
    verification_calls: list[list[tuple[str, str]]] = []

    def generate_edges(
        *,
        subgraph1: list[tuple[str, str]],
        subgraph2: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        generation_calls.append(
            {
                "subgraph1": subgraph1,
                "subgraph2": subgraph2,
            }
        )
        return [
            ("alpha", "bridge_a"),
            ("bridge_b", "gamma"),
        ]

    def verify_edges(candidate_edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
        verification_calls.append(candidate_edges)
        return [("alpha", "bridge_a")]

    actual = execute_method1(
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        generate_edges=generate_edges,
        verify_edges=verify_edges,
    )

    assert actual == Method1ExecutionResult(
        candidate_edges=[
            ("alpha", "bridge_a"),
            ("bridge_b", "gamma"),
        ],
        verified_edges=[("alpha", "bridge_a")],
    )
    assert generation_calls == [
        {
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
        }
    ]
    assert verification_calls == [
        [
            ("alpha", "bridge_a"),
            ("bridge_b", "gamma"),
        ]
    ]
