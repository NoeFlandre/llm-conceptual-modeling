from llm_conceptual_modeling.algo2.method import (
    Method2ExecutionResult,
    execute_method2,
)
from llm_conceptual_modeling.common.client_protocols import EmbeddingClient
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        embeddings_by_label = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "bridge_a": [1.0, 0.0],
            "bridge_b": [0.0, 1.0],
            "bridge_c": [1.0, 1.0],
            "Cholesterol": [1.0, 0.0],
            "Weight loss": [0.0, 1.0],
        }
        return {text: embeddings_by_label[text] for text in texts}


def test_execute_method2_runs_expansion_then_normalizes_suggested_edges() -> None:
    thesaurus = load_algo2_thesaurus()
    embedding_client: EmbeddingClient = FakeEmbeddingClient()
    proposal_calls: list[list[str]] = []
    edge_calls: list[list[str]] = []

    def propose_labels(current_labels: list[str]) -> list[str]:
        proposal_calls.append(current_labels)
        if len(current_labels) == 2:
            return ["bridge_a", "bridge_b"]
        return ["bridge_c"]

    def suggest_edges(expanded_label_context: list[str]) -> list[tuple[str, str]]:
        edge_calls.append(expanded_label_context)
        return [
            ("Cholesterol", "Weight loss"),
            ("bridge_c", "beta"),
        ]

    actual = execute_method2(
        seed_labels=["alpha", "beta"],
        propose_labels=propose_labels,
        suggest_edges=suggest_edges,
        embedding_client=embedding_client,
        convergence_threshold=0.01,
        thesaurus=thesaurus,
    )

    assert actual == Method2ExecutionResult(
        expanded_labels=["bridge_a", "bridge_b", "bridge_c"],
        raw_edges=[
            ("Cholesterol", "Weight loss"),
            ("bridge_c", "beta"),
        ],
        normalized_edges=[
            ("Blood saturated fatty acid level", "Obesity"),
            ("bridge_c", "beta"),
        ],
        final_similarity=0.9023689270621825,
        iteration_count=3,
    )
    assert proposal_calls == [
        ["alpha", "beta"],
        ["alpha", "beta", "bridge_a", "bridge_b"],
        ["alpha", "beta", "bridge_a", "bridge_b", "bridge_c"],
    ]
    assert edge_calls == [
        ["alpha", "beta", "bridge_a", "bridge_b", "bridge_c"],
    ]


def test_execute_method2_verifies_final_edges_after_normalization() -> None:
    thesaurus = load_algo2_thesaurus()
    embedding_client: EmbeddingClient = FakeEmbeddingClient()
    verification_calls: list[list[tuple[str, str]]] = []

    def propose_labels(current_labels: list[str]) -> list[str]:
        if len(current_labels) == 2:
            return ["bridge_a", "bridge_b"]
        return ["bridge_c"]

    def suggest_edges(expanded_label_context: list[str]) -> list[tuple[str, str]]:
        _ = expanded_label_context
        return [
            ("Cholesterol", "Weight loss"),
            ("bridge_c", "beta"),
        ]

    def verify_edges(candidate_edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
        verification_calls.append(candidate_edges)
        return [candidate_edges[1]]

    actual = execute_method2(
        seed_labels=["alpha", "beta"],
        propose_labels=propose_labels,
        suggest_edges=suggest_edges,
        verify_edges=verify_edges,
        embedding_client=embedding_client,
        convergence_threshold=0.01,
        thesaurus=thesaurus,
    )

    assert actual.normalized_edges == [("bridge_c", "beta")]
    assert verification_calls == [
        [
            ("Blood saturated fatty acid level", "Obesity"),
            ("bridge_c", "beta"),
        ]
    ]
