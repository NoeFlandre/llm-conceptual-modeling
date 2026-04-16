from __future__ import annotations

import importlib
import sys
from types import ModuleType

from llm_conceptual_modeling.algo2 import expansion, method, mistral, types
from llm_conceptual_modeling.common.client_protocols import EmbeddingClient


def test_label_proposal_function_protocol_is_shared() -> None:
    assert method.LabelProposalFunction is expansion.LabelProposalFunction
    assert method.LabelProposalFunction.__module__ == "llm_conceptual_modeling.algo2.types"


def test_method_protocols_come_from_shared_types_module() -> None:
    assert method.EdgeSuggestionFunction.__module__ == "llm_conceptual_modeling.algo2.types"
    assert method.EdgeVerificationFunction.__module__ == "llm_conceptual_modeling.algo2.types"


def test_method_embedding_client_is_imported_from_shared_protocol_module(
    monkeypatch,
) -> None:
    fake_embeddings = ModuleType("llm_conceptual_modeling.algo2.embeddings")
    fake_embeddings.EmbeddingClient = object()
    fake_embeddings.compute_average_best_match_similarity = lambda **_: 0.0
    monkeypatch.setitem(sys.modules, "llm_conceptual_modeling.algo2.embeddings", fake_embeddings)

    spec = importlib.util.spec_from_file_location("temp_algo2_method", method.__file__)
    temp_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(temp_module)

    assert temp_module.EmbeddingClient is EmbeddingClient
    assert EmbeddingClient.__module__ == "llm_conceptual_modeling.common.client_protocols"


def test_mistral_protocol_aliases_point_at_canonical_types() -> None:
    assert mistral.LabelProposer is types.LabelProposalFunction
    assert mistral.EdgeSuggester is types.EdgeSuggestionFunction
