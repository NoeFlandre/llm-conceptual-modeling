from __future__ import annotations

import llm_conceptual_modeling.algo3.types as types
from llm_conceptual_modeling.algo3 import method, mistral, tree


def test_child_proposer_protocol_is_shared_with_tree() -> None:
    assert method.TreeExpansionFunction.__module__ == "llm_conceptual_modeling.algo3.types"
    assert method.ChildDictionaryProposer.__module__ == "llm_conceptual_modeling.algo3.types"
    assert tree.ChildProposer.__module__ == "llm_conceptual_modeling.algo3.types"
    assert mistral.ChildProposer is types.ChildProposer
    assert types.ChildProposer.__module__ == "llm_conceptual_modeling.algo3.types"
