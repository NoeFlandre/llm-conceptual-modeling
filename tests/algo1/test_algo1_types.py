from llm_conceptual_modeling.algo1 import method, mistral
from llm_conceptual_modeling.algo1 import types as algo1_types


def test_algo1_edge_protocols_have_a_single_canonical_module() -> None:
    assert method.EdgeGenerator is algo1_types.EdgeGenerator
    assert method.CoveVerifier is algo1_types.CoveVerifier
    assert mistral.EdgeGenerator is algo1_types.EdgeGenerator
    assert mistral.CoveVerifier is algo1_types.CoveVerifier
    assert algo1_types.EdgeGenerator.__module__ == "llm_conceptual_modeling.algo1.types"
    assert algo1_types.CoveVerifier.__module__ == "llm_conceptual_modeling.algo1.types"
