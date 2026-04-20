from llm_conceptual_modeling.algo1 import types as algo1_types
from llm_conceptual_modeling.algo2 import method as algo2_method
from llm_conceptual_modeling.algo2 import thesaurus as algo2_thesaurus
from llm_conceptual_modeling.algo3 import evaluation as algo3_evaluation
from llm_conceptual_modeling.analysis import variability
from llm_conceptual_modeling.common import baseline, connection_eval, mistral
from llm_conceptual_modeling.common import types as common_types
from llm_conceptual_modeling.common.types import Edge
from llm_conceptual_modeling.hf_batch import types as hf_batch_types


def test_edge_alias_has_one_canonical_definition() -> None:
    assert algo1_types.Edge is Edge
    assert algo2_method.Edge is Edge
    assert algo2_thesaurus.Edge is Edge
    assert algo3_evaluation.Edge is Edge
    assert baseline.Edge is Edge
    assert connection_eval.Edge is Edge
    assert mistral.Edge is Edge
    assert variability.Edge is Edge
    assert hf_batch_types.Edge is Edge
    assert common_types.Edge is Edge
