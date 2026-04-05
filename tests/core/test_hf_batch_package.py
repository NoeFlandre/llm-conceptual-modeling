from llm_conceptual_modeling.hf_batch.outputs import write_aggregated_outputs
from llm_conceptual_modeling.hf_batch.planning import plan_paper_batch_specs
from llm_conceptual_modeling.hf_batch.prompts import build_prompt_bundle
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import RecordingChatClient, slugify_model


def test_hf_batch_package_exposes_refactored_modules() -> None:
    assert callable(write_aggregated_outputs)
    assert callable(plan_paper_batch_specs)
    assert callable(build_prompt_bundle)
    assert callable(RecordingChatClient)
    assert callable(slugify_model)
    assert HFRunSpec.__name__ == "HFRunSpec"
    assert RuntimeResult.__name__ == "dict"
