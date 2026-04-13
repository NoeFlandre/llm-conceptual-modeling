from llm_conceptual_modeling.hf_batch.outputs import write_aggregated_outputs
from llm_conceptual_modeling.hf_batch.planning import plan_paper_batch_specs
from llm_conceptual_modeling.hf_batch.prompts import build_prompt_bundle, render_prompt
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import (
    RecordingChatClient,
    runtime_details,
    slugify_model,
    write_text,
)


def test_hf_batch_package_exposes_refactored_modules() -> None:
    assert callable(write_aggregated_outputs)
    assert callable(plan_paper_batch_specs)
    assert callable(build_prompt_bundle)
    assert callable(render_prompt)
    assert callable(RecordingChatClient)
    assert callable(runtime_details)
    assert callable(slugify_model)
    assert callable(write_text)
    assert HFRunSpec.__name__ == "HFRunSpec"
    assert RuntimeResult.__name__ == "dict"


def test_hf_batch_planning_is_implemented_in_the_package_module() -> None:
    assert plan_paper_batch_specs.__module__ == "llm_conceptual_modeling.hf_batch.planning"


def test_hf_batch_outputs_is_implemented_in_the_package_module() -> None:
    assert write_aggregated_outputs.__module__ == "llm_conceptual_modeling.hf_batch.outputs"


def test_hf_batch_prompts_is_implemented_in_the_package_module() -> None:
    assert build_prompt_bundle.__module__ == "llm_conceptual_modeling.hf_batch.prompts"


def test_hf_batch_utils_is_implemented_in_the_package_module() -> None:
    assert runtime_details.__module__ == "llm_conceptual_modeling.hf_batch.utils"
