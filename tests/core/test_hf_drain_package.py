from llm_conceptual_modeling.hf_drain.planning import build_drain_plan
from llm_conceptual_modeling.hf_drain.runtime import run_drain_supervisor


def test_hf_drain_package_imports() -> None:
    assert callable(build_drain_plan)
    assert callable(run_drain_supervisor)
