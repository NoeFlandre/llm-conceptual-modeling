from llm_conceptual_modeling.hf_batch.run_artifacts import build_run_summary, write_run_artifacts


def test_build_run_summary_module() -> None:
    assert build_run_summary.__module__ == "llm_conceptual_modeling.hf_batch.run_artifacts"


def test_write_run_artifacts_module() -> None:
    assert write_run_artifacts.__module__ == "llm_conceptual_modeling.hf_batch.run_artifacts"
