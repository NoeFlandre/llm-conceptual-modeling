from llm_conceptual_modeling.hf_resume.preflight import build_resume_preflight_report
from llm_conceptual_modeling.hf_resume.profile import resolve_resume_profile
from llm_conceptual_modeling.hf_resume.sweep import build_resume_sweep_report


def test_hf_resume_package_imports() -> None:
    assert resolve_resume_profile("qwen").profile_name == "qwen-safe"
    assert callable(build_resume_preflight_report)
    assert callable(build_resume_sweep_report)
