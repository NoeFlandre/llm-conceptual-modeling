from pathlib import Path

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_batch_types import HFRunSpec
from llm_conceptual_modeling.hf_state.resume_state import resume_priority_key


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        supports_thinking_toggle=False,
        quantization="none",
        device="cuda",
        dtype="bfloat16",
        context_limit=None,
    )


def test_hf_state_resume_state_package_imports_and_prioritizes_models(tmp_path: Path) -> None:
    spec = HFRunSpec(
        algorithm="algo1",
        model="Qwen/Qwen3.5-9B",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        pair_name="sg1_sg2",
        condition_bits="00000",
        condition_label="greedy",
        replication=0,
        prompt_factors={},
        prompt_bundle=None,
        decoding=DecodingConfig(algorithm="greedy", temperature=0.0),
        input_payload={"graph": [], "subgraph1": [], "subgraph2": []},
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        seed=1,
        runtime_profile=_runtime_profile(),
        max_new_tokens_by_schema={"edge_list": 32, "vote_list": 16, "label_list": 16},
        context_policy={"resume_pass_mode": "throughput"},
    )

    key = resume_priority_key(
        spec=spec,
        history={"by_pair": {}, "by_pair_condition": {}, "by_family": {}},
        run_dir=tmp_path / "run",
    )

    assert isinstance(key, tuple)
    assert len(key) == 9
