from pathlib import Path

import pytest

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    RuntimeProfile,
    supports_decoding_config,
)
from llm_conceptual_modeling.hf_batch_planning import plan_paper_batch_specs
from llm_conceptual_modeling.hf_run_config import load_hf_run_config


def test_supports_decoding_config_rejects_qwen_contrastive() -> None:
    assert not supports_decoding_config(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
    )
    assert supports_decoding_config(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="beam", num_beams=2),
    )


def test_plan_paper_batch_specs_skips_qwen_contrastive_runs_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 1
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  thinking_mode_by_model:
    Qwen/Qwen3.5-9B: disabled
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 256
models:
  chat_models:
    - Qwen/Qwen3.5-9B
  embedding_model: Qwen/Qwen3-Embedding-0.6B
decoding:
  greedy:
    enabled: true
  beam:
    enabled: true
    num_beams: [2]
  contrastive:
    enabled: true
    penalty_alpha: [0.2]
    top_k: 4
inputs:
  graph_source: default
shared_fragments: {}
algorithms:
  algo2:
    pair_names: [sg1_sg2]
    base_fragments: []
    factors:
      explanation:
        column: Explanation
        levels: [-1, 1]
        runtime_field: include_explanation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      example:
        column: Example
        levels: [-1, 1]
        runtime_field: include_example
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      counterexample:
        column: Counterexample
        levels: [-1, 1]
        runtime_field: include_counterexample
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      array:
        column: Array/List(1/-1)
        levels: [-1, 1]
        runtime_field: use_array_representation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      adjacency:
        column: Tag/Adjacency(1/-1)
        levels: [-1, 1]
        runtime_field: use_adjacency_notation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      convergence:
        column: Convergence
        levels: [-1, 1]
        runtime_field: use_relaxed_convergence
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
    fragment_definitions: {}
    prompt_templates:
      label_expansion: "Expand labels."
      edge_suggestion: "Suggest edges."
      cove_verification: "Verify edges."
""",
        encoding="utf-8",
    )
    config = load_hf_run_config(config_path)

    specs = plan_paper_batch_specs(
        models=config.models.chat_models,
        embedding_model=config.models.embedding_model,
        replications=config.run.replications,
        algorithms=("algo2",),
        config=config,
        runtime_profile_provider=lambda _model: RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=True,
            context_limit=None,
        ),
    )

    assert specs
    assert {spec.decoding.algorithm for spec in specs} == {"greedy", "beam"}
    assert all(spec.decoding.algorithm != "contrastive" for spec in specs)


def test_qwen_contrastive_chat_client_is_rejected() -> None:
    from llm_conceptual_modeling.common.hf_transformers import HFTransformersChatClient

    with pytest.raises(ValueError, match="unsupported for model Qwen/Qwen3.5-9B"):
        HFTransformersChatClient(
            model="Qwen/Qwen3.5-9B",
            decoding_config=DecodingConfig(
                algorithm="contrastive",
                penalty_alpha=0.2,
                top_k=4,
            ),
            tokenizer=object(),
            model_object=object(),
            device="cuda",
        )
