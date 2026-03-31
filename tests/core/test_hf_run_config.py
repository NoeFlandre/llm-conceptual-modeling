from pathlib import Path

import pytest

from llm_conceptual_modeling.hf_run_config import (
    load_hf_run_config,
    write_resolved_run_preview,
)


def _write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_hf_run_config_resolves_prompt_fragments_and_runtime_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    _write_config(
        config_path,
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 5
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  thinking_mode_by_model:
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 256
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  greedy:
    enabled: true
  beam:
    enabled: true
    num_beams: [2, 6]
  contrastive:
    enabled: true
    penalty_alpha: [0.2, 0.8]
    top_k: 4
inputs:
  graph_source: default
shared_fragments:
  assistant_role: "You are a helpful assistant."
algorithms:
  algo1:
    base_fragments: [assistant_role]
    factors:
      explanation:
        column: Explanation
        levels: [-1, 1]
        low_fragments: []
        high_fragments: [explanation_text]
    fragment_definitions:
      explanation_text: "Explain the notation."
    prompt_templates:
      body: "Task body."
""",
    )

    config = load_hf_run_config(config_path)

    assert config.runtime.temperature == 0.0
    assert config.models.chat_models == ["mistralai/Ministral-3-8B-Instruct-2512"]
    assert config.algorithms["algo1"].assemble_prompt(["explanation"]) == (
        "You are a helpful assistant. Explain the notation. Task body."
    )


def test_load_hf_run_config_rejects_unknown_fragment_reference(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    _write_config(
        config_path,
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 5
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  thinking_mode_by_model:
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 256
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  greedy:
    enabled: true
inputs:
  graph_source: default
shared_fragments:
  assistant_role: "You are a helpful assistant."
algorithms:
  algo1:
    base_fragments: [missing_fragment]
    factors: {}
    fragment_definitions: {}
    prompt_templates:
      body: "Task body."
""",
    )

    with pytest.raises(ValueError, match="missing_fragment"):
        load_hf_run_config(config_path)


def test_write_resolved_run_preview_writes_source_of_truth_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    output_dir = tmp_path / "preview"
    _write_config(
        config_path,
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 5
runtime:
  seed: 7
  temperature: 1.0
  quantization: none
  device_policy: cuda-only
  thinking_mode_by_model:
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 256
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  greedy:
    enabled: true
inputs:
  graph_source: default
shared_fragments:
  assistant_role: "You are a helpful assistant."
algorithms:
  algo1:
    base_fragments: [assistant_role]
    factors: {}
    fragment_definitions: {}
    prompt_templates:
      body: "Task body."
""",
    )

    config = load_hf_run_config(config_path)
    write_resolved_run_preview(config=config, output_dir=output_dir)

    resolved_yaml = (output_dir / "resolved_run_config.yaml").read_text(encoding="utf-8")
    resolved_plan = (output_dir / "resolved_run_plan.json").read_text(encoding="utf-8")
    prompt_preview = (
        output_dir / "prompt_preview" / "algo1" / "base.txt"
    ).read_text(encoding="utf-8")

    assert "temperature: 1.0" in resolved_yaml
    assert '"replications": 5' in resolved_plan
    assert "You are a helpful assistant. Task body." == prompt_preview


def test_load_hf_run_config_rejects_unacknowledged_unsupported_thinking_control(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.yaml"
    _write_config(
        config_path,
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 5
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 256
  thinking_mode_by_model:
    mistralai/Ministral-3-8B-Instruct-2512: disabled
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  greedy:
    enabled: true
inputs:
  graph_source: default
shared_fragments: {}
algorithms:
  algo1:
    base_fragments: []
    factors: {}
    fragment_definitions: {}
    prompt_templates:
      body: "Task body."
""",
    )

    with pytest.raises(ValueError, match="acknowledged-unsupported"):
        load_hf_run_config(config_path)


def test_load_hf_run_config_accepts_explicit_thinking_control_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    _write_config(
        config_path,
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 5
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 256
  thinking_mode_by_model:
    Qwen/Qwen3.5-9B: disabled
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
models:
  chat_models:
    - Qwen/Qwen3.5-9B
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  greedy:
    enabled: true
inputs:
  graph_source: default
shared_fragments: {}
algorithms:
  algo1:
    base_fragments: []
    factors: {}
    fragment_definitions: {}
    prompt_templates:
      body: "Task body."
""",
    )

    config = load_hf_run_config(config_path)

    assert config.runtime.thinking_mode_by_model == {
        "Qwen/Qwen3.5-9B": "disabled",
        "mistralai/Ministral-3-8B-Instruct-2512": "acknowledged-unsupported",
    }
