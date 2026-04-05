from pathlib import Path

import pytest

from llm_conceptual_modeling.common.hf_transformers import RuntimeProfile
from llm_conceptual_modeling.hf_batch.planning import plan_paper_batch_specs
from llm_conceptual_modeling.hf_run_config import (
    exclude_decoding_conditions_from_payload,
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


def test_load_hf_run_config_accepts_resolved_preview_decoding_list(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    output_dir = tmp_path / "preview"
    _write_config(
        config_path,
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 2
runtime:
  seed: 13
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
  beam:
    enabled: true
    num_beams: [2]
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

    resolved_config = load_hf_run_config(output_dir / "resolved_run_config.yaml")

    assert [item.algorithm for item in resolved_config.decoding] == ["greedy", "beam"]
    assert resolved_config.decoding[0].temperature == 1.0
    assert resolved_config.decoding[1].num_beams == 2
    assert resolved_config.algorithms["algo1"].assemble_prompt([]) == (
        "You are a helpful assistant. Task body."
    )


def test_qwen_batch_runtime_config_is_hardened_for_resume() -> None:
    config = load_hf_run_config("results/hf-paper-batch-algo1-qwen/runtime_config.yaml")

    assert config.run.output_root == "/workspace/results/hf-paper-batch-algo1-qwen"
    assert [decoding.algorithm for decoding in config.decoding] == ["greedy", "beam", "beam"]
    assert [decoding.num_beams for decoding in config.decoding] == [None, 2, 6]
    assert [decoding.penalty_alpha for decoding in config.decoding] == [None, None, None]
    assert config.runtime.context_policy["generation_timeout_seconds"] == 60.0
    assert config.runtime.context_policy["resume_pass_mode"] == "retry-timeouts"
    assert config.runtime.context_policy["retry_timeout_failures_on_resume"] is True
    assert config.algorithms["algo1"].pair_names == ["sg1_sg2", "sg2_sg3", "sg3_sg1"]
    assert config.algorithms["algo1"].prompt_templates["direct_edge"] == ""


def test_exclude_decoding_conditions_from_payload_removes_named_branch() -> None:
    payload = {
        "runtime": {"temperature": 0.0},
        "decoding": {
            "greedy": {"enabled": True},
            "beam": {"enabled": True, "num_beams": [2, 6]},
            "contrastive": {"enabled": True, "penalty_alpha": [0.2, 0.8], "top_k": 4},
        },
    }

    exclude_decoding_conditions_from_payload(
        payload,
        excluded_condition_labels={"contrastive_penalty_alpha_0.8"},
    )

    assert payload["decoding"] == [
        {
            "algorithm": "greedy",
            "num_beams": None,
            "penalty_alpha": None,
            "top_k": None,
            "temperature": 0.0,
        },
        {
            "algorithm": "beam",
            "num_beams": 2,
            "penalty_alpha": None,
            "top_k": None,
            "temperature": 0.0,
        },
        {
            "algorithm": "beam",
            "num_beams": 6,
            "penalty_alpha": None,
            "top_k": None,
            "temperature": 0.0,
        },
        {
            "algorithm": "contrastive",
            "num_beams": None,
            "penalty_alpha": 0.2,
            "top_k": 4,
            "temperature": 0.0,
        },
    ]


def test_qwen_algo1_only_resume_config_is_hardened_for_retry_all() -> None:
    config = load_hf_run_config("qwen_algo1_only.yaml")

    assert config.models.chat_models == ["Qwen/Qwen3.5-9B"]
    assert config.algorithms["algo1"].pair_names == ["sg1_sg2", "sg2_sg3", "sg3_sg1"]
    assert config.runtime.context_policy["generation_timeout_seconds"] == 600
    assert config.runtime.context_policy["resume_pass_mode"] == "retry-all"
    assert config.runtime.context_policy["retry_timeout_failures_on_resume"] is True
    assert config.runtime.context_policy["retry_oom_failures_on_resume"] is True
    assert config.runtime.context_policy["retry_infrastructure_failures_on_resume"] is True


def test_algo2_olmo_batch_runtime_config_is_hardened_for_resume() -> None:
    config = load_hf_run_config("configs/hf_transformers_algo2_olmo.yaml")

    assert config.models.chat_models == ["allenai/Olmo-3-7B-Instruct"]
    assert [decoding.algorithm for decoding in config.decoding] == ["greedy", "beam"]
    assert [decoding.num_beams for decoding in config.decoding] == [None, 2]
    assert config.algorithms["algo2"].pair_names == ["sg1_sg2", "sg2_sg3", "sg3_sg1"]
    assert config.runtime.context_policy["generation_timeout_seconds"] == 60
    assert config.runtime.context_policy["retry_structural_failures_on_resume"] is True
    assert config.runtime.context_policy["max_requests_per_worker_process"] == 16


def test_algo1_olmo_batch_runtime_config_is_hardened_for_resume() -> None:
    config = load_hf_run_config("results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml")

    assert config.models.chat_models == ["allenai/Olmo-3-7B-Instruct"]
    assert config.runtime.context_policy["generation_timeout_seconds"] == 180.0
    assert config.runtime.context_policy["resume_pass_mode"] == "throughput"
    assert config.runtime.context_policy["retry_structural_failures_on_resume"] is True
    assert config.runtime.context_policy["max_requests_per_worker_process"] == 25


def test_write_resolved_run_preview_writes_all_condition_prompt_variants(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    output_dir = tmp_path / "preview"
    _write_config(
        config_path,
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
shared_fragments: {}
algorithms:
  algo1:
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
    fragment_definitions:
      system_message: "You are a helpful assistant who understands Knowledge Maps."
      explanation_fixed: "Explanation fixed."
      explanation_variable_RDF: "Explanation RDF."
      explanation_variable_edges: "Explanation edges."
      explanation_variable_matrix: "Explanation matrix."
      explanation_variable_markup: "Explanation markup."
      example_variable_RDF: "Example RDF."
      example_variable_edges: "Example edges."
      example_variable_matrix: "Example matrix."
      example_variable_markup: "Example markup."
      counter_example_variable_RDF: "Counter RDF."
      counter_example_variable_edges: "Counter edges."
      counter_example_variable_matrix: "Counter matrix."
      counter_example_variable_markup: "Counter markup."
      task_fixed_sub_1: "Map 1:"
      task_fixed_sub_2: "Map 2:"
      task_fixed_sub_3: "Task."
      conclusion_fixed: "Conclusion."
      cove_verification_template: "Verify {candidate_edges}"
    prompt_templates:
      direct_edge: ""
      cove_verification: ""
""",
    )

    config = load_hf_run_config(config_path)
    write_resolved_run_preview(config=config, output_dir=output_dir)

    condition_root = output_dir / "prompt_preview" / "algo1" / "conditions"
    condition_dirs = sorted(path.name for path in condition_root.iterdir() if path.is_dir())
    low_prompt = (condition_root / "00" / "direct_edge.txt").read_text(encoding="utf-8")
    high_prompt = (condition_root / "11" / "direct_edge.txt").read_text(encoding="utf-8")
    condition_matrix = (output_dir / "condition_matrix.csv").read_text(encoding="utf-8")

    assert condition_dirs == ["00", "01", "10", "11"]
    assert "Explanation fixed." not in low_prompt
    assert "Example RDF." not in low_prompt
    assert "Explanation fixed." in high_prompt
    assert "Example RDF." in high_prompt
    assert (
        "algorithm,model,decoding_condition,replication,pair_name,condition_bits"
        in condition_matrix
    )


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


def test_checked_in_algo3_single_model_configs_are_locally_rent_ready(tmp_path: Path) -> None:
    repo_root = Path("/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling")
    cases = [
        (
            repo_root / "configs" / "hf_transformers_algo3_olmo.yaml",
            "allenai/Olmo-3-7B-Instruct",
            "/workspace/results/hf-paper-batch-algo3-olmo-current",
            {"greedy", "beam", "contrastive"},
        ),
        (
            repo_root / "configs" / "hf_transformers_algo3_qwen.yaml",
            "Qwen/Qwen3.5-9B",
            "/workspace/results/hf-paper-batch-algo3-qwen-current",
            {"greedy", "beam"},
        ),
        (
            repo_root / "configs" / "hf_transformers_algo3_mistral.yaml",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "/workspace/results/hf-paper-batch-algo3-mistral-current",
            {"greedy", "beam", "contrastive"},
        ),
    ]

    for config_path, expected_model, expected_output_root, expected_algorithms in cases:
        config = load_hf_run_config(config_path)

        assert config.run.output_root == expected_output_root
        assert config.models.chat_models == [expected_model]
        assert set(config.algorithms) == {"algo3"}
        expected_generation_timeout = (
            60 if expected_model == "allenai/Olmo-3-7B-Instruct" else 600
        )
        actual_timeout = config.runtime.context_policy["generation_timeout_seconds"]
        assert actual_timeout == expected_generation_timeout
        if expected_model == "allenai/Olmo-3-7B-Instruct":
            assert config.runtime.context_policy["retry_structural_failures_on_resume"] is True
        assert config.algorithms["algo3"].pair_names == [
            "subgraph_1_to_subgraph_3",
            "subgraph_2_to_subgraph_1",
            "subgraph_2_to_subgraph_3",
        ]
        planned_specs = plan_paper_batch_specs(
            models=config.models.chat_models,
            embedding_model=config.models.embedding_model,
            replications=config.run.replications,
            algorithms=["algo3"],
            config=config,
            runtime_profile_provider=lambda _model: RuntimeProfile(
                device="cuda",
                dtype="bfloat16",
                quantization="none",
                supports_thinking_toggle=True,
                context_limit=None,
            ),
        )
        assert {spec.decoding.algorithm for spec in planned_specs} == expected_algorithms

        preview_dir = tmp_path / config_path.stem
        write_resolved_run_preview(config=config, output_dir=preview_dir)

        assert (preview_dir / "resolved_run_config.yaml").exists()
        assert (preview_dir / "resolved_run_plan.json").exists()
        assert (preview_dir / "prompt_preview" / "algo3" / "tree_expansion.txt").exists()


def test_checked_in_algo2_single_model_configs_are_locally_rent_ready(tmp_path: Path) -> None:
    repo_root = Path("/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling")
    cases = [
        (
            repo_root / "configs" / "hf_transformers_algo2_olmo.yaml",
            "allenai/Olmo-3-7B-Instruct",
            "/workspace/results/hf-paper-batch-algo2-olmo-current",
            {"greedy", "beam"},
        ),
        (
            repo_root / "configs" / "hf_transformers_algo2_qwen.yaml",
            "Qwen/Qwen3.5-9B",
            "/workspace/results/hf-paper-batch-algo2-qwen-current",
            {"greedy", "beam", "contrastive"},
        ),
        (
            repo_root / "configs" / "hf_transformers_algo2_mistral.yaml",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "/workspace/results/hf-paper-batch-algo2-mistral-current",
            {"greedy", "beam", "contrastive"},
        ),
    ]

    for config_path, expected_model, expected_output_root, expected_algorithms in cases:
        config = load_hf_run_config(config_path)

        assert config.run.output_root == expected_output_root
        assert config.models.chat_models == [expected_model]
        assert set(config.algorithms) == {"algo2"}
        expected_generation_timeout = (
            60 if expected_model == "allenai/Olmo-3-7B-Instruct" else 600
        )
        actual_timeout = config.runtime.context_policy["generation_timeout_seconds"]
        assert actual_timeout == expected_generation_timeout
        assert config.runtime.context_policy["resume_pass_mode"] == "retry-timeouts"
        assert config.runtime.context_policy["retry_timeout_failures_on_resume"] is True
        if expected_model == "allenai/Olmo-3-7B-Instruct":
            assert config.runtime.context_policy["retry_structural_failures_on_resume"] is True
        assert config.algorithms["algo2"].pair_names == [
            "sg1_sg2",
            "sg2_sg3",
            "sg3_sg1",
        ]
        if config_path.name == "hf_transformers_algo2_mistral.yaml":
            assert [item.algorithm for item in config.decoding] == [
                "greedy",
                "beam",
                "contrastive",
            ]
            assert [item.num_beams for item in config.decoding] == [None, 2, None]
            assert [item.penalty_alpha for item in config.decoding] == [None, None, 0.2]
        if config_path.name == "hf_transformers_algo2_qwen.yaml":
            assert [item.algorithm for item in config.decoding] == [
                "greedy",
                "beam",
                "beam",
                "contrastive",
                "contrastive",
            ]
        planned_specs = plan_paper_batch_specs(
            models=config.models.chat_models,
            embedding_model=config.models.embedding_model,
            replications=config.run.replications,
            algorithms=["algo2"],
            config=config,
            runtime_profile_provider=lambda _model: RuntimeProfile(
                device="cuda",
                dtype="bfloat16",
                quantization="none",
                supports_thinking_toggle=True,
                context_limit=None,
            ),
        )
        assert {spec.decoding.algorithm for spec in planned_specs} == expected_algorithms

        preview_dir = tmp_path / config_path.stem
        write_resolved_run_preview(config=config, output_dir=preview_dir)

        assert (preview_dir / "resolved_run_config.yaml").exists()
        assert (preview_dir / "resolved_run_plan.json").exists()
        assert (preview_dir / "prompt_preview" / "algo2" / "label_expansion.txt").exists()
