from pathlib import Path

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    RuntimeProfile,
    supports_decoding_config,
)
from llm_conceptual_modeling.hf_batch.planning import (
    default_runtime_profile_provider,
    plan_paper_batch,
    plan_paper_batch_specs,
    select_run_spec,
)
from llm_conceptual_modeling.hf_run_config import load_hf_run_config


def test_supports_decoding_config_allows_qwen_contrastive() -> None:
    assert supports_decoding_config(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
    )
    assert supports_decoding_config(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="beam", num_beams=2),
    )


def test_hf_batch_planning_package_exports_runtime_profile_provider() -> None:
    profile = default_runtime_profile_provider("Qwen/Qwen3.5-9B")

    assert profile == RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=True,
        context_limit=None,
    )


def test_plan_paper_batch_specs_includes_qwen_contrastive_runs_from_config(tmp_path: Path) -> None:
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
    assert {spec.decoding.algorithm for spec in specs} == {"greedy", "beam", "contrastive"}
    assert any(spec.decoding.algorithm == "contrastive" for spec in specs)


def test_manifest_for_spec_includes_graph_source() -> None:
    from llm_conceptual_modeling.hf_batch.types import HFRunSpec
    from llm_conceptual_modeling.hf_batch.utils import manifest_for_spec

    spec = HFRunSpec(
        algorithm="algo3",
        model="Qwen/Qwen3.5-9B",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="beam", num_beams=6),
        replication=0,
        pair_name="subgraph_1_to_subgraph_3",
        condition_bits="000",
        condition_label="beam_num_beams_6",
        prompt_factors={},
        raw_context={"pair_name": "subgraph_1_to_subgraph_3", "Repetition": 0},
        input_payload={"source_graph": [], "target_graph": [], "mother_graph": []},
        runtime_profile=RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=True,
            context_limit=None,
        ),
        graph_source="clarice_starling",
    )

    assert manifest_for_spec(spec)["graph_source"] == "clarice_starling"


def test_derive_run_seed_preserves_default_seed_material() -> None:
    from llm_conceptual_modeling.hf_batch.utils import derive_run_seed

    assert derive_run_seed(
        base_seed=7,
        algorithm="algo3",
        model="Qwen/Qwen3.5-9B",
        pair_name="subgraph_1_to_subgraph_3",
        condition_bits="000",
        decoding=DecodingConfig(algorithm="beam", num_beams=6),
        replication=0,
    ) == 403453940


def test_derive_run_seed_distinguishes_non_default_graph_sources() -> None:
    from llm_conceptual_modeling.hf_batch.utils import derive_run_seed

    base_kwargs = {
        "base_seed": 7,
        "algorithm": "algo3",
        "model": "Qwen/Qwen3.5-9B",
        "pair_name": "subgraph_1_to_subgraph_3",
        "condition_bits": "000",
        "decoding": DecodingConfig(algorithm="beam", num_beams=6),
        "replication": 0,
    }

    babs_seed = derive_run_seed(**base_kwargs, graph_source="babs_johnson")
    clarice_seed = derive_run_seed(**base_kwargs, graph_source="clarice_starling")

    assert babs_seed != clarice_seed


def test_plan_paper_batch_specs_iterates_configured_graph_sources_for_algo3(
    tmp_path: Path,
) -> None:
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
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    children_by_label: 256
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  beam:
    enabled: true
    num_beams: [6]
inputs:
  graph_sources: [babs_johnson, clarice_starling, philip_marlowe]
shared_fragments: {}
algorithms:
  algo3:
    pair_names:
      - subgraph_1_to_subgraph_3
      - subgraph_2_to_subgraph_1
      - subgraph_2_to_subgraph_3
    base_fragments: []
    fixed_runtime_fields:
      include_counterexample: false
    fixed_columns:
      Counter-Example: -1
    factors:
      example:
        column: Example
        levels: [-1, 1]
        runtime_field: include_example
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      number_of_words:
        column: Number of Words
        levels: [3, 5]
        runtime_field: child_count
        low_runtime_value: 3
        high_runtime_value: 5
        low_fragments: []
        high_fragments: []
      depth:
        column: Depth
        levels: [1, 2]
        runtime_field: max_depth
        low_runtime_value: 1
        high_runtime_value: 2
        low_fragments: []
        high_fragments: []
    fragment_definitions: {}
    prompt_templates:
      tree_expansion: "Generate children."
""",
        encoding="utf-8",
    )
    config = load_hf_run_config(config_path)

    specs = plan_paper_batch_specs(
        models=config.models.chat_models,
        embedding_model=config.models.embedding_model,
        replications=config.run.replications,
        algorithms=("algo3",),
        config=config,
        runtime_profile_provider=lambda _model: RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=None,
        ),
    )

    assert len(specs) == 72
    assert {spec.graph_source for spec in specs} == {
        "babs_johnson",
        "clarice_starling",
        "philip_marlowe",
    }
    assert all(len(spec.condition_bits) == 3 for spec in specs)
    assert all(spec.prompt_factors["include_counterexample"] is False for spec in specs)
    assert all(spec.raw_context["Counter-Example"] == -1 for spec in specs)
    assert {
        spec.graph_source: len(spec.input_payload["mother_graph"])
        for spec in specs
    } == {
        "babs_johnson": 113,
        "clarice_starling": 64,
        "philip_marlowe": 38,
    }


def test_qwen_contrastive_chat_client_is_constructible() -> None:
    from llm_conceptual_modeling.common.hf_transformers import HFTransformersChatClient

    client = HFTransformersChatClient(
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

    assert client is not None


def test_select_run_spec_module() -> None:
    assert select_run_spec.__module__ == "llm_conceptual_modeling.hf_batch.planning"


def test_plan_paper_batch_module() -> None:
    assert plan_paper_batch.__module__ == "llm_conceptual_modeling.hf_batch.planning"
