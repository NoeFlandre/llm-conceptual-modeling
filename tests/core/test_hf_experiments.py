import json
from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.algo1.mistral import Method1PromptConfig
from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.common.mistral import _format_knowledge_map
from llm_conceptual_modeling.hf_batch.monitoring import collect_batch_status
from llm_conceptual_modeling.hf_experiments import (
    HFRunSpec,
    _build_prompt_bundle,
    _run_algo1,
    _run_algo2,
    _run_algo3,
    plan_paper_batch,
    run_paper_batch,
    run_single_spec,
)
from llm_conceptual_modeling.hf_run_config import load_hf_run_config


def test_plan_paper_batch_covers_full_factorial_surface() -> None:
    specs = plan_paper_batch(
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
    )

    assert len(specs) == 1680


def test_run_paper_batch_writes_resumable_state_and_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )

    manifest_paths = list(output_root.rglob("manifest.json"))
    state_paths = list(output_root.rglob("state.json"))
    summary_paths = list(output_root.rglob("summary.json"))

    assert manifest_paths
    assert state_paths
    assert summary_paths

    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["provider"] == "hf-transformers"
    assert manifest["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
    assert manifest["decoding"]["algorithm"] in {"greedy", "beam", "contrastive"}
    assert manifest["runtime"]["device"] == "cuda"
    assert manifest["runtime"]["quantization"] == "none"


def test_run_paper_batch_resumes_without_recomputing_finished_runs(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    call_count = {"count": 0}

    def runtime_factory(spec):
        call_count["count"] += 1
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )
    first_count = call_count["count"]

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
        resume=True,
    )

    assert call_count["count"] == first_count


def test_run_paper_batch_resume_recomputes_partially_finished_run(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    call_count = {"count": 0}

    def runtime_factory(spec):
        call_count["count"] += 1
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )
    first_count = call_count["count"]

    runtime_paths = sorted(output_root.rglob("runtime.json"))
    assert runtime_paths
    runtime_paths[0].unlink()

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
        resume=True,
    )

    assert call_count["count"] == first_count + 1


def test_run_paper_batch_resume_does_not_skip_failed_state(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    call_count = {"count": 0}

    def runtime_factory(spec):
        call_count["count"] += 1
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )
    first_count = call_count["count"]

    state_paths = sorted(output_root.rglob("state.json"))
    assert state_paths
    state_paths[0].write_text('{"status": "failed"}', encoding="utf-8")

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
        resume=True,
    )

    assert call_count["count"] == first_count + 1


def test_run_paper_batch_writes_batch_summary_csv(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": spec.model.startswith("mistralai/")},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )

    summary = pd.read_csv(output_root / "batch_summary.csv")

    assert {"algorithm", "model", "decoding_algorithm", "replication", "status"}.issubset(
        summary.columns
    )


def test_run_paper_batch_summary_includes_result_metrics_for_connection_algorithms(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        row = {
            **spec.raw_context,
            "Result": "[('alpha', 'gamma')]",
            "graph": "[('alpha', 'gamma')]",
            "subgraph1": "[('alpha', 'beta')]",
            "subgraph2": "[('gamma', 'delta')]",
        }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=2,
        algorithms=("algo1",),
        runtime_factory=runtime_factory,
    )

    summary_paths = sorted(output_root.rglob("summary.json"))
    assert summary_paths
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))

    assert summary["candidate_edge_count"] == 1
    assert summary["verified_edge_count"] == 1
    assert isinstance(summary["accuracy"], float)
    assert isinstance(summary["precision"], float)
    assert isinstance(summary["recall"], float)


def test_run_paper_batch_writes_live_batch_status_file(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        status_path = output_root / "batch_status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["total_runs"] == 1680
        assert status["running_count"] == 1
        assert status["current_run"]["algorithm"] == spec.algorithm
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )

    status = json.loads((output_root / "batch_status.json").read_text(encoding="utf-8"))

    assert status["total_runs"] == 1680
    assert status["finished_count"] == 1680
    assert status["failed_count"] == 0
    assert status["pending_count"] == 0
    assert status["running_count"] == 0
    assert status["current_run"] is None
    assert status["last_completed_run"] is not None
    assert status["percent_complete"] == 100.0


def test_collect_batch_status_reconstructs_health_from_run_tree(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    finished_dir = (
        output_root / "runs" / "algo1" / "model" / "greedy" / "sg1_sg2" / "00000" / "rep_00"
    )
    failed_dir = (
        output_root / "runs" / "algo2" / "model" / "greedy" / "sg1_sg2" / "000000" / "rep_00"
    )
    running_dir = (
        output_root
        / "runs"
        / "algo3"
        / "model"
        / "greedy"
        / "subgraph_1_to_subgraph_3"
        / "0000"
        / "rep_00"
    )
    pending_dir = (
        output_root / "runs" / "algo1" / "model" / "greedy" / "sg2_sg3" / "00000" / "rep_00"
    )
    for directory in [finished_dir, failed_dir, running_dir, pending_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    (finished_dir / "state.json").write_text('{"status": "finished"}', encoding="utf-8")
    (finished_dir / "summary.json").write_text('{"status": "finished"}', encoding="utf-8")
    (failed_dir / "state.json").write_text('{"status": "failed"}', encoding="utf-8")
    (failed_dir / "error.json").write_text('{"message": "boom"}', encoding="utf-8")
    (running_dir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (output_root / "batch_status.json").write_text(
        json.dumps(
            {
                "total_runs": 4,
                "current_run": {"algorithm": "algo3", "pair_name": "subgraph_1_to_subgraph_3"},
                "last_completed_run": {"algorithm": "algo1", "pair_name": "sg1_sg2"},
                "started_at": "2026-03-31T00:00:00+00:00",
                "updated_at": "2026-03-31T00:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    status = collect_batch_status(output_root)

    assert status["total_runs"] == 4
    assert status["finished_count"] == 1
    assert status["failed_count"] == 1
    assert status["running_count"] == 1
    assert status["pending_count"] == 1
    assert status["failure_count"] == 1
    assert status["percent_complete"] == 25.0
    assert status["current_run"]["algorithm"] == "algo3"
    assert status["last_completed_run"]["algorithm"] == "algo1"


def test_run_paper_batch_writes_error_artifact_and_marks_failed_state(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="boom"):
        run_paper_batch(
            output_root=output_root,
            models=["mistralai/Ministral-3-8B-Instruct-2512"],
            embedding_model="Qwen/Qwen3-Embedding-8B",
            replications=1,
            runtime_factory=lambda _spec: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    error_paths = list(output_root.rglob("error.json"))
    state_paths = list(output_root.rglob("state.json"))

    assert error_paths
    assert any(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "failed"
        for path in state_paths
    )


def test_run_paper_batch_writes_aggregated_outputs_and_ci_reports(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=2,
        runtime_factory=runtime_factory,
    )

    assert list((output_root / "aggregated").rglob("raw.csv"))
    assert list((output_root / "aggregated").rglob("evaluated.csv"))
    assert list((output_root / "aggregated").rglob("factorial.csv"))
    assert list((output_root / "aggregated").rglob("replication_budget_strict.csv"))
    assert list((output_root / "aggregated").rglob("replication_budget_relaxed.csv"))


def test_run_paper_batch_uses_yaml_config_as_execution_source_of_truth(tmp_path: Path) -> None:
    config_path = tmp_path / "paper_batch.yaml"
    configured_output_root = tmp_path / "configured-runs"
    config_path.write_text(
        f"""
run:
  provider: hf-transformers
  output_root: {configured_output_root}
  replications: 1
runtime:
  seed: 11
  temperature: 1.0
  quantization: none
  device_policy: cuda-only
  thinking_mode_by_model:
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
  context_policy:
    prompt_truncation: forbid
    safety_margin_tokens: 32
  max_new_tokens_by_schema:
    edge_list: 256
    vote_list: 64
    label_list: 128
    children_by_label: 384
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
    pair_names: [sg1_sg2]
    base_fragments: [assistant_role]
    factors:
      explanation:
        column: Explanation
        levels: [-1, 1]
        runtime_field: include_explanation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: [explanation_text]
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
      array_repr:
        column: Array/List(1/-1)
        levels: [-1, 1]
        runtime_field: use_array_representation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      adjacency_repr:
        column: Tag/Adjacency(1/-1)
        levels: [-1, 1]
        runtime_field: use_adjacency_notation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
    fragment_definitions:
      explanation_text: "Explain the notation."
    prompt_templates:
      direct_edge: >-
        Knowledge map 1: {{formatted_subgraph1}} Knowledge map 2: {{formatted_subgraph2}}
      cove_verification: "Candidate pairs: {{candidate_edges}}"
""",
        encoding="utf-8",
    )
    config = load_hf_run_config(config_path)

    run_paper_batch(
        output_root=tmp_path / "ignored",
        models=["ignored/model"],
        embedding_model="ignored/embedding",
        replications=99,
        config=config,
        dry_run=True,
    )

    summary = pd.read_csv(configured_output_root / "batch_summary.csv")
    manifest_path = next(configured_output_root.rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["algorithm"].unique().tolist() == ["algo1"]
    assert manifest["temperature"] == 1.0
    assert manifest["base_seed"] == 11
    assert isinstance(manifest["seed"], int)


def test_checked_in_config_algo1_prompt_matches_paper_matrix_variant() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo1 = config.algorithms["algo1"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo1",
        algorithm_config=algo1,
        active_high_factors=["explanation", "example", "counterexample"],
        prompt_factors={
            "use_adjacency_notation": True,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": True,
            "include_counterexample": True,
        },
    )

    direct_prompt = prompt_bundle["direct_edge"]
    expected_prompt = (
        "You are a helpful assistant who understands Knowledge Maps. "
        "A knowledge map is a network consisting of nodes and edges. Nodes must have a clear "
        "meaning, such that we can interpret having 'more' or 'less' of a node. Edges represent "
        "the existence of a direct relation between two nodes. "
        "The knowledge map is encoded using a list of nodes and an associated adjacency matrix. "
        "The adjacency matrix is an n*n square matrix that represents whether each edge exists. "
        "In the matrix, each row and each column corresponds to a node. Rows and columns come in "
        "the same order as the list of nodes. A relation between node A and node B is represented "
        "as a 1 in the row corresponding to A and the column corresponding to B. "
        "Here is an example of a desired output for your task. In knowledge map 1, we have the "
        "list of nodes ['capacity to hire', 'bad employees', 'good reputation'] and the "
        "associated adjacency matrix [[0,1,0],[0,0,1],[1,0,0]]. In knowledge map 2, we have the "
        "list of nodes ['work motivation', 'productivity', 'financial growth'] and the associated "
        "adjacency matrix [[0,1,0],[0,0,1],[0,0,0]]. In this example, you could recommend 3 new "
        "links: 'quality of managers' with 'work motivation', 'productivity' with 'good "
        "reputation' and 'bad employees' with 'quality of managers'. These links implicitly "
        "create 1 new node: 'quality of managers'. Therefore, this is the expected output: "
        "[('quality of managers', 'work motivation'), ('productivity', 'good reputation'), "
        "('bad employees', 'quality of managers')]. "
        "Here is an example of a bad output that we do not want to see. In knowledge map 1, we "
        "have the list of nodes ['capacity to hire', 'bad employees', 'good reputation'] and the "
        "associated adjacency matrix [[0,1,0],[0,0,1],[1,0,0]]. In knowledge map 2, we have the "
        "list of nodes ['work motivation', 'productivity', 'financial growth'] and the associated "
        "adjacency matrix [[0,1,0],[0,0,1],[0,0,0]]. A bad output would be: [('moon', 'bad "
        "employees')]. The error is the recommended link between 'moon' and 'bad employees'. "
        "Adding the node 'moon' would be incorrect since it has no relationship with the other "
        "nodes. The proposed link does not represent a true causal relationship. "
        "You will get two inputs: Knowledge map 1: {formatted_subgraph1} Knowledge map 2: "
        "{formatted_subgraph2} Your task is to recommend more links between the two maps. These "
        "links can use new nodes. Do not suggest links that are already in the maps. Do not "
        "suggest links between nodes of the same map. Return the recommended links as a list of "
        "edges in the format [(A, Z), …, (X, D)]. Your output must only be the list of proposed "
        "edges. Do not repeat any instructions I have given you and do not add unnecessary words "
        "or phrases."
    )

    assert direct_prompt == expected_prompt


def test_checked_in_config_algo2_prompt_matches_paper_markup_variant() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo2 = config.algorithms["algo2"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo2",
        algorithm_config=algo2,
        active_high_factors=["explanation", "example", "counterexample"],
        prompt_factors={
            "use_adjacency_notation": True,
            "use_array_representation": False,
            "include_explanation": True,
            "include_example": True,
            "include_counterexample": True,
            "use_relaxed_convergence": False,
        },
    )

    label_prompt = prompt_bundle["label_expansion"]
    expected_prompt = (
        "You are a helpful assistant who understands Knowledge Maps. "
        "A knowledge map is a network consisting of nodes and edges. Nodes must have a clear "
        "meaning, such that we can interpret having 'more' or 'less' of a node. Edges represent "
        "the existence of a direct relation between two nodes. "
        "The knowledge map is encoded using a hierarchical markup language representation. The "
        "list of nodes is defined between the opening tag <NODES> and the matching closing tag "
        "</NODES>. For each node, we list all other nodes by ID and indicate whether there is a "
        "connection ('True') or not ('False'). "
        "Here is an example of a desired output for your task. In knowledge map 1, we have the "
        "following hierarchical markup language representation: <NODES><NODE ID= 'capacity to "
        "hire'><TARGET ID= 'capacity to hire' isConnected=False/><TARGET ID= 'bad employees' "
        "isConnected=True/><TARGET ID= 'good reputation' isConnected=False/></NODE><NODE ID= "
        "'bad employees'><TARGET ID= 'capacity to hire' isConnected=False/><TARGET ID= 'bad "
        "employees' isConnected=False/><TARGET ID= 'good reputation' "
        "isConnected=True/></NODE><NODE ID= 'good reputation'><TARGET ID= 'capacity to hire' "
        "isConnected=True/><TARGET ID= 'bad employees' isConnected=False/><TARGET ID= 'good "
        "reputation' isConnected=False/></NODE></NODES>. In knowledge map 2, we have the "
        "following hierarchical markup language representation: <NODES><NODE ID= 'work "
        "motivation'><TARGET ID= 'work motivation' isConnected=False/><TARGET ID= 'productivity' "
        "isConnected=True/><TARGET ID= 'financial growth' isConnected=False/></NODE><NODE ID= "
        "'productivity'><TARGET ID= 'work motivation' isConnected=False/><TARGET ID= "
        "'productivity' isConnected=False/><TARGET ID= 'financial growth' "
        "isConnected=True/></NODE><NODE ID= 'financial growth'><TARGET ID= 'work motivation' "
        "isConnected=False/><TARGET ID= 'productivity' isConnected=False/><TARGET ID= 'financial "
        "growth' isConnected=False/></NODE></NODES>. In this example, you could recommend these 5 "
        "new nodes: 'quality of managers',  'employee satisfaction', 'customer satisfaction', "
        "'market share', 'performance incentives . Therefore, this is the expected output: "
        "[ 'quality of managers', 'employee satisfaction', 'customer satisfaction', 'market "
        "share', 'performance incentives']. "
        "Here is an example of a bad output that we do not want to see. In knowledge map 1, we "
        "have the following hierarchical markup language representation: <NODES><NODE ID= "
        "'capacity to hire'><TARGET ID= 'capacity to hire' isConnected=False/><TARGET ID= 'bad "
        "employees' isConnected=True/><TARGET ID= 'good reputation' "
        "isConnected=False/></NODE><NODE ID= 'bad employees'><TARGET ID= 'capacity to hire' "
        "isConnected=False/><TARGET ID= 'bad employees' isConnected=False/><TARGET ID= 'good "
        "reputation' isConnected=True/></NODE><NODE ID= 'good reputation'><TARGET ID= 'capacity "
        "to hire' isConnected=True/><TARGET ID= 'bad employees' isConnected=False/><TARGET ID= "
        "'good reputation' isConnected=False/></NODE></NODES>. In knowledge map 2, we have the "
        "following hierarchical markup language representation: <NODES><NODE ID= 'work "
        "motivation'><TARGET ID= 'work motivation' isConnected=False/><TARGET ID= 'productivity' "
        "isConnected=True/><TARGET ID= 'financial growth' isConnected=False/></NODE><NODE ID= "
        "'productivity'><TARGET ID= 'work motivation' isConnected=False/><TARGET ID= "
        "'productivity' isConnected=False/><TARGET ID= 'financial growth' "
        "isConnected=True/></NODE><NODE ID= 'financial growth'><TARGET ID= 'work motivation' "
        "isConnected=False/><TARGET ID= 'productivity' isConnected=False/><TARGET ID= 'financial "
        "growth' isConnected=False/></NODE></NODES>. A bad output would be: ['moon', 'dog', "
        "'thermodynamics', 'swimming', 'red']. Adding the proposed nodes would be incorrect since "
        "they have no relationship with the nodes in the input. "
        "You will get two inputs: Knowledge map 1: {formatted_subgraph1} Knowledge map 2: "
        "{formatted_subgraph2} Current label context: {seed_labels}. Your task is to recommend 5 "
        "more nodes in relation to those already in the two knowledge maps. Do not suggest nodes "
        "that are already in the maps. Return the recommended nodes as a list of nodes in the "
        "format ['A', 'B', 'C', 'D', 'E']. Your output must only be the list of proposed nodes. "
        "Do not repeat any instructions I have given you and do not add unnecessary words or "
        "phrases."
    )

    assert label_prompt == expected_prompt


def test_checked_in_config_algo2_edge_suggestion_prompt_carries_same_prompt_factors() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo2 = config.algorithms["algo2"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo2",
        algorithm_config=algo2,
        active_high_factors=["explanation", "example", "counterexample"],
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": True,
            "include_counterexample": True,
            "use_relaxed_convergence": False,
        },
    )

    edge_prompt = prompt_bundle["edge_suggestion"]

    assert edge_prompt.startswith("You are a helpful assistant who understands Knowledge Maps.")
    assert "A knowledge map is a network consisting of nodes and edges." in edge_prompt
    assert "Here is an example of a desired output for your task." in edge_prompt
    assert "Here is an example of a bad output that we do not want to see." in edge_prompt
    assert "Available concepts: {expanded_label_context}." in edge_prompt


def test_checked_in_config_algo3_prompt_matches_paper_and_excludes_depth_text() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo3 = config.algorithms["algo3"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo3",
        algorithm_config=algo3,
        active_high_factors=["example", "counterexample", "number_of_words"],
        prompt_factors={
            "include_example": True,
            "include_counterexample": True,
            "child_count": 5,
            "max_depth": 2,
        },
    )

    tree_prompt = prompt_bundle["tree_expansion"]
    expected_prompt = (
        "You are a helpful assistant who can creatively suggest relevant ideas. "
        "Your input is a set of concept names. All concept names must have a clear meaning, such "
        "that we can interpret having 'more' or 'less' of a concept. "
        "Your input is the following list of concept names: {source_labels} "
        "Your task is to recommend 5 related concept names for each of the names in the input. Do "
        "not suggest names that are in the input. Your output must include the list of the 5 "
        "proposed names for each of the input names. Do not include any other text. Return your "
        "proposed names in a dictionary format { 'A' : ['B' , 'C', 'D'], 'E' : ['F' , 'G' , "
        "'H'], …, 'U' : ['V' , 'W' , 'X'] }. "
        "Here is an example of a desired output for your task. We have the list of concepts "
        "['capacity to hire', 'bad employees', 'good reputation']. In this example, you could "
        "recommend these 15 new concepts: 'employment potential', 'hiring capability', 'staffing "
        "ability', 'underperformers', 'inefficient staff', 'problematic workers', 'positive "
        "image', 'favorable standing', 'high regard'. Therefore, this is the expected output: "
        '{ "capacity to hire": ["employment potential", "hiring capability", "staffing ability", '
        '"recruitment capacity", "talent acquisition"], "bad employees": ["underperformers", '
        '"inefficient staff", "problematic workers", "low performers", "unproductive staff"], '
        '"good reputation": ["positive image", "favorable standing", "high regard", "excellent '
        'reputation", "commendable status"] }. '
        "Here is an example of a bad output that we do not want to see. We have the list of "
        "nodes ['capacity to hire', 'bad employees', 'good reputation']. A bad output would be: "
        '{ "capacity to hire": [\'moon\', \'dog\', \'thermodynamics\', \'country\', \'pillow\'], '
        '"bad employees": [\'swimming\', \'red\', \'happiness\', \'food\', \'shoe\'], '
        '"good reputation": [\'judo\', \'canada\', \'light\', \'phone\', \'electricity\'] }. '
        "Adding the proposed concepts would be incorrect since they have no relationship with the "
        "concepts in the input. "
        "Your output must only be the list of proposed concepts. Do not repeat any instructions I "
        "have given you and do not add unnecessary words or phrases."
    )

    assert tree_prompt == expected_prompt


def test_knowledge_map_formatting_matches_paper_representations() -> None:
    edges = [
        ("capacity to hire", "bad employees"),
        ("bad employees", "good reputation"),
        ("good reputation", "capacity to hire"),
    ]

    matrix_format = _format_knowledge_map(
        edges,
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=True,
            include_explanation=False,
            include_example=False,
            include_counterexample=False,
        ),
    )
    markup_format = _format_knowledge_map(
        edges,
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=False,
            include_explanation=False,
            include_example=False,
            include_counterexample=False,
        ),
    )
    edges_format = _format_knowledge_map(
        edges,
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=False,
            use_array_representation=True,
            include_explanation=False,
            include_example=False,
            include_counterexample=False,
        ),
    )
    rdf_format = _format_knowledge_map(
        edges,
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=False,
            use_array_representation=False,
            include_explanation=False,
            include_example=False,
            include_counterexample=False,
        ),
    )

    assert matrix_format == (
        "the list of nodes ['capacity to hire', 'bad employees', 'good reputation'] "
        "and the associated adjacency matrix [[0, 1, 0], [0, 0, 1], [1, 0, 0]]"
    )
    assert markup_format.startswith(
        "<NODES><NODE ID= 'capacity to hire'>"
        "<TARGET ID= 'capacity to hire' isConnected=False/>"
    )
    assert edges_format == (
        "the following list of edges: [('capacity to hire', 'bad employees'), "
        "('bad employees', 'good reputation'), ('good reputation', 'capacity to hire')]"
    )
    assert rdf_format == (
        "the following RDF representation: <S><H>capacity to hire<T>bad employees"
        "<H>bad employees<T>good reputation<H>good reputation<T>capacity to hire<E>"
    )


class _StubChatClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)

    def complete_json(self, *, prompt: str, schema_name: str, schema: dict[str, object]):
        _ = schema
        response = self._responses.pop(0)
        response["_prompt"] = prompt
        response["_schema_name"] = schema_name
        return response


class _StubEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        for text in texts:
            if text == "alpha":
                vectors[text] = [1.0, 0.0]
            elif text == "beta":
                vectors[text] = [0.0, 1.0]
            else:
                vectors[text] = [1.0, 0.0]
        return vectors


class _StubRuntimeFactory:
    def __init__(
        self,
        *,
        chat_responses: list[dict[str, object]],
    ) -> None:
        self._chat_client = _StubChatClient(chat_responses)

    def build_chat_client(self, **kwargs):
        _ = kwargs
        return self._chat_client

    def build_embedding_client(self, **kwargs):
        _ = kwargs
        return _StubEmbeddingClient()


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=False,
        context_limit=4096,
    )


def test_run_algo3_configured_prompt_accepts_literal_dictionary_braces() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo3 = config.algorithms["algo3"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo3",
        algorithm_config=algo3,
        active_high_factors=["example", "counterexample", "number_of_words"],
        prompt_factors={
            "include_example": True,
            "include_counterexample": True,
            "child_count": 5,
            "max_depth": 1,
        },
    )
    spec = HFRunSpec(
        algorithm="algo3",
        model="Qwen/Qwen3.5-9B",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="subgraph_1_to_subgraph_3",
        condition_bits="1110",
        condition_label="greedy",
        prompt_factors={
            "include_example": True,
            "include_counterexample": True,
            "child_count": 5,
            "max_depth": 1,
        },
        raw_context={"pair_name": "subgraph_1_to_subgraph_3", "Repetition": 0},
        input_payload={
            "source_graph": [("alpha", "beta")],
            "target_graph": [("gamma", "delta")],
            "mother_graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        prompt_bundle=prompt_bundle,
        max_new_tokens_by_schema=config.runtime.max_new_tokens_by_schema,
        context_policy=config.runtime.context_policy,
        seed=7,
    )
    runtime = _StubRuntimeFactory(
        chat_responses=[
            {"children_by_label": {"alpha": ["alpha1", "alpha2", "alpha3", "alpha4", "alpha5"]}},
            {"children_by_label": {"beta": ["beta1", "beta2", "beta3", "beta4", "beta5"]}},
        ]
    )

    result = _run_algo3(spec, hf_runtime=runtime)
    raw_response = json.loads(result["raw_response"])

    assert "dictionary format { 'A' : ['B' , 'C', 'D']" in raw_response[0]["prompt"]
    assert "{source_labels}" not in raw_response[0]["prompt"]


def test_run_algo2_configured_prompt_uses_evolving_label_context_across_iterations() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo2 = config.algorithms["algo2"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo2",
        algorithm_config=algo2,
        active_high_factors=[],
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
            "use_relaxed_convergence": False,
        },
    )
    spec = HFRunSpec(
        algorithm="algo2",
        model="Qwen/Qwen3.5-9B",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg1_sg2",
        condition_bits="000000",
        condition_label="greedy",
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
            "use_relaxed_convergence": False,
        },
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        prompt_bundle=prompt_bundle,
        max_new_tokens_by_schema=config.runtime.max_new_tokens_by_schema,
        context_policy=config.runtime.context_policy,
        seed=11,
    )
    runtime = _StubRuntimeFactory(
        chat_responses=[
            {"labels": ["theta", "iota", "kappa", "lambda", "mu"]},
            {"labels": ["theta", "iota", "kappa", "lambda", "mu"]},
            {"edges": [{"source": "alpha", "target": "theta"}]},
            {"votes": ["Y"]},
        ]
    )

    result = _run_algo2(spec, hf_runtime=runtime)
    raw_response = json.loads(result["raw_response"])

    assert "theta" in raw_response[1]["prompt"]
    assert raw_response[0]["prompt"] != raw_response[1]["prompt"]


def test_checked_in_config_algo2_prompt_bundle_exposes_cove_verification_template() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo2 = config.algorithms["algo2"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo2",
        algorithm_config=algo2,
        active_high_factors=[],
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
            "use_relaxed_convergence": False,
        },
    )

    assert sorted(prompt_bundle) == ["cove_verification", "edge_suggestion", "label_expansion"]


def test_run_algo2_configured_prompt_applies_cove_verification_to_final_edges() -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo2 = config.algorithms["algo2"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo2",
        algorithm_config=algo2,
        active_high_factors=["explanation", "example", "counterexample", "array_repr"],
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": True,
            "include_counterexample": True,
            "use_relaxed_convergence": False,
        },
    )
    spec = HFRunSpec(
        algorithm="algo2",
        model="Qwen/Qwen3.5-9B",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg1_sg2",
        condition_bits="111100",
        condition_label="greedy",
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": True,
            "include_counterexample": True,
            "use_relaxed_convergence": False,
        },
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        prompt_bundle=prompt_bundle,
        max_new_tokens_by_schema=config.runtime.max_new_tokens_by_schema,
        context_policy=config.runtime.context_policy,
        seed=19,
    )
    runtime = _StubRuntimeFactory(
        chat_responses=[
            {"labels": ["theta", "iota", "kappa", "lambda", "mu"]},
            {"labels": ["theta", "iota", "kappa", "lambda", "mu"]},
            {"edges": [{"source": "alpha", "target": "theta"}]},
            {"votes": ["N"]},
        ]
    )

    result = _run_algo2(spec, hf_runtime=runtime)
    raw_response = json.loads(result["raw_response"])

    assert raw_response[-1]["schema_name"] == "vote_list"
    assert raw_response[-1]["prompt"] == (
        "Return whether a causal relationship exists between the source and target concepts for "
        "each pair in a list. For example, given [('smoking', 'cancer'), ('ice cream sales', "
        "'shark attacks')], return ['Y', 'N'] with no other text. Candidate pairs: "
        "[('alpha', 'theta')]"
    )
    assert result["raw_row"]["Result"] == "[]"


def test_run_algo2_writes_intermediate_stage_artifacts(tmp_path: Path) -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    spec = HFRunSpec(
        algorithm="algo2",
        model="Qwen/Qwen3.5-9B",
        embedding_model=config.models.embedding_model,
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg1_sg2",
        condition_bits="000000",
        condition_label="greedy",
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
            "use_relaxed_convergence": False,
        },
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        prompt_bundle=None,
        max_new_tokens_by_schema=config.runtime.max_new_tokens_by_schema,
        context_policy=config.runtime.context_policy,
        seed=23,
    )
    runtime = _StubRuntimeFactory(
        chat_responses=[
            {"labels": ["theta", "iota", "kappa", "lambda", "mu"]},
            {"labels": ["theta", "iota", "kappa", "lambda", "mu"]},
            {"edges": [{"source": "alpha", "target": "theta"}]},
            {"votes": ["Y"]},
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _run_algo2(spec, hf_runtime=runtime, run_dir=run_dir)

    label_stage = json.loads((run_dir / "stages" / "algo2_label_expansion.json").read_text())
    edge_stage = json.loads((run_dir / "stages" / "algo2_edge_generation.json").read_text())
    raw_trace = json.loads((run_dir / "raw_response.json").read_text())

    assert label_stage["expanded_labels"] == ["theta", "iota", "kappa", "lambda", "mu"]
    assert label_stage["iteration_count"] == 2
    assert edge_stage["verified_edges"] == [["alpha", "theta"]]
    assert raw_trace[-1]["schema_name"] == "vote_list"


def test_run_algo1_writes_and_reuses_edge_generation_stage(tmp_path: Path) -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo1 = config.algorithms["algo1"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo1",
        algorithm_config=algo1,
        active_high_factors=[],
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": False,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
        },
    )
    spec = HFRunSpec(
        algorithm="algo1",
        model="Qwen/Qwen3.5-9B",
        embedding_model=config.models.embedding_model,
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg1_sg2",
        condition_bits="00000",
        condition_label="greedy",
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": False,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
        },
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma"), ("theta", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        prompt_bundle=prompt_bundle,
        max_new_tokens_by_schema=config.runtime.max_new_tokens_by_schema,
        context_policy=config.runtime.context_policy,
        seed=29,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    first_runtime = _StubRuntimeFactory(
        chat_responses=[
            {"edges": [{"source": "alpha", "target": "theta"}]},
            {"votes": ["Y"]},
        ]
    )
    first_result = _run_algo1(spec, hf_runtime=first_runtime, run_dir=run_dir)

    stage = json.loads((run_dir / "stages" / "algo1_edge_generation.json").read_text())
    assert stage["candidate_edges"] == [["alpha", "theta"]]
    assert first_result["raw_row"]["Result"] == "[('alpha', 'theta')]"

    second_runtime = _StubRuntimeFactory(
        chat_responses=[
            {"votes": ["Y"]},
        ]
    )
    second_result = _run_algo1(spec, hf_runtime=second_runtime, run_dir=run_dir)
    raw_trace = json.loads(second_result["raw_response"])

    assert [record["schema_name"] for record in raw_trace] == ["vote_list"]
    assert second_result["raw_row"]["Result"] == "[('alpha', 'theta')]"


def test_run_algo1_writes_active_stage_tracking(tmp_path: Path) -> None:
    config = load_hf_run_config(
        "/Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/"
        "configs/hf_transformers_paper_batch.yaml"
    )
    algo1 = config.algorithms["algo1"]
    prompt_bundle = _build_prompt_bundle(
        algorithm_name="algo1",
        algorithm_config=algo1,
        active_high_factors=[],
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": False,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
        },
    )
    spec = HFRunSpec(
        algorithm="algo1",
        model="Qwen/Qwen3.5-9B",
        embedding_model=config.models.embedding_model,
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg2_sg3",
        condition_bits="00000",
        condition_label="greedy",
        prompt_factors={
            "use_adjacency_notation": False,
            "use_array_representation": False,
            "include_explanation": False,
            "include_example": False,
            "include_counterexample": False,
        },
        raw_context={"pair_name": "sg2_sg3", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma"), ("theta", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        prompt_bundle=prompt_bundle,
        max_new_tokens_by_schema=config.runtime.max_new_tokens_by_schema,
        context_policy=config.runtime.context_policy,
        seed=29,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = _StubRuntimeFactory(
        chat_responses=[
            {"edges": [{"source": "alpha", "target": "theta"}]},
            {"votes": ["Y"]},
        ]
    )

    _run_algo1(spec, hf_runtime=runtime, run_dir=run_dir)

    active_stage = json.loads((run_dir / "active_stage.json").read_text())
    assert active_stage["status"] == "completed"
    assert active_stage["schema_name"] == "vote_list"
    assert active_stage["algorithm"] == "algo1"
    assert active_stage["pair_name"] == "sg2_sg3"


def test_run_single_spec_writes_smoke_artifacts(tmp_path: Path) -> None:
    spec = HFRunSpec(
        algorithm="algo1",
        model="Qwen/Qwen3.5-9B",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg2_sg3",
        condition_bits="00000",
        condition_label="greedy",
        prompt_factors={},
        raw_context={"pair_name": "sg2_sg3", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
    )

    def runtime_factory(spec, *, run_dir=None):
        _ = (spec, run_dir)
        return {
            "raw_row": {
                "pair_name": "sg2_sg3",
                "Repetition": 0,
                "Result": "[('alpha', 'gamma')]",
                "graph": "[('alpha', 'gamma')]",
                "subgraph1": "[('alpha', 'beta')]",
                "subgraph2": "[('gamma', 'delta')]",
            },
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    summary = run_single_spec(
        spec=spec,
        output_root=tmp_path / "smoke",
        runtime_factory=runtime_factory,
        dry_run=False,
        resume=False,
    )

    run_dir = (
        tmp_path
        / "smoke"
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg2_sg3"
        / "00000"
        / "rep_00"
    )
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "runtime.json").exists()
    assert (run_dir / "raw_row.json").exists()
    assert (run_dir / "summary.json").exists()
    assert summary["pair_name"] == "sg2_sg3"


def test_run_paper_batch_writes_combined_factorial_with_decoding_and_error_rows(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            base_value = 0.0 if spec.decoding.algorithm == "greedy" else 1.0
            row = {
                **spec.raw_context,
                "Result": "[('a', 'b')]",
                "graph": "[('a', 'b')]",
                "subgraph1": "[('a', 'b')]",
                "subgraph2": "[('c', 'd')]",
            }
            if spec.algorithm == "algo1":
                row["Result"] = "[('a', 'c')]" if base_value else "[]"
            if spec.algorithm == "algo2":
                row["Result"] = "[('a', 'c')]" if base_value else "[]"
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=2,
        algorithms=("algo1",),
        runtime_factory=runtime_factory,
    )

    factorial_path = (
        output_root
        / "aggregated"
        / "algo1"
        / "mistralai__Ministral-3-8B-Instruct-2512"
        / "combined"
        / "factorial.csv"
    )
    actual = pd.read_csv(factorial_path)

    assert "Decoding Algorithm" in actual["Feature"].tolist()
    assert "Error" in actual["Feature"].tolist()
