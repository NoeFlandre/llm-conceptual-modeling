import json
from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.algo1.mistral import Method1PromptConfig
from llm_conceptual_modeling.common.mistral import _format_knowledge_map
from llm_conceptual_modeling.hf_experiments import (
    _build_prompt_bundle,
    plan_paper_batch,
    run_paper_batch,
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
        "{formatted_subgraph2} Your task is to recommend 5 more nodes in relation to those "
        "already in the two knowledge maps. Do not suggest nodes that are already in the maps. "
        "Return the recommended nodes as a list of nodes in the format ['A', 'B', 'C', 'D', "
        "'E']. Your output must only be the list of proposed nodes. Do not repeat any "
        "instructions I have given you and do not add unnecessary words or phrases."
    )

    assert label_prompt == expected_prompt


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
