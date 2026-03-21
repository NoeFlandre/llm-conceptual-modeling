from pathlib import Path

from llm_conceptual_modeling.algo3.experiment import (
    build_algo3_experiment_specs,
)


def test_build_algo3_experiment_specs_expands_full_factorial_grid(tmp_path: Path) -> None:
    actual = build_algo3_experiment_specs(
        pair_name="subgraph_1_to_subgraph_3",
        output_root=tmp_path,
    )

    assert len(actual) == 80
    first_spec = actual[0]
    last_spec = actual[-1]

    assert first_spec.run_name == "algo3_subgraph_1_to_subgraph_3_rep0_cond0000"
    assert first_spec.prompt_config.include_example is False
    assert first_spec.prompt_config.include_counterexample is False
    assert first_spec.child_count == 3
    assert first_spec.max_depth == 1
    assert first_spec.output_dir == (
        tmp_path / "algo3" / "subgraph_1_to_subgraph_3" / "rep0_cond0000"
    )

    assert last_spec.run_name == "algo3_subgraph_1_to_subgraph_3_rep4_cond1111"
    assert last_spec.prompt_config.include_example is True
    assert last_spec.prompt_config.include_counterexample is True
    assert last_spec.child_count == 5
    assert last_spec.max_depth == 2


def test_build_algo3_experiment_specs_maps_pair_to_expected_node_sets(tmp_path: Path) -> None:
    actual = build_algo3_experiment_specs(
        pair_name="subgraph_2_to_subgraph_1",
        output_root=tmp_path,
    )

    first_spec = actual[0]

    assert first_spec.source_labels[0] == "Culture of eating that promotes healthy choices"
    assert first_spec.target_labels[0] == "Prevalence of walking trails"
