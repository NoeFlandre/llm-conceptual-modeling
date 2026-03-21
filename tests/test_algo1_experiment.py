from pathlib import Path

from llm_conceptual_modeling.algo1.experiment import (
    build_algo1_experiment_specs,
)


def test_build_algo1_experiment_specs_expands_full_factorial_grid(tmp_path: Path) -> None:
    actual = build_algo1_experiment_specs(
        pair_name="sg1_sg2",
        output_root=tmp_path,
    )

    assert len(actual) == 160
    first_spec = actual[0]
    last_spec = actual[-1]

    assert first_spec.run_name == "algo1_sg1_sg2_rep0_cond00000"
    assert first_spec.prompt_config.include_explanation is False
    assert first_spec.prompt_config.include_example is False
    assert first_spec.prompt_config.include_counterexample is False
    assert first_spec.output_dir == tmp_path / "algo1" / "sg1_sg2" / "rep0_cond00000"

    assert last_spec.run_name == "algo1_sg1_sg2_rep4_cond11111"
    assert last_spec.prompt_config.include_explanation is True
    assert last_spec.prompt_config.include_example is True
    assert last_spec.prompt_config.include_counterexample is True


def test_build_algo1_experiment_specs_maps_pair_to_expected_subgraphs(tmp_path: Path) -> None:
    actual = build_algo1_experiment_specs(
        pair_name="sg2_sg3",
        output_root=tmp_path,
    )

    first_spec = actual[0]
    first_left_edge = first_spec.subgraph1[0]
    first_right_edge = first_spec.subgraph2[0]

    assert first_left_edge == (
        "Culture of eating that promotes healthy choices",
        "Physical well-being",
    )
    assert first_right_edge == (
        "Medications",
        "Balance",
    )
