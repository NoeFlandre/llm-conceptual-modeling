from pathlib import Path

from llm_conceptual_modeling.algo1.experiment import (
    build_algo1_experiment_specs,
)
from llm_conceptual_modeling.experiment_manifest import parse_manifest


def test_build_algo1_experiment_specs_expands_full_factorial_grid(tmp_path: Path) -> None:
    actual = build_algo1_experiment_specs(
        pair_name="sg1_sg2",
        model="gpt-5",
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
    first_manifest = parse_manifest(first_spec.output_dir / "manifest.yaml")
    assert first_manifest.algorithm == "algo1"
    assert first_manifest.model == "gpt-5"
    assert first_manifest.provider == "mistral"
    assert first_manifest.temperature == 0.0
    assert first_manifest.pair_name == "sg1_sg2"
    assert first_manifest.condition_bits == "00000"
    assert first_manifest.repetitions == 5
    assert first_manifest.full_prompt.startswith(
        "You are a helpful assistant who understands Knowledge Maps."
    )

    assert last_spec.run_name == "algo1_sg1_sg2_rep4_cond11111"
    assert last_spec.prompt_config.include_explanation is True
    assert last_spec.prompt_config.include_example is True
    assert last_spec.prompt_config.include_counterexample is True


def test_build_algo1_experiment_specs_maps_pair_to_expected_subgraphs(tmp_path: Path) -> None:
    actual = build_algo1_experiment_specs(
        pair_name="sg2_sg3",
        model="gpt-5",
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
