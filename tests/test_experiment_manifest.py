"""Tests for the experiment YAML manifest schema and serialization."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from llm_conceptual_modeling.algo1.mistral import Method1PromptConfig
from llm_conceptual_modeling.algo2.mistral import Method2PromptConfig
from llm_conceptual_modeling.algo3.mistral import Method3PromptConfig
from llm_conceptual_modeling.common.types import ExperimentManifest


class TestExperimentManifestSchema:
    """Tests that ExperimentManifest captures all fields required for reproducibility."""

    def test_experiment_manifest_can_be_constructed_with_all_fields(self) -> None:
        """Smoke test: ExperimentManifest can be constructed with all required fields."""
        manifest = ExperimentManifest(
            experiment_id="exp_algo1_sg1_sg2_rep0_cond00000",
            algorithm="algo1",
            model="mistral-small-2603",
            provider="mistral",
            temperature=0.0,
            top_p=None,
            max_tokens=None,
            prompt_factors={
                "use_adjacency_notation": False,
                "use_array_representation": False,
                "include_explanation": False,
                "include_example": False,
                "include_counterexample": False,
            },
            full_prompt="You are a helpful assistant...",
            input_subgraph_pairs=[
                {"subgraph_name": "sg1", "edges": [["A", "B"], ["B", "C"]]},
                {"subgraph_name": "sg2", "edges": [["X", "Y"], ["Y", "Z"]]},
            ],
            output_dir="/data/results/algo1/sg1_sg2/rep0_cond00000",
            timestamp=datetime.now(UTC).isoformat(),
            repetitions=5,
            condition_bits="00000",
            pair_name="sg1_sg2",
        )
        assert manifest.experiment_id == "exp_algo1_sg1_sg2_rep0_cond00000"
        assert manifest.algorithm == "algo1"
        assert manifest.model == "mistral-small-2603"
        assert manifest.temperature == 0.0

    def test_experiment_manifest_serializes_to_yaml(self, tmp_path: Path) -> None:
        """ExperimentManifest.to_yaml() produces valid YAML with all fields."""
        manifest = ExperimentManifest(
            experiment_id="exp_algo2_sg2_sg3_rep1_cond10101",
            algorithm="algo2",
            model="mistral-medium-2508",
            provider="mistral",
            temperature=0.0,
            top_p=None,
            max_tokens=None,
            prompt_factors={
                "use_adjacency_notation": True,
                "use_array_representation": False,
                "include_explanation": True,
                "include_example": False,
                "include_counterexample": True,
            },
            full_prompt="You are a helpful assistant who can creatively suggest...",
            input_subgraph_pairs=[
                {"subgraph_name": "sg2", "edges": [["A", "B"]]},
                {"subgraph_name": "sg3", "edges": [["X", "Y"]]},
            ],
            output_dir="/data/results/algo2/sg2_sg3/rep1_cond10101",
            timestamp="2026-03-22T10:00:00+00:00",
            repetitions=5,
            condition_bits="10101",
            pair_name="sg2_sg3",
        )
        yaml_path = tmp_path / "manifest.yaml"
        manifest.to_yaml(yaml_path)

        # Verify the YAML file was written and can be parsed back
        with yaml_path.open() as f:
            parsed = yaml.safe_load(f)

        assert parsed["experiment_id"] == "exp_algo2_sg2_sg3_rep1_cond10101"
        assert parsed["algorithm"] == "algo2"
        assert parsed["model"] == "mistral-medium-2508"
        assert parsed["temperature"] == 0.0
        assert parsed["prompt_factors"]["use_adjacency_notation"] is True
        assert parsed["prompt_factors"]["include_explanation"] is True
        assert parsed["repetitions"] == 5

    def test_algo1_prompt_config_serializes_correctly(self) -> None:
        """Method1PromptConfig can be converted to prompt_factors dict."""
        config = Method1PromptConfig(
            use_adjacency_notation=True,
            use_array_representation=True,
            include_explanation=False,
            include_example=True,
            include_counterexample=False,
        )
        factors = ExperimentManifest.prompt_config_to_factors(config, algorithm="algo1")
        assert factors == {
            "use_adjacency_notation": True,
            "use_array_representation": True,
            "include_explanation": False,
            "include_example": True,
            "include_counterexample": False,
        }

    def test_algo2_prompt_config_serializes_correctly(self) -> None:
        """Method2PromptConfig can be converted to prompt_factors dict."""
        config = Method2PromptConfig(
            use_adjacency_notation=False,
            use_array_representation=True,
            include_explanation=True,
            include_example=True,
            include_counterexample=True,
        )
        factors = ExperimentManifest.prompt_config_to_factors(config, algorithm="algo2")
        assert factors == {
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": True,
            "include_counterexample": True,
        }

    def test_algo3_prompt_config_serializes_correctly(self) -> None:
        """Method3PromptConfig can be converted to prompt_factors dict."""
        config = Method3PromptConfig(
            include_example=True,
            include_counterexample=True,
        )
        factors = ExperimentManifest.prompt_config_to_factors(config, algorithm="algo3")
        assert factors == {
            "include_example": True,
            "include_counterexample": True,
        }

    def test_experiment_manifest_from_algo1_probe_spec(self) -> None:
        """Can construct ExperimentManifest from Algo1ProbeSpec + prompt text."""
        from llm_conceptual_modeling.algo1.probe import Algo1ProbeSpec
        from llm_conceptual_modeling.algo1.mistral import build_direct_edge_prompt

        spec = Algo1ProbeSpec(
            run_name="algo1_sg1_sg2_rep0_cond00000",
            model="mistral-small-2603",
            subgraph1=[("A", "B"), ("B", "C")],
            subgraph2=[("X", "Y"), ("Y", "Z")],
            prompt_config=Method1PromptConfig(
                use_adjacency_notation=False,
                use_array_representation=False,
                include_explanation=False,
                include_example=False,
                include_counterexample=False,
            ),
            output_dir=Path("/data/results/algo1/sg1_sg2/rep0_cond00000"),
            resume=False,
        )
        full_prompt = build_direct_edge_prompt(
            subgraph1=spec.subgraph1,
            subgraph2=spec.subgraph2,
            prompt_config=spec.prompt_config,
        )
        manifest = ExperimentManifest.from_probe_spec(
            spec=spec,
            algorithm="algo1",
            provider="mistral",
            temperature=0.0,
            top_p=None,
            max_tokens=None,
            full_prompt=full_prompt,
            pair_name="sg1_sg2",
            condition_bits="00000",
            repetitions=5,
        )
        assert manifest.experiment_id == "algo1_sg1_sg2_rep0_cond00000"
        assert manifest.algorithm == "algo1"
        assert manifest.model == "mistral-small-2603"
        assert manifest.pair_name == "sg1_sg2"
        assert manifest.condition_bits == "00000"
        assert manifest.repetitions == 5
        assert len(manifest.input_subgraph_pairs) == 2
        assert manifest.prompt_factors["include_example"] is False

    def test_experiment_manifest_roundtrip_via_yaml(self, tmp_path: Path) -> None:
        """A manifest written to YAML and read back is identical."""
        manifest = ExperimentManifest(
            experiment_id="exp_test",
            algorithm="algo3",
            model="mistral-medium-2508",
            provider="mistral",
            temperature=0.0,
            top_p=None,
            max_tokens=None,
            prompt_factors={"include_example": True, "include_counterexample": False},
            full_prompt="Test prompt",
            input_subgraph_pairs=[
                {"subgraph_name": "src", "edges": [["1", "2"]]},
                {"subgraph_name": "tgt", "edges": [["3", "4"]]},
            ],
            output_dir="/data/results/algo3/src_tgt/rep2_cond0100",
            timestamp="2026-03-22T12:00:00+00:00",
            repetitions=5,
            condition_bits="0100",
            pair_name="src_tgt",
        )
        path = tmp_path / "roundtrip.yaml"
        manifest.to_yaml(path)
        with path.open() as f:
            parsed = yaml.safe_load(f)
        # Reconstruct from dict
        reconstructed = ExperimentManifest.from_dict(parsed)
        assert reconstructed.experiment_id == manifest.experiment_id
        assert reconstructed.algorithm == manifest.algorithm
        assert reconstructed.model == manifest.model
        assert reconstructed.prompt_factors == manifest.prompt_factors
        assert reconstructed.repetitions == manifest.repetitions
