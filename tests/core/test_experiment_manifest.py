"""Tests for the experiment YAML manifest schema and serialization."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from llm_conceptual_modeling.algo1.mistral import Method1PromptConfig
from llm_conceptual_modeling.algo2.mistral import Method2PromptConfig
from llm_conceptual_modeling.algo3.mistral import Method3PromptConfig
from llm_conceptual_modeling.common.types import ExperimentManifest


class TestParseManifest:
    """Tests for the parse_manifest() function in experiment_manifest module.

    These tests define the contract for the YAML manifest schema:
    - experiment_id, algorithm (algo1|algo2|algo3), model, provider
    - temperature, top_p, max_tokens (sampling params)
    - prompt_factors (dict of factor names → bool/int levels)
    - full_prompt (the complete prompt text)
    - input_subgraph_pairs (list of {subgraph_name, edges} or {subgraph_name, labels})
    - output_dir, repetitions (int), timestamp, condition_bits, pair_name
    """

    def test_parse_manifest_loads_valid_yaml(self, tmp_path: Path) -> None:
        """parse_manifest() returns an ExperimentManifest when given a valid YAML file."""
        from llm_conceptual_modeling.experiment_manifest import parse_manifest

        manifest_yaml = tmp_path / "manifest.yaml"
        manifest_yaml.write_text(
            "experiment_id: algo1_sg1_sg2_rep0_cond00000\n"
            "algorithm: algo1\n"
            "model: mistral-small-2603\n"
            "provider: mistral\n"
            "temperature: 0.0\n"
            "top_p: null\n"
            "max_tokens: null\n"
            "prompt_factors:\n"
            "  use_adjacency_notation: false\n"
            "  use_array_representation: false\n"
            "  include_explanation: false\n"
            "  include_example: false\n"
            "  include_counterexample: false\n"
            "full_prompt: 'You are a helpful assistant.'\n"
            "input_subgraph_pairs:\n"
            "  - subgraph_name: subgraph1\n"
            "    edges: [['A', 'B'], ['B', 'C']]\n"
            "  - subgraph_name: subgraph2\n"
            "    edges: [['X', 'Y'], ['Y', 'Z']]\n"
            "output_dir: /data/results/algo1/sg1_sg2/rep0_cond00000\n"
            "timestamp: '2026-03-22T10:00:00+00:00'\n"
            "repetitions: 5\n"
            "condition_bits: '00000'\n"
            "pair_name: sg1_sg2\n"
        )
        manifest = parse_manifest(manifest_yaml)
        assert isinstance(manifest, ExperimentManifest)
        assert manifest.experiment_id == "algo1_sg1_sg2_rep0_cond00000"
        assert manifest.algorithm == "algo1"
        assert manifest.model == "mistral-small-2603"
        assert manifest.temperature == 0.0
        assert manifest.provider == "mistral"
        assert manifest.repetitions == 5
        assert manifest.pair_name == "sg1_sg2"
        assert manifest.condition_bits == "00000"

    def test_parse_manifest_rejects_missing_experiment_id(self, tmp_path: Path) -> None:
        """parse_manifest() raises if experiment_id is missing."""
        from llm_conceptual_modeling.experiment_manifest import parse_manifest

        manifest_yaml = tmp_path / "manifest.yaml"
        manifest_yaml.write_text(
            "algorithm: algo1\n"
            "model: mistral-small-2603\n"
            "provider: mistral\n"
            "temperature: 0.0\n"
            "prompt_factors: {}\n"
            "full_prompt: ''\n"
            "input_subgraph_pairs: []\n"
            "output_dir: /tmp/out\n"
            "timestamp: '2026-03-22T10:00:00+00:00'\n"
            "repetitions: 5\n"
            "condition_bits: '00000'\n"
            "pair_name: sg1_sg2\n"
        )
        with pytest.raises((KeyError, ValueError)):
            parse_manifest(manifest_yaml)

    def test_parse_manifest_rejects_invalid_algorithm(self, tmp_path: Path) -> None:
        """parse_manifest() raises if algorithm is not algo1|algo2|algo3."""
        from llm_conceptual_modeling.experiment_manifest import parse_manifest

        manifest_yaml = tmp_path / "manifest.yaml"
        manifest_yaml.write_text(
            "experiment_id: exp_test\n"
            "algorithm: algo99\n"
            "model: mistral-small-2603\n"
            "provider: mistral\n"
            "temperature: 0.0\n"
            "prompt_factors: {}\n"
            "full_prompt: ''\n"
            "input_subgraph_pairs: []\n"
            "output_dir: /tmp/out\n"
            "timestamp: '2026-03-22T10:00:00+00:00'\n"
            "repetitions: 5\n"
            "condition_bits: '00000'\n"
            "pair_name: sg1_sg2\n"
        )
        with pytest.raises(ValueError):
            parse_manifest(manifest_yaml)

    def test_parse_manifest_rejects_missing_input_subgraph_pairs(self, tmp_path: Path) -> None:
        """parse_manifest() raises if input_subgraph_pairs is missing."""
        from llm_conceptual_modeling.experiment_manifest import parse_manifest

        manifest_yaml = tmp_path / "manifest.yaml"
        manifest_yaml.write_text(
            "experiment_id: exp_test\n"
            "algorithm: algo1\n"
            "model: mistral-small-2603\n"
            "provider: mistral\n"
            "temperature: 0.0\n"
            "prompt_factors: {}\n"
            "full_prompt: ''\n"
            "output_dir: /tmp/out\n"
            "timestamp: '2026-03-22T10:00:00+00:00'\n"
            "repetitions: 5\n"
            "condition_bits: '00000'\n"
            "pair_name: sg1_sg2\n"
        )
        with pytest.raises((KeyError, ValueError)):
            parse_manifest(manifest_yaml)

    def test_parse_manifest_algo3_includes_labels(self, tmp_path: Path) -> None:
        """parse_manifest() correctly handles algo3 with labels-based subgraph format."""
        from llm_conceptual_modeling.experiment_manifest import parse_manifest

        manifest_yaml = tmp_path / "manifest.yaml"
        manifest_yaml.write_text(
            "experiment_id: algo3_subgraph_1_rep2_cond1010\n"
            "algorithm: algo3\n"
            "model: mistral-medium-2508\n"
            "provider: mistral\n"
            "temperature: 0.0\n"
            "top_p: null\n"
            "max_tokens: null\n"
            "prompt_factors:\n"
            "  include_example: true\n"
            "  include_counterexample: false\n"
            "  child_count: 5\n"
            "  max_depth: 2\n"
            "full_prompt: 'You are a helpful assistant.'\n"
            "input_subgraph_pairs:\n"
            "  - subgraph_name: source_labels\n"
            "    labels: ['A', 'B', 'C']\n"
            "  - subgraph_name: target_labels\n"
            "    labels: ['X', 'Y', 'Z']\n"
            "output_dir: /data/results/algo3/subgraph_1/rep2_cond1010\n"
            "timestamp: '2026-03-22T12:00:00+00:00'\n"
            "repetitions: 5\n"
            "condition_bits: '1010'\n"
            "pair_name: subgraph_1_to_subgraph_3\n"
        )
        manifest = parse_manifest(manifest_yaml)
        assert manifest.algorithm == "algo3"
        assert manifest.prompt_factors["include_example"] is True
        assert manifest.prompt_factors["child_count"] == 5
        assert len(manifest.input_subgraph_pairs) == 2
        assert manifest.input_subgraph_pairs[0]["subgraph_name"] == "source_labels"
        assert manifest.input_subgraph_pairs[0]["labels"] == ["A", "B", "C"]


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
            use_relaxed_convergence=True,
        )
        factors = ExperimentManifest.prompt_config_to_factors(config, algorithm="algo2")
        assert factors == {
            "use_adjacency_notation": False,
            "use_array_representation": True,
            "include_explanation": True,
            "include_example": True,
            "include_counterexample": True,
            "use_relaxed_convergence": True,
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
        from llm_conceptual_modeling.algo1.mistral import build_direct_edge_prompt

        class ProbeLikeSpec:
            def __init__(self) -> None:
                self.run_name = "algo1_sg1_sg2_rep0_cond00000"
                self.model = "mistral-small-2603"
                self.subgraph1 = [("A", "B"), ("B", "C")]
                self.subgraph2 = [("X", "Y"), ("Y", "Z")]
                self.prompt_config = Method1PromptConfig(
                    use_adjacency_notation=False,
                    use_array_representation=False,
                    include_explanation=False,
                    include_example=False,
                    include_counterexample=False,
                )
                self.output_dir = Path("/data/results/algo1/sg1_sg2/rep0_cond00000")

        spec = ProbeLikeSpec()
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
