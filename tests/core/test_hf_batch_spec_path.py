"""Tests for hf_batch.spec_path: pure spec-identity and path-resolution helpers."""

import json
from dataclasses import replace
from pathlib import Path

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_batch.types import HFRunSpec


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=False,
        context_limit=4096,
    )


def _greedy_spec() -> HFRunSpec:
    return HFRunSpec(
        algorithm="algo1",
        model="allenai/Olmo-3-7B-Instruct",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg1_sg2",
        condition_bits="00000",
        condition_label="greedy",
        prompt_factors={},
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
    )


def _beam_spec() -> HFRunSpec:
    return HFRunSpec(
        algorithm="algo2",
        model="Qwen/Qwen3.5-9B",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="beam", num_beams=2),
        replication=1,
        pair_name="sg2_sg3",
        condition_bits="000000",
        condition_label="beam_num_beams_2",
        prompt_factors={},
        raw_context={"pair_name": "sg2_sg3", "Repetition": 1},
        input_payload={"subgraph1": [], "subgraph2": [], "graph": []},
        runtime_profile=_runtime_profile(),
    )


# ---------------------------------------------------------------------------
# spec_identity
# ---------------------------------------------------------------------------

def test_spec_identity_returns_deterministic_tuple() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import spec_identity

    spec = _greedy_spec()
    identity = spec_identity(spec)

    assert identity == (
        "algo1",
        "allenai/Olmo-3-7B-Instruct",
        "greedy",
        "sg1_sg2",
        "00000",
        0,
    )


def test_spec_identity_distinguishes_different_specs() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import spec_identity

    assert spec_identity(_greedy_spec()) != spec_identity(_beam_spec())


def test_spec_identity_includes_non_default_graph_source() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import spec_identity

    spec = replace(_greedy_spec(), graph_source="babs_johnson")

    assert spec_identity(spec) == (
        "algo1",
        "allenai/Olmo-3-7B-Instruct",
        "greedy",
        "babs_johnson",
        "sg1_sg2",
        "00000",
        0,
    )


# ---------------------------------------------------------------------------
# smoke_spec_identity
# ---------------------------------------------------------------------------

def test_smoke_spec_identity_returns_dict_with_expected_keys() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import smoke_spec_identity

    spec = _greedy_spec()
    identity = smoke_spec_identity(spec)

    assert identity["algorithm"] == "algo1"
    assert identity["model"] == "allenai/Olmo-3-7B-Instruct"
    assert identity["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert identity["decoding_algorithm"] == "greedy"
    assert identity["pair_name"] == "sg1_sg2"
    assert identity["condition_bits"] == "00000"
    assert identity["replication"] == 0


def test_smoke_spec_identity_includes_graph_source() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import smoke_spec_identity

    spec = replace(_greedy_spec(), graph_source="babs_johnson")

    assert smoke_spec_identity(spec)["graph_source"] == "babs_johnson"


# ---------------------------------------------------------------------------
# run_dir_for_spec
# ---------------------------------------------------------------------------

def test_run_dir_for_spec_constructs_expected_path() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import run_dir_for_spec

    output_root = Path("/tmp/test-output")
    spec = _greedy_spec()
    run_dir = run_dir_for_spec(output_root=output_root, spec=spec)

    assert run_dir == (
        output_root
        / "runs"
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )


def test_run_dir_for_spec_handles_higher_replication() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import run_dir_for_spec

    output_root = Path("/tmp/test-output")
    spec = _beam_spec()
    run_dir = run_dir_for_spec(output_root=output_root, spec=spec)

    assert run_dir == (
        output_root
        / "runs"
        / "algo2"
        / "Qwen__Qwen3.5-9B"
        / "beam_num_beams_2"
        / "sg2_sg3"
        / "000000"
        / "rep_01"
    )


def test_run_dir_for_spec_includes_non_default_graph_source() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import run_dir_for_spec

    output_root = Path("/tmp/test-output")
    spec = replace(_greedy_spec(), graph_source="babs_johnson")

    assert run_dir_for_spec(output_root=output_root, spec=spec) == (
        output_root
        / "runs"
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "greedy"
        / "babs_johnson"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )


# ---------------------------------------------------------------------------
# run_dir_identity
# ---------------------------------------------------------------------------

def test_run_dir_identity_returns_model_slug_and_manifest_identity() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import run_dir_identity

    output_root = Path("/tmp/test-output")
    run_dir = (
        output_root
        / "runs"
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )

    identity = run_dir_identity(runs_root=output_root / "runs", run_dir=run_dir)

    assert identity == (
        "allenai__Olmo-3-7B-Instruct",
        (
            "algo1",
            "allenai/Olmo-3-7B-Instruct",
            "greedy",
            "sg1_sg2",
            "00000",
            0,
        ),
    )


def test_run_dir_identity_parses_non_default_graph_source_path() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import run_dir_identity

    output_root = Path("/tmp/test-output")
    run_dir = (
        output_root
        / "runs"
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "greedy"
        / "babs_johnson"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )

    assert run_dir_identity(runs_root=output_root / "runs", run_dir=run_dir) == (
        "allenai__Olmo-3-7B-Instruct",
        (
            "algo1",
            "allenai/Olmo-3-7B-Instruct",
            "greedy",
            "babs_johnson",
            "sg1_sg2",
            "00000",
            0,
        ),
    )


def test_run_dir_identity_returns_none_for_malformed_path() -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import run_dir_identity

    output_root = Path("/tmp/test-output")
    run_dir = (
        output_root
        / "runs"
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "greedy"
        / "sg1_sg2"
    )

    assert run_dir_identity(runs_root=output_root / "runs", run_dir=run_dir) is None


# ---------------------------------------------------------------------------
# filter_planned_specs_for_output_root
# ---------------------------------------------------------------------------

def test_filter_planned_specs_returns_all_when_no_manifest(tmp_path: Path) -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import filter_planned_specs_for_output_root

    specs = [_greedy_spec(), _beam_spec()]
    output_root = tmp_path / "no-manifest"

    result = filter_planned_specs_for_output_root(
        planned_specs=specs,
        output_root=output_root,
    )

    assert result == specs


def test_filter_planned_specs_filters_by_shard_manifest(tmp_path: Path) -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import filter_planned_specs_for_output_root

    spec_a = _greedy_spec()
    spec_b = _beam_spec()

    output_root = tmp_path / "sharded"
    output_root.mkdir()
    (output_root / "shard_manifest.json").write_text(
        json.dumps(
            {
                "shard_count": 2,
                "shard_index": 0,
                "identities": [
                    {
                        "algorithm": spec_a.algorithm,
                        "model": spec_a.model,
                        "condition_label": spec_a.condition_label,
                        "pair_name": spec_a.pair_name,
                        "condition_bits": spec_a.condition_bits,
                        "replication": spec_a.replication,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = filter_planned_specs_for_output_root(
        planned_specs=[spec_a, spec_b],
        output_root=output_root,
    )

    assert result == [spec_a]


def test_filter_planned_specs_filters_by_graph_source_manifest(tmp_path: Path) -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import filter_planned_specs_for_output_root

    spec_a = replace(_greedy_spec(), graph_source="babs_johnson")
    spec_b = replace(_greedy_spec(), graph_source="clarice_starling")

    output_root = tmp_path / "sharded"
    output_root.mkdir()
    (output_root / "shard_manifest.json").write_text(
        json.dumps(
            {
                "shard_count": 2,
                "shard_index": 0,
                "identities": [
                    {
                        "algorithm": spec_a.algorithm,
                        "model": spec_a.model,
                        "condition_label": spec_a.condition_label,
                        "graph_source": spec_a.graph_source,
                        "pair_name": spec_a.pair_name,
                        "condition_bits": spec_a.condition_bits,
                        "replication": spec_a.replication,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = filter_planned_specs_for_output_root(
        planned_specs=[spec_a, spec_b],
        output_root=output_root,
    )

    assert result == [spec_a]


def test_filter_planned_specs_preserves_order(tmp_path: Path) -> None:
    from llm_conceptual_modeling.hf_batch.spec_path import filter_planned_specs_for_output_root

    spec_a = _greedy_spec()
    spec_b = _beam_spec()

    output_root = tmp_path / "ordered"
    output_root.mkdir()
    (output_root / "shard_manifest.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "shard_index": 0,
                "identities": [
                    {
                        "algorithm": spec_b.algorithm,
                        "model": spec_b.model,
                        "condition_label": spec_b.condition_label,
                        "pair_name": spec_b.pair_name,
                        "condition_bits": spec_b.condition_bits,
                        "replication": spec_b.replication,
                    },
                    {
                        "algorithm": spec_a.algorithm,
                        "model": spec_a.model,
                        "condition_label": spec_a.condition_label,
                        "pair_name": spec_a.pair_name,
                        "condition_bits": spec_a.condition_bits,
                        "replication": spec_a.replication,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = filter_planned_specs_for_output_root(
        planned_specs=[spec_a, spec_b],
        output_root=output_root,
    )

    # Order must match input order, not manifest order
    assert result == [spec_a, spec_b]
