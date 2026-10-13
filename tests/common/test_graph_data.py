import json
from pathlib import Path

import pytest

from llm_conceptual_modeling.algo1.generation import build_generation_manifest as algo1_manifest
from llm_conceptual_modeling.algo2.generation import build_generation_manifest as algo2_manifest
from llm_conceptual_modeling.algo3.generation import build_generation_manifest as algo3_manifest
from llm_conceptual_modeling.common.graph_data import (
    load_algo2_thesaurus,
    load_default_graph,
    load_wordnet_label_lexicon,
)
from tests.common.graph_data_fixtures import write_synthetic_default_graph


@pytest.fixture
def synthetic_inputs_root(monkeypatch, tmp_path, request: pytest.FixtureRequest) -> Path:
    inputs_root = tmp_path / "inputs"
    write_synthetic_default_graph(inputs_root)
    (inputs_root / "algo2_thesaurus.json").write_text(
        json.dumps({"synonyms": {"Obesity": ["Adiposity"]}, "antonyms": {"Obesity": ["Fitness"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LCM_INPUTS_ROOT", str(inputs_root))
    load_wordnet_label_lexicon.cache_clear()
    request.addfinalizer(load_wordnet_label_lexicon.cache_clear)
    return inputs_root


def test_default_graph_partitions_are_non_empty(synthetic_inputs_root) -> None:
    sg1, sg2, sg3, mother = load_default_graph()

    assert sg1
    assert sg2
    assert sg3
    assert mother


def test_default_graph_subgraphs_are_edge_disjoint(synthetic_inputs_root) -> None:
    sg1, sg2, sg3, _ = load_default_graph()

    assert set(sg1).isdisjoint(sg2)
    assert set(sg1).isdisjoint(sg3)
    assert set(sg2).isdisjoint(sg3)


def test_open_weight_map_extension_sources_are_discoverable() -> None:
    from llm_conceptual_modeling.common.graph_data import available_graph_sources, load_graph_source

    expected_counts = {
        "babs_johnson": {
            "node_count": 90,
            "edge_count": 113,
            "subgraph_node_counts": (33, 32, 25),
        },
        "clarice_starling": {
            "node_count": 53,
            "edge_count": 64,
            "subgraph_node_counts": (11, 22, 20),
        },
        "philip_marlowe": {
            "node_count": 29,
            "edge_count": 38,
            "subgraph_node_counts": (8, 7, 11),
        },
    }

    assert "babs_johnson" in available_graph_sources()

    for source_id, counts in expected_counts.items():
        sg1, sg2, sg3, mother = load_graph_source(source_id)
        nodes = {node for edge in mother for node in edge}

        assert len(nodes) == counts["node_count"]
        assert len(mother) == counts["edge_count"]
        assert (
            tuple(len({node for edge in subgraph for node in edge}) for subgraph in (sg1, sg2, sg3))
            == counts["subgraph_node_counts"]
        )
        assert sg1
        assert sg2
        assert sg3


def test_load_graph_source_default_matches_legacy_loader(synthetic_inputs_root) -> None:
    from llm_conceptual_modeling.common.graph_data import load_graph_source

    assert load_graph_source("default") == load_default_graph()


def test_available_graph_sources_honors_lcm_inputs_root(monkeypatch, tmp_path) -> None:
    from llm_conceptual_modeling.common.graph_data import available_graph_sources, load_graph_source

    inputs_root = tmp_path / "inputs"
    write_synthetic_default_graph(inputs_root)
    extension_root = inputs_root / "open_weight_map_extension"
    extension_root.mkdir()
    (extension_root / "fixture_categories.csv").write_text(
        "A,alpha\nB,alpha\nC,beta\nD,beta\nE,gamma\nF,gamma\n", encoding="utf-8"
    )
    (extension_root / "fixture_edges.csv").write_text("A,B,1\nC,D,1\nE,F,1\n", encoding="utf-8")
    (extension_root / "manifest.yaml").write_text(
        """graph_sources:
  fixture_map:
    display_name: Fixture map
    categories_path: open_weight_map_extension/fixture_categories.csv
    edges_path: open_weight_map_extension/fixture_edges.csv
    cluster_labels: [alpha, beta, gamma]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LCM_INPUTS_ROOT", str(inputs_root))

    assert available_graph_sources() == ("default", "fixture_map")
    sg1, sg2, sg3, _ = load_graph_source("fixture_map")
    assert sg1 == [("A", "B")]
    assert sg2 == [("C", "D")]
    assert sg3 == [("E", "F")]

    alternate_root = tmp_path / "alternate_inputs"
    write_synthetic_default_graph(alternate_root)
    monkeypatch.setenv("LCM_INPUTS_ROOT", str(alternate_root))

    assert available_graph_sources() == ("default",)


def test_load_graph_source_rejects_unknown_source() -> None:
    import pytest

    from llm_conceptual_modeling.common.graph_data import load_graph_source

    with pytest.raises(ValueError, match="missing_map"):
        load_graph_source("missing_map")


def test_generation_manifests_preserve_legacy_condition_counts(synthetic_inputs_root) -> None:
    assert algo1_manifest(fixture_only=False)["condition_count"] == 32
    assert algo2_manifest(fixture_only=False)["condition_count"] == 64
    assert algo3_manifest(fixture_only=False)["condition_count"] == 16


def test_algo2_thesaurus_is_available_and_non_empty() -> None:
    thesaurus = load_algo2_thesaurus()
    synonyms = thesaurus["synonyms"]
    antonyms = thesaurus["antonyms"]

    assert synonyms
    assert antonyms
    assert "Obesity" in synonyms
    assert "Obesity" in antonyms


def test_graph_data_loaders_honor_lcm_inputs_root(monkeypatch, tmp_path) -> None:
    inputs_root = tmp_path / "inputs"
    inputs_root.mkdir()
    (inputs_root / "algo2_thesaurus.json").write_text(
        json.dumps({"synonyms": {"Obesity": ["Adiposity"]}, "antonyms": {"Obesity": ["Fitness"]}})
    )
    (inputs_root / "wordnet_label_lexicon.json").write_text(
        json.dumps({"Obesity": ["obesity.n.01", "corpulence.n.01"]})
    )
    write_synthetic_default_graph(inputs_root)

    monkeypatch.setenv("LCM_INPUTS_ROOT", str(inputs_root))
    load_wordnet_label_lexicon.cache_clear()

    thesaurus = load_algo2_thesaurus()
    lexicon = load_wordnet_label_lexicon()
    sg1, sg2, sg3, mother = load_default_graph()

    assert thesaurus["synonyms"]["Obesity"] == ["Adiposity"]
    assert lexicon["Obesity"] == ["obesity.n.01", "corpulence.n.01"]
    assert sg1 == [("A", "B")]
    assert sg2 == [("C", "D")]
    assert sg3 == [("E", "F")]
    assert mother == [("A", "B"), ("C", "D"), ("E", "F"), ("A", "C")]
