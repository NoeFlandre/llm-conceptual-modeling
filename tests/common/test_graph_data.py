import json

from llm_conceptual_modeling.algo1.generation import build_generation_manifest as algo1_manifest
from llm_conceptual_modeling.algo2.generation import build_generation_manifest as algo2_manifest
from llm_conceptual_modeling.algo3.generation import build_generation_manifest as algo3_manifest
from llm_conceptual_modeling.common.graph_data import (
    load_algo2_thesaurus,
    load_default_graph,
    load_wordnet_label_lexicon,
)


def test_default_graph_partitions_are_non_empty() -> None:
    sg1, sg2, sg3, mother = load_default_graph()

    assert sg1
    assert sg2
    assert sg3
    assert mother


def test_default_graph_subgraphs_are_edge_disjoint() -> None:
    sg1, sg2, sg3, _ = load_default_graph()

    assert set(sg1).isdisjoint(sg2)
    assert set(sg1).isdisjoint(sg3)
    assert set(sg2).isdisjoint(sg3)


def test_open_weight_map_extension_sources_are_discoverable() -> None:
    from llm_conceptual_modeling.common.graph_data import load_graph_source

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

    for source_id, counts in expected_counts.items():
        sg1, sg2, sg3, mother = load_graph_source(source_id)
        nodes = {node for edge in mother for node in edge}

        assert len(nodes) == counts["node_count"]
        assert len(mother) == counts["edge_count"]
        assert tuple(
            len({node for edge in subgraph for node in edge})
            for subgraph in (sg1, sg2, sg3)
        ) == counts["subgraph_node_counts"]
        assert sg1
        assert sg2
        assert sg3


def test_generation_manifests_preserve_legacy_condition_counts() -> None:
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
    (inputs_root / "Giabbanelli & Macewan (categories).csv").write_text(
        "\n".join(
            [
                "A,Consumption",
                "B,Environment",
                "C,Well-being",
                "D,Social",
                "E,Weight",
                "F,Disease",
            ]
        )
    )
    (inputs_root / "Giabbanelli & Macewan (edges).csv").write_text(
        "\n".join(["A,B,1", "C,D,1", "E,F,1", "A,C,1"])
    )

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
