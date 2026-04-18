import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from llm_conceptual_modeling.paths import default_inputs_root

GraphEdges = list[tuple[str, str]]


@dataclass(frozen=True)
class GraphSourceSpec:
    source_id: str
    display_name: str
    categories_path: Path
    edges_path: Path
    cluster_labels: tuple[str, str, str]


def _inputs_root() -> Path:
    return Path(default_inputs_root())


def categories_csv_path() -> Path:
    return _inputs_root() / "Giabbanelli & Macewan (categories).csv"


def edges_csv_path() -> Path:
    return _inputs_root() / "Giabbanelli & Macewan (edges).csv"


def algo2_thesaurus_json_path() -> Path:
    return _inputs_root() / "algo2_thesaurus.json"


def wordnet_label_lexicon_json_path() -> Path:
    return _inputs_root() / "wordnet_label_lexicon.json"


def open_weight_map_extension_manifest_path() -> Path:
    return _inputs_root() / "open_weight_map_extension" / "manifest.yaml"


def load_algo2_thesaurus() -> dict[str, dict[str, list[str]]]:
    raw_text = algo2_thesaurus_json_path().read_text()
    parsed_thesaurus = json.loads(raw_text)
    return parsed_thesaurus


@lru_cache(maxsize=1)
def load_wordnet_label_lexicon() -> dict[str, list[str]]:
    raw_text = wordnet_label_lexicon_json_path().read_text()
    parsed_lexicon = json.loads(raw_text)
    return {str(key): list(value) for key, value in parsed_lexicon.items()}


def load_default_graph() -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    return load_graph_source("default")


def available_graph_sources() -> tuple[str, ...]:
    return ("default", *tuple(_load_map_extension_specs().keys()))


def load_graph_source(source_id: str) -> tuple[GraphEdges, GraphEdges, GraphEdges, GraphEdges]:
    if source_id == "default":
        return _load_legacy_default_graph()
    specs = _load_map_extension_specs()
    if source_id not in specs:
        raise ValueError(f"Unknown graph source: {source_id}")
    return _load_clustered_graph(specs[source_id])


def _load_legacy_default_graph() -> tuple[GraphEdges, GraphEdges, GraphEdges, GraphEdges]:
    categories = pd.read_csv(categories_csv_path(), header=None)
    edges_frame = pd.read_csv(edges_csv_path(), header=None)
    category_map = dict(zip(categories.iloc[:, 0], categories.iloc[:, 1], strict=True))
    edges = [(row[0], row[1]) for _, row in edges_frame.iterrows() if row[2] != 0]

    def subgraph(allowed_categories: list[str]) -> list[tuple[str, str]]:
        return [
            (head, tail)
            for head, tail in edges
            if category_map.get(head) in allowed_categories
            and category_map.get(tail) in allowed_categories
        ]

    sg1 = subgraph(["Consumption", "Production", "Environment"])
    sg2 = subgraph(["Well-being", "Social", "Psychology"])
    sg3 = subgraph(["Weight", "Physiology", "Disease", "Physical activity"])
    return sg1, sg2, sg3, edges


@lru_cache(maxsize=1)
def _load_map_extension_specs() -> dict[str, GraphSourceSpec]:
    manifest_path = open_weight_map_extension_manifest_path()
    if not manifest_path.exists():
        return {}
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw_sources = payload.get("graph_sources", {})
    specs: dict[str, GraphSourceSpec] = {}
    for source_id, raw_spec in raw_sources.items():
        cluster_labels = tuple(str(label) for label in raw_spec["cluster_labels"])
        if len(cluster_labels) != 3:
            raise ValueError(f"Graph source {source_id} must define exactly three clusters.")
        specs[str(source_id)] = GraphSourceSpec(
            source_id=str(source_id),
            display_name=str(raw_spec["display_name"]),
            categories_path=_inputs_root() / str(raw_spec["categories_path"]),
            edges_path=_inputs_root() / str(raw_spec["edges_path"]),
            cluster_labels=(
                cluster_labels[0],
                cluster_labels[1],
                cluster_labels[2],
            ),
        )
    return specs


def _load_clustered_graph(
    spec: GraphSourceSpec,
) -> tuple[GraphEdges, GraphEdges, GraphEdges, GraphEdges]:
    categories = pd.read_csv(spec.categories_path, header=None)
    edges_frame = pd.read_csv(spec.edges_path, header=None)
    category_map = {str(row[0]): str(row[1]) for _, row in categories.iterrows()}
    edges = [
        (str(row[0]), str(row[1]))
        for _, row in edges_frame.iterrows()
        if row[2] != 0
    ]
    unknown_nodes = sorted(
        {
            node
            for edge in edges
            for node in edge
            if node not in category_map
        }
    )
    if unknown_nodes:
        raise ValueError(
            f"Graph source {spec.source_id} has edges with unknown nodes: {unknown_nodes}"
        )
    subgraphs = tuple(
        [
            (head, tail)
            for head, tail in edges
            if category_map[head] == cluster_label and category_map[tail] == cluster_label
        ]
        for cluster_label in spec.cluster_labels
    )
    if any(not subgraph for subgraph in subgraphs):
        raise ValueError(f"Graph source {spec.source_id} must have three non-empty subgraphs.")
    return subgraphs[0], subgraphs[1], subgraphs[2], edges
