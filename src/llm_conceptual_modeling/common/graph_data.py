import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.paths import default_inputs_root


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
