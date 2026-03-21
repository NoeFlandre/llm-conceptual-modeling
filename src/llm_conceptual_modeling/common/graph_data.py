from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "inputs"
CATEGORIES_CSV = DATA_ROOT / "Giabbanelli & Macewan (categories).csv"
EDGES_CSV = DATA_ROOT / "Giabbanelli & Macewan (edges).csv"

def load_default_graph() -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    categories = pd.read_csv(CATEGORIES_CSV, header=None)
    edges_frame = pd.read_csv(EDGES_CSV, header=None)
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
