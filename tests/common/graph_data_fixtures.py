from pathlib import Path


def write_synthetic_default_graph(inputs_root: Path) -> None:
    inputs_root.mkdir(parents=True, exist_ok=True)
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
        ),
        encoding="utf-8",
    )
    (inputs_root / "Giabbanelli & Macewan (edges).csv").write_text(
        "\n".join(["A,B,1", "C,D,1", "E,F,1", "A,C,1"]),
        encoding="utf-8",
    )
