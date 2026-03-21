import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_analyze_failures_classifies_raw_rows(tmp_path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "failures.csv"
    pd.DataFrame(
        {
            "Result": [
                "[('a', 'b')]",
                "[]",
                "not parseable",
                None,
            ],
            "Setting": ["x", "y", "z", "w"],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "failures",
            "--input",
            str(input_path),
            "--result-column",
            "Result",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    assert list(actual["failure_category"]) == [
        "valid_output",
        "empty_output",
        "malformed_output",
        "empty_output",
    ]
    assert list(actual["parsed_edge_count"]) == [1, 0, 0, 0]
    assert list(actual["is_failure"]) == [False, True, True, True]


def test_cli_analyze_failures_rejects_missing_result_column(tmp_path, capsys) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "failures.csv"
    pd.DataFrame({"Results": ["[]"]}).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "failures",
            "--input",
            str(input_path),
            "--result-column",
            "Result",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing required result columns" in captured.err
    assert not output_path.exists()
