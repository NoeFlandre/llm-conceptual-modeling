import pandas as pd

from llm_conceptual_modeling.post_revision_debug.mistral_probe import (
    extract_edge_list_from_chat_content,
    score_algo1_row,
    score_algo3_row,
)


def test_extract_edge_list_from_chat_content_supports_markdown_python_list() -> None:
    content = """Here are the edges:

```python
[("A", "B"), ("C", "D")]
```
"""

    actual = extract_edge_list_from_chat_content(content)

    assert actual == [("A", "B"), ("C", "D")]


def test_extract_edge_list_from_chat_content_supports_json_schema_payload() -> None:
    content = '{"edges":[{"source":"A","target":"B"},{"source":"C","target":"D"}]}'

    actual = extract_edge_list_from_chat_content(content)

    assert actual == [("A", "B"), ("C", "D")]


def test_score_algo1_row_matches_fixture_metrics() -> None:
    raw_path = "tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv"
    evaluated_path = "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv"

    raw_row = pd.read_csv(raw_path).iloc[0]
    evaluated_row = pd.read_csv(evaluated_path).iloc[0]

    actual = score_algo1_row(raw_row, raw_row["Result"])

    assert round(actual["accuracy"], 6) == round(float(evaluated_row["accuracy"]), 6)
    assert round(actual["recall"], 6) == round(float(evaluated_row["recall"]), 6)
    assert round(actual["precision"], 6) == round(float(evaluated_row["precision"]), 6)


def test_score_algo3_row_matches_fixture_recall() -> None:
    raw_path = "tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv"
    evaluated_path = (
        "tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    )

    raw_row = pd.read_csv(raw_path).iloc[0]
    evaluated_row = pd.read_csv(evaluated_path).iloc[0]

    actual = score_algo3_row(raw_row, raw_row["Results"])

    assert round(actual["recall"], 6) == round(float(evaluated_row["Recall"]), 6)
