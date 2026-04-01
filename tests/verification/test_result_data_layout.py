from pathlib import Path

import pytest

from llm_conceptual_modeling.paths import default_results_root


def test_imported_result_layout_contains_primary_models() -> None:
    root = Path(default_results_root())
    if not root.exists():
        pytest.skip("Imported result corpus is not present locally.")

    expected = {
        "algo1": {
            "deepseek-chat-v3.1",
            "deepseek-v3-chat-0324",
            "gemini-2.0-flash-exp",
            "gemini-2.5-pro",
            "gpt-5",
            "openai-gpt-4o",
        },
        "algo2": {
            "deepseek-chat-v3-0324",
            "deepseek-chat-v3.1",
            "gemini-2.0-flash-exp",
            "gemini-2.5-pro",
            "gpt-5",
            "openai-gpt-4o",
        },
        "algo3": {
            "deepseek-chat-v3.1",
            "deepseek-v3-chat-0324",
            "gemini-2.0-flash-exp",
            "google-gemini-2.5-pro",
            "gpt-5",
            "openai-gpt-4o",
        },
    }

    for algorithm, models in expected.items():
        algorithm_root = root / algorithm
        assert algorithm_root.exists()
        assert {path.name for path in algorithm_root.iterdir() if path.is_dir()} == models


def test_imported_result_layout_contains_raw_and_evaluated_csvs() -> None:
    root = Path(default_results_root())
    if not root.exists():
        pytest.skip("Imported result corpus is not present locally.")

    for model_root in (path for path in root.glob("algo1/*") if path.is_dir()):
        assert len(list((model_root / "raw").glob("*.csv"))) == 3
        assert len(list((model_root / "evaluated").glob("*.csv"))) == 3

    for model_root in (path for path in root.glob("algo2/*") if path.is_dir()):
        assert len(list((model_root / "raw").glob("*.csv"))) == 3
        assert len(list((model_root / "evaluated").glob("*.csv"))) == 3

    for model_root in (path for path in root.glob("algo3/*") if path.is_dir()):
        assert len(list((model_root / "raw").glob("*.csv"))) == 1
        assert len(list((model_root / "evaluated").glob("*.csv"))) == 1
