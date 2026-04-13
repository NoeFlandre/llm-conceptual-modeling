from pathlib import Path

import pytest

from llm_conceptual_modeling.paths import default_results_root


def test_default_results_root_is_implemented_in_the_common_package() -> None:
    assert default_results_root.__module__ == "llm_conceptual_modeling.common.paths"


def test_results_layout_contains_frontier_models() -> None:
    root = Path(default_results_root())
    if not root.exists():
        pytest.skip("Result corpus is not present locally.")

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

    frontier_root = root / "frontier"
    assert frontier_root.exists()

    for algorithm, models in expected.items():
        algorithm_root = frontier_root / algorithm
        assert algorithm_root.exists()
        assert {path.name for path in algorithm_root.iterdir() if path.is_dir()} == models


def test_results_layout_contains_open_weights_canonical_bundle() -> None:
    root = Path(default_results_root())
    if not root.exists():
        pytest.skip("Result corpus is not present locally.")

    open_weights_root = root / "open_weights" / "hf-paper-batch-canonical"
    assert open_weights_root.exists()
    assert (open_weights_root / "ledger.json").exists()
    assert (open_weights_root / "runs").exists()
    assert (open_weights_root / "variance_decomposition").exists()
    assert (root / "archives").exists()
