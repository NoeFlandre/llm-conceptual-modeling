from pathlib import Path
from shutil import copyfile

import pandas as pd

from llm_conceptual_modeling.analysis.hypothesis_bundle import write_hypothesis_testing_bundle


def test_write_hypothesis_testing_bundle_creates_organized_outputs(tmp_path) -> None:
    results_root = tmp_path / "results"
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv",
        results_root / "algo3" / "gpt-5" / "evaluated" / "method3_results_evaluated_gpt5.csv",
    )
    output_dir = tmp_path / "bundle"

    write_hypothesis_testing_bundle(results_root=results_root, output_dir=output_dir)

    assert (output_dir / "README.md").exists()
    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "algo1" / "explanation" / "paired_tests.csv").exists()
    assert (output_dir / "algo2" / "convergence" / "factor_overview.csv").exists()
    assert (output_dir / "algo3" / "depth" / "significance_summary.csv").exists()

    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    assert set(overview["algorithm"]) == {"algo1", "algo2", "algo3"}
    assert "Convergence" in set(overview["factor"])
    assert "Depth" in set(overview["factor"])
    assert {"significant_test_count", "significant_share"}.issubset(overview.columns)


def _copy_fixture(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, destination)
