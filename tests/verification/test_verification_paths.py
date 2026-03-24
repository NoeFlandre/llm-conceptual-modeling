"""Verify fixture paths don't have double extensions like .csv.csv"""

from pathlib import Path

VER_FILE = (
    Path(__file__).resolve().parents[2] / "src" / "llm_conceptual_modeling" / "verification.py"
)

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "reference_fixtures" / "legacy"

FACTORIAL_FIXTURES = [
    FIXTURES_ROOT
    / "algo1"
    / "gpt-5"
    / "factorial"
    / "factorial_analysis_algo1_gpt_5_without_error.csv",
    FIXTURES_ROOT
    / "algo2"
    / "gpt-5"
    / "factorial"
    / "factorial_analysis_gpt_5_algo2_without_error.csv",
    FIXTURES_ROOT
    / "algo3"
    / "gpt-5"
    / "factorial"
    / "factorial_analysis_results_gpt5_without_error.csv",
]


def test_no_double_csv_extension_in_verification_source():
    """Regression: ensure verification.py has no .csv.csv paths."""
    content = VER_FILE.read_text()
    bad_lines = [line.strip() for line in content.splitlines() if ".csv.csv" in line]
    assert len(bad_lines) == 0, f"Found .csv.csv: {bad_lines}"


def test_all_factorial_fixtures_exist():
    """Ensure all expected fixture files actually exist."""
    missing = [p for p in FACTORIAL_FIXTURES if not p.exists()]
    assert len(missing) == 0, f"Missing fixtures: {missing}"
