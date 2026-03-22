"""Verify fixture paths don't have double extensions like .csv.csv"""
from pathlib import Path

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "legacy"

def test_no_double_csv_extension_in_verification_source():
    """Regression test: ensure verification.py doesn't reference paths with .csv.csv"""
    ver_file = Path(__file__).resolve().parents[2] / "src" / "llm_conceptual_modeling" / "verification.py"
    content = ver_file.read_text()
    
    # Check that no line references .csv.csv
    lines_with_double_csv = [line.strip() for line in content.splitlines() if ".csv.csv" in line]
    assert len(lines_with_double_csv) == 0, f"Found .csv.csv in verification.py: {lines_with_double_csv}"

def test_all_factorial_fixtures_exist():
    """Ensure all expected fixture files actually exist."""
    fixture_paths = [
        FIXTURES_ROOT / "algo1" / "gpt-5" / "factorial" / "factorial_analysis_algo1_gpt_5_without_error.csv",
        FIXTURES_ROOT / "algo2" / "gpt-5" / "factorial" / "factorial_analysis_gpt_5_algo2_without_error.csv",
        FIXTURES_ROOT / "algo3" / "gpt-5" / "factorial" / "factorial_analysis_results_gpt5_without_error.csv",
    ]
    
    missing = [p for p in fixture_paths if not p.exists()]
    assert len(missing) == 0, f"Missing fixtures: {missing}"
