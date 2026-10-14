import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.common.graph_data_fixtures import write_synthetic_default_graph

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_installed_cli_validates_paper_config_with_synthetic_graph_inputs(tmp_path: Path) -> None:
    inputs_root = tmp_path / "inputs"
    output_dir = tmp_path / "preview"
    write_synthetic_default_graph(inputs_root)

    environment = os.environ.copy()
    environment["LCM_INPUTS_ROOT"] = str(inputs_root)
    executable = Path(sys.executable).with_name("lcm")
    assert executable.is_file(), "the lcm console script must be installed for this smoke test"

    result = subprocess.run(
        [
            str(executable),
            "run",
            "validate-config",
            "--config",
            str(REPO_ROOT / "configs" / "hf_transformers_paper_batch.yaml"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads((output_dir / "resolved_run_plan.json").read_text(encoding="utf-8"))
    assert plan["graph_sources"] == ["default"]
    with (output_dir / "condition_matrix.csv").open(encoding="utf-8") as condition_matrix:
        rows = list(csv.DictReader(condition_matrix))
    assert len(rows) == plan["planned_total_runs"]
    assert (output_dir / "prompt_preview" / "algo1" / "base.txt").is_file()
