import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.cli import main
from llm_conceptual_modeling.verification.cases import FIXTURES_ROOT


def _run_installed_lcm(
    tmp_path: Path, fixtures_root: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    startup_dir = tmp_path / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "sitecustomize.py").write_text(
        "import importlib, os\n"
        "from pathlib import Path\n"
        "from llm_conceptual_modeling.verification import cases\n"
        "root = Path(os.environ['LCM_TEST_FIXTURES_ROOT'])\n"
        "cases.FIXTURES_ROOT = root\n"
        "doctor = importlib.import_module(\n"
        "    'llm_conceptual_modeling.verification.doctor')\n"
        "doctor.FIXTURES_ROOT = root\n",
        encoding="utf-8",
    )
    executable = Path(sys.executable).with_name("lcm")
    assert executable.is_file(), "the lcm console script must be installed for this test"
    environment = os.environ.copy()
    environment["LCM_TEST_FIXTURES_ROOT"] = str(fixtures_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(startup_dir), environment.get("PYTHONPATH")])
    )
    return subprocess.run(
        [str(executable), *args],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _temporary_fixture_tree(tmp_path: Path) -> Path:
    fixtures_root = tmp_path / "legacy"
    shutil.copytree(FIXTURES_ROOT, fixtures_root, copy_function=os.symlink)
    expected = fixtures_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv"
    expected.unlink()
    frame = pd.read_csv(FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv")
    frame.loc[0, "accuracy"] = float(frame.loc[0, "accuracy"]) + 0.01
    frame.to_csv(expected, index=False)
    return fixtures_root


def test_cli_doctor_reports_ok_status(capsys) -> None:
    exit_code = main(["doctor", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["checks"]["fixtures_present"] is True
    assert payload["checks"]["package_import"] is True


def test_cli_doctor_reports_remote_preflight_checks(tmp_path, capsys) -> None:
    results_root = tmp_path / "results"
    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir()
    (smoke_root / "smoke_verdict.json").write_text(
        json.dumps({"status": "success", "worker_loaded_model": True}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "doctor",
            "--json",
            "--results-root",
            str(results_root),
            "--smoke-root",
            str(smoke_root),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["checks"]["results_root_writable"] is True
    assert payload["checks"]["smoke_verdict_present"] is True
    assert payload["checks"]["smoke_verdict"]["status"] == "success"


def test_cli_verify_legacy_parity_reports_all_workflows_green(capsys) -> None:
    exit_code = main(["verify", "legacy-parity", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert all(result["status"] == "passed" for result in payload["results"])
    assert len(payload["results"]) == 6


def test_installed_verify_returns_failure_for_a_temporary_parity_mismatch(tmp_path) -> None:
    fixtures_root = _temporary_fixture_tree(tmp_path)

    result = _run_installed_lcm(tmp_path, fixtures_root, "verify", "all", "--json")

    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["legacy_parity"]["status"] == "error"
    assert any(case["status"] == "failed" for case in payload["legacy_parity"]["results"])
    assert result.returncode == 1


def test_installed_doctor_exit_status_tracks_its_report_and_keeps_json(tmp_path) -> None:
    error_result = _run_installed_lcm(
        tmp_path / "error", tmp_path / "missing-fixtures", "doctor", "--json"
    )
    error_payload = json.loads(error_result.stdout)
    assert error_result.returncode == 1
    assert error_payload["status"] == "error"

    good_result = _run_installed_lcm(tmp_path / "good", FIXTURES_ROOT, "doctor", "--json")
    good_payload = json.loads(good_result.stdout)
    assert good_result.returncode == 0
    assert good_payload == {
        "status": "ok",
        "checks": {"fixtures_present": True, "package_import": True},
    }
