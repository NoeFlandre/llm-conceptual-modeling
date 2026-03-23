import json

from llm_conceptual_modeling.cli import main


def test_cli_monitor_algo1_reports_progress(tmp_path, capsys) -> None:
    root = tmp_path / "runs" / "demo-doe"
    algo_root = root / "algo1"
    completed = algo_root / "sg1_sg2" / "rep0_cond00000"
    failed = algo_root / "sg1_sg2" / "rep0_cond00001"
    active = algo_root / "sg2_sg3" / "rep0_cond00000"
    completed.mkdir(parents=True)
    failed.mkdir(parents=True)
    active.mkdir(parents=True)

    (completed / "summary.json").write_text(json.dumps({"status": "done"}))
    (completed / "state.json").write_text(
        json.dumps({"completed_stages": ["manifest_written", "probe_finished"]})
    )
    (failed / "error.json").write_text(json.dumps({"error_type": "ValueError"}))
    (failed / "state.json").write_text(json.dumps({"completed_stages": ["probe_failed"]}))
    (active / "state.json").write_text(json.dumps({"completed_stages": ["manifest_written"]}))

    exit_code = main(
        [
            "monitor",
            "algo1",
            "--root",
            str(root),
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Method 1 monitor:" in captured
    assert "sg1_sg2" in captured
    assert "sg2_sg3" in captured
    assert "Done" in captured
    assert "Fail" in captured
    assert "Active" in captured
    assert "480" in captured
    assert "Recent activity:" in captured


def test_cli_monitor_algo2_reports_progress(tmp_path, capsys) -> None:
    root = tmp_path / "runs" / "demo-doe"
    algo_root = root / "algo2"
    completed = algo_root / "sg1_sg2" / "rep0_cond000000"
    failed = algo_root / "sg1_sg2" / "rep0_cond000001"
    active = algo_root / "sg2_sg3" / "rep0_cond000000"
    completed.mkdir(parents=True)
    failed.mkdir(parents=True)
    active.mkdir(parents=True)

    (completed / "summary.json").write_text(json.dumps({"status": "done"}))
    (failed / "error.json").write_text(json.dumps({"error_type": "ValueError"}))
    (active / "state.json").write_text(json.dumps({"completed_stages": ["manifest_written"]}))

    exit_code = main(
        [
            "monitor",
            "algo2",
            "--root",
            str(root),
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Method 1 monitor:" in captured
    assert "sg1_sg2" in captured
    assert "sg2_sg3" in captured
    assert "960" in captured
    assert "Recent activity:" in captured


def test_cli_monitor_algo3_reports_progress(tmp_path, capsys) -> None:
    root = tmp_path / "runs" / "demo-doe"
    algo_root = root / "algo3"
    completed = algo_root / "subgraph_1_to_subgraph_3" / "rep0_cond0000"
    failed = algo_root / "subgraph_1_to_subgraph_3" / "rep0_cond0001"
    active = algo_root / "subgraph_2_to_subgraph_1" / "rep0_cond0000"
    completed.mkdir(parents=True)
    failed.mkdir(parents=True)
    active.mkdir(parents=True)

    (completed / "summary.json").write_text(json.dumps({"status": "done"}))
    (failed / "error.json").write_text(json.dumps({"error_type": "ValueError"}))
    (active / "state.json").write_text(json.dumps({"completed_stages": ["manifest_written"]}))

    exit_code = main(
        [
            "monitor",
            "algo3",
            "--root",
            str(root),
        ]
    )

    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Method 1 monitor:" in captured
    assert "subgraph_1_to_subgraph_3" in captured
    assert "subgraph_2_to_subgraph_1" in captured
    assert "240" in captured
    assert "Recent activity:" in captured
