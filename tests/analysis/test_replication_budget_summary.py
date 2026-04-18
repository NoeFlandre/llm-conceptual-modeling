import json
from pathlib import Path

import pandas as pd


def _ledger_record(
    *,
    algorithm: str,
    model: str,
    condition_label: str,
    metric_values: dict[str, float],
    replication: int,
    graph_source: str = "default",
) -> dict[str, object]:
    return {
        "identity": {
            "algorithm": algorithm,
            "model": model,
            "condition_label": condition_label,
            "graph_source": graph_source,
            "pair_name": "sg1_sg2",
            "condition_bits": "000000",
            "replication": replication,
        },
        "status": "finished",
        "winner": {
            "metrics": metric_values,
            "status": "finished",
            "validated": True,
        },
    }


def test_write_replication_budget_sufficiency_summary_groups_underpowered_conditions(
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.analysis.replication_budget_summary import (
        write_replication_budget_sufficiency_summary,
    )

    results_root = tmp_path / "results"
    results_root.mkdir()
    qwen_accuracy_values = [88.0, 88.0, 100.0, 112.0, 112.0]
    records = [
        _ledger_record(
            algorithm="algo2",
            model="Qwen/Qwen3.5-9B",
            condition_label="contrastive_penalty_alpha_0.8",
            metric_values={"accuracy": accuracy, "precision": 100.0},
            replication=replication,
        )
        for replication, accuracy in enumerate(qwen_accuracy_values)
    ]
    records.extend(
        _ledger_record(
            algorithm="algo3",
            model="mistralai/Ministral-3-8B-Instruct-2512",
            condition_label="beam_num_beams_2",
            metric_values={"recall": 50.0},
            replication=replication,
        )
        for replication in range(5)
    )
    (results_root / "ledger.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.csv"

    write_replication_budget_sufficiency_summary(
        results_root=results_root,
        output_csv_path=output_path,
    )

    summary = pd.read_csv(output_path)
    overall_95 = summary[
        (summary["profile"] == "ci95_rel05") & (summary["grouping"] == "overall")
    ].iloc[0]
    overall_90 = summary[
        (summary["profile"] == "ci90_rel05") & (summary["grouping"] == "overall")
    ].iloc[0]
    qwen_accuracy_95 = summary[
        (summary["profile"] == "ci95_rel05")
        & (summary["grouping"] == "algorithm_model_decoding_metric")
        & (summary["algorithm"] == "algo2")
        & (summary["metric"] == "accuracy")
    ].iloc[0]
    qwen_accuracy_detail_95 = summary[
        (summary["profile"] == "ci95_rel05")
        & (summary["grouping"] == "underpowered_condition_metric")
        & (summary["algorithm"] == "algo2")
        & (summary["metric"] == "accuracy")
    ].iloc[0]
    algo2_95 = summary[
        (summary["profile"] == "ci95_rel05")
        & (summary["grouping"] == "algorithm")
        & (summary["algorithm"] == "algo2")
    ].iloc[0]

    assert overall_95["source_finished_run_count"] == 10
    assert overall_95["condition_count"] == 3
    assert overall_95["conditions_needing_more_runs"] == 1
    assert overall_95["additional_runs_needed_total"] == 18
    assert overall_90["additional_runs_needed_total"] == 11
    assert algo2_95["model"] == "ALL"
    assert qwen_accuracy_95["model"] == "Qwen/Qwen3.5-9B"
    assert qwen_accuracy_95["required_total_runs_max"] == 23
    assert qwen_accuracy_detail_95["pair_name"] == "sg1_sg2"
    assert qwen_accuracy_detail_95["condition_bits"] == "000000"


def test_write_replication_budget_sufficiency_summary_groups_by_graph_source(
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.analysis.replication_budget_summary import (
        write_replication_budget_sufficiency_summary,
    )

    results_root = tmp_path / "results"
    results_root.mkdir()
    records = [
        _ledger_record(
            algorithm="algo3",
            model="Qwen/Qwen3.5-9B",
            condition_label="beam_num_beams_6",
            metric_values={"recall": recall},
            replication=replication,
            graph_source="babs_johnson",
        )
        for replication, recall in enumerate([20.0, 20.0, 50.0, 80.0, 80.0])
    ]
    records.extend(
        _ledger_record(
            algorithm="algo3",
            model="Qwen/Qwen3.5-9B",
            condition_label="beam_num_beams_6",
            metric_values={"recall": 50.0},
            replication=replication,
            graph_source="clarice_starling",
        )
        for replication in range(5)
    )
    (results_root / "ledger.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.csv"

    write_replication_budget_sufficiency_summary(
        results_root=results_root,
        output_csv_path=output_path,
    )

    summary = pd.read_csv(output_path)

    assert "graph_source" in summary.columns
    assert {"graph_source", "algorithm_model_graph_source", "algorithm_model_graph_source_decoding",
        "algorithm_model_graph_source_decoding_metric"} <= set(summary["grouping"])
    graph_rows = summary[
        (summary["profile"] == "ci95_rel05") & (summary["grouping"] == "graph_source")
    ]
    assert set(graph_rows["graph_source"]) == {"babs_johnson", "clarice_starling"}
    babs_row = graph_rows[graph_rows["graph_source"] == "babs_johnson"].iloc[0]
    clarice_row = graph_rows[graph_rows["graph_source"] == "clarice_starling"].iloc[0]
    assert babs_row["conditions_needing_more_runs"] == 1
    assert clarice_row["conditions_needing_more_runs"] == 0


def test_write_compact_replication_budget_sufficiency_table_pivots_by_model(
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.analysis.replication_budget_summary import (
        write_compact_replication_budget_sufficiency_table,
    )

    results_root = tmp_path / "results"
    results_root.mkdir()
    records = [
        _ledger_record(
            algorithm="algo1",
            model="Qwen/Qwen3.5-9B",
            condition_label="greedy",
            metric_values={"accuracy": accuracy, "precision": 100.0},
            replication=replication,
        )
        for replication, accuracy in enumerate([88.0, 88.0, 100.0, 112.0, 112.0])
    ]
    records.extend(
        _ledger_record(
            algorithm="algo1",
            model="mistralai/Ministral-3-8B-Instruct-2512",
            condition_label="greedy",
            metric_values={"accuracy": 100.0, "precision": 100.0},
            replication=replication,
        )
        for replication in range(5)
    )
    (results_root / "ledger.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )
    output_path = tmp_path / "compact.csv"

    write_compact_replication_budget_sufficiency_table(
        results_root=results_root,
        output_csv_path=output_path,
    )

    table = pd.read_csv(output_path)

    assert table.columns.tolist() == [
        "algorithm",
        "decoding",
        "qwen_runs",
        "qwen_condition_metrics",
        "qwen_ci90_needing_more",
        "qwen_ci90_share",
        "qwen_ci90_max_required_runs",
        "qwen_ci95_needing_more",
        "qwen_ci95_share",
        "qwen_ci95_max_required_runs",
        "mistral_runs",
        "mistral_condition_metrics",
        "mistral_ci90_needing_more",
        "mistral_ci90_share",
        "mistral_ci90_max_required_runs",
        "mistral_ci95_needing_more",
        "mistral_ci95_share",
        "mistral_ci95_max_required_runs",
    ]
    assert table.to_dict("records") == [
        {
            "algorithm": "Algorithm 1",
            "decoding": "greedy",
            "qwen_runs": 5,
            "qwen_condition_metrics": 2,
            "qwen_ci90_needing_more": 1,
            "qwen_ci90_share": 0.5,
            "qwen_ci90_max_required_runs": 16,
            "qwen_ci95_needing_more": 1,
            "qwen_ci95_share": 0.5,
            "qwen_ci95_max_required_runs": 23,
            "mistral_runs": 5,
            "mistral_condition_metrics": 2,
            "mistral_ci90_needing_more": 0,
            "mistral_ci90_share": 0.0,
            "mistral_ci90_max_required_runs": 5,
            "mistral_ci95_needing_more": 0,
            "mistral_ci95_share": 0.0,
            "mistral_ci95_max_required_runs": 5,
        }
    ]


def test_write_compact_replication_budget_sufficiency_table_can_split_by_graph_source(
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.analysis.replication_budget_summary import (
        write_compact_replication_budget_sufficiency_table,
    )

    results_root = tmp_path / "results"
    results_root.mkdir()
    records = [
        _ledger_record(
            algorithm="algo3",
            model="Qwen/Qwen3.5-9B",
            condition_label="beam_num_beams_6",
            metric_values={"recall": recall},
            replication=replication,
            graph_source="babs_johnson",
        )
        for replication, recall in enumerate([20.0, 20.0, 50.0, 80.0, 80.0])
    ]
    records.extend(
        _ledger_record(
            algorithm="algo3",
            model="mistralai/Ministral-3-8B-Instruct-2512",
            condition_label="beam_num_beams_6",
            metric_values={"recall": 50.0},
            replication=replication,
            graph_source="babs_johnson",
        )
        for replication in range(5)
    )
    records.extend(
        _ledger_record(
            algorithm="algo3",
            model="Qwen/Qwen3.5-9B",
            condition_label="beam_num_beams_6",
            metric_values={"recall": 50.0},
            replication=replication,
            graph_source="clarice_starling",
        )
        for replication in range(5)
    )
    records.extend(
        _ledger_record(
            algorithm="algo3",
            model="mistralai/Ministral-3-8B-Instruct-2512",
            condition_label="beam_num_beams_6",
            metric_values={"recall": 50.0},
            replication=replication,
            graph_source="clarice_starling",
        )
        for replication in range(5)
    )
    (results_root / "ledger.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )
    output_path = tmp_path / "compact.csv"

    write_compact_replication_budget_sufficiency_table(
        results_root=results_root,
        output_csv_path=output_path,
        include_graph_source=True,
    )

    table = pd.read_csv(output_path)

    assert table.columns.tolist()[:3] == ["algorithm", "graph_source", "decoding"]
    assert table.to_dict("records") == [
        {
            "algorithm": "Algorithm 3",
            "graph_source": "babs_johnson",
            "decoding": "beam_num_beams_6",
            "qwen_runs": 5,
            "qwen_condition_metrics": 1,
            "qwen_ci90_needing_more": 1,
            "qwen_ci90_share": 1.0,
            "qwen_ci90_max_required_runs": 390,
            "qwen_ci95_needing_more": 1,
            "qwen_ci95_share": 1.0,
            "qwen_ci95_max_required_runs": 554,
            "mistral_runs": 5,
            "mistral_condition_metrics": 1,
            "mistral_ci90_needing_more": 0,
            "mistral_ci90_share": 0.0,
            "mistral_ci90_max_required_runs": 5,
            "mistral_ci95_needing_more": 0,
            "mistral_ci95_share": 0.0,
            "mistral_ci95_max_required_runs": 5,
        },
        {
            "algorithm": "Algorithm 3",
            "graph_source": "clarice_starling",
            "decoding": "beam_num_beams_6",
            "qwen_runs": 5,
            "qwen_condition_metrics": 1,
            "qwen_ci90_needing_more": 0,
            "qwen_ci90_share": 0.0,
            "qwen_ci90_max_required_runs": 5,
            "qwen_ci95_needing_more": 0,
            "qwen_ci95_share": 0.0,
            "qwen_ci95_max_required_runs": 5,
            "mistral_runs": 5,
            "mistral_condition_metrics": 1,
            "mistral_ci90_needing_more": 0,
            "mistral_ci90_share": 0.0,
            "mistral_ci90_max_required_runs": 5,
            "mistral_ci95_needing_more": 0,
            "mistral_ci95_share": 0.0,
            "mistral_ci95_max_required_runs": 5,
        },
    ]
