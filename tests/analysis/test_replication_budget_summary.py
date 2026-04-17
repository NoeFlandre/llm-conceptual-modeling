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
) -> dict[str, object]:
    return {
        "identity": {
            "algorithm": algorithm,
            "model": model,
            "condition_label": condition_label,
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
