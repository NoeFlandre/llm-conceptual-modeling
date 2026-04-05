from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from llm_conceptual_modeling.algo1.factorial import (
    run_factorial_analysis as run_algo1_factorial_analysis,
)
from llm_conceptual_modeling.algo2.factorial import (
    run_factorial_analysis as run_algo2_factorial_analysis,
)
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as evaluate_algo3_results
from llm_conceptual_modeling.algo3.factorial import (
    run_factorial_analysis as run_algo3_factorial_analysis,
)
from llm_conceptual_modeling.analysis.replication_budget import write_replication_budget_analysis
from llm_conceptual_modeling.analysis.stability import write_grouped_metric_stability
from llm_conceptual_modeling.analysis.variability import write_output_variability_analysis
from llm_conceptual_modeling.common.evaluation_core import evaluate_connection_results_file
from llm_conceptual_modeling.common.factorial_core import run_generalized_factorial_analysis
from llm_conceptual_modeling.common.types import GeneralizedFactorialSpec
from llm_conceptual_modeling.hf_batch_utils import (
    add_decoding_factor_columns,
    slugify_model,
)


def write_aggregated_outputs(output_root: Path, summary_frame: pd.DataFrame) -> None:
    aggregated_root = output_root / "aggregated"
    for group_key, group_frame in summary_frame.groupby(
        ["algorithm", "model", "condition_label"],
        dropna=False,
    ):
        algorithm, model, condition_label = cast(tuple[object, object, object], group_key)
        combo_root = (
            aggregated_root / str(algorithm) / slugify_model(str(model)) / str(condition_label)
        )
        combo_root.mkdir(parents=True, exist_ok=True)
        raw_rows = [
            json.loads(
                _resolve_materialized_artifact_path(
                    output_root=output_root,
                    artifact_path=Path(path),
                ).read_text(encoding="utf-8")
            )
            for path in group_frame["raw_row_path"].tolist()
        ]
        raw_frame = pd.DataFrame.from_records(raw_rows)
        raw_path = combo_root / "raw.csv"
        raw_frame.to_csv(raw_path, index=False)

        evaluated_path = combo_root / "evaluated.csv"
        factorial_path = combo_root / "factorial.csv"
        variability_path = combo_root / "output_variability.csv"
        stability_path = combo_root / "condition_stability.csv"
        strict_budget_path = combo_root / "replication_budget_strict.csv"
        relaxed_budget_path = combo_root / "replication_budget_relaxed.csv"

        if algorithm in {"algo1", "algo2"}:
            evaluate_connection_results_file(raw_path, evaluated_path)
            if algorithm == "algo1":
                run_algo1_factorial_analysis([evaluated_path], factorial_path)
                write_grouped_metric_stability(
                    [evaluated_path],
                    stability_path,
                    group_by=[
                        "pair_name",
                        "Explanation",
                        "Example",
                        "Counterexample",
                        "Array/List(1/-1)",
                        "Tag/Adjacency(1/-1)",
                    ],
                    metrics=["accuracy", "recall", "precision"],
                )
            else:
                run_algo2_factorial_analysis([evaluated_path], factorial_path)
                write_grouped_metric_stability(
                    [evaluated_path],
                    stability_path,
                    group_by=[
                        "pair_name",
                        "Convergence",
                        "Explanation",
                        "Example",
                        "Counterexample",
                        "Array/List(1/-1)",
                        "Tag/Adjacency(1/-1)",
                    ],
                    metrics=["accuracy", "recall", "precision"],
                )
            write_output_variability_analysis(
                [raw_path],
                variability_path,
                group_by=[
                    "pair_name",
                    "Explanation",
                    "Example",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                ]
                + (["Convergence"] if algorithm == "algo2" else []),
                result_column="Result",
            )
        else:
            evaluate_algo3_results(raw_path, evaluated_path)
            run_algo3_factorial_analysis(evaluated_path, factorial_path)
            write_grouped_metric_stability(
                [evaluated_path],
                stability_path,
                group_by=[
                    "pair_name",
                    "Depth",
                    "Number of Words",
                    "Example",
                    "Counter-Example",
                ],
                metrics=["Recall"],
            )
            write_output_variability_analysis(
                [raw_path],
                variability_path,
                group_by=[
                    "pair_name",
                    "Depth",
                    "Number of Words",
                    "Example",
                    "Counter-Example",
                ],
                result_column="Results",
            )
            _backfill_algo3_summary_artifacts(
                output_root=output_root,
                summary_frame=summary_frame,
                group_frame=group_frame,
                evaluated_path=evaluated_path,
            )

        write_replication_budget_analysis(
            [stability_path],
            strict_budget_path,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        write_replication_budget_analysis(
            [stability_path],
            relaxed_budget_path,
            relative_half_width_target=0.10,
            z_score=1.645,
        )
    _write_combined_model_outputs(aggregated_root=aggregated_root, summary_frame=summary_frame)
    summary_frame.to_csv(output_root / "batch_summary.csv", index=False)


def _backfill_algo3_summary_artifacts(
    *,
    output_root: Path,
    summary_frame: pd.DataFrame,
    group_frame: pd.DataFrame,
    evaluated_path: Path,
) -> None:
    evaluated_frame = pd.read_csv(evaluated_path)
    if len(evaluated_frame) != len(group_frame):
        raise ValueError(
            "Algo3 evaluated frame length does not match the summary frame length "
            f"for {evaluated_path}."
        )

    recalls = evaluated_frame["Recall"].tolist()
    for summary_index, recall_value, raw_row_path in zip(
        group_frame.index,
        recalls,
        group_frame["raw_row_path"].tolist(),
        strict=True,
    ):
        summary_frame.at[summary_index, "recall"] = float(recall_value)
        raw_row_summary_path = _resolve_materialized_artifact_path(
            output_root=output_root,
            artifact_path=Path(str(raw_row_path)),
        ).with_name("summary.json")
        if raw_row_summary_path.exists():
            summary_artifact = json.loads(raw_row_summary_path.read_text(encoding="utf-8"))
            summary_artifact["recall"] = float(recall_value)
            raw_row_summary_path.write_text(
                json.dumps(summary_artifact, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def _resolve_materialized_artifact_path(*, output_root: Path, artifact_path: Path) -> Path:
    if artifact_path.exists():
        return artifact_path
    if not artifact_path.is_absolute():
        return artifact_path
    try:
        relative_path = artifact_path.relative_to(Path("/workspace/results"))
    except ValueError:
        return artifact_path
    candidate_paths = [output_root / relative_path]
    if relative_path.parts and relative_path.parts[0] == output_root.name:
        candidate_paths.append(output_root / Path(*relative_path.parts[1:]))
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return artifact_path


def _write_combined_model_outputs(*, aggregated_root: Path, summary_frame: pd.DataFrame) -> None:
    for group_key, group_frame in summary_frame.groupby(["algorithm", "model"], dropna=False):
        algorithm, model = cast(tuple[object, object], group_key)
        combo_root = aggregated_root / str(algorithm) / slugify_model(str(model)) / "combined"
        combo_root.mkdir(parents=True, exist_ok=True)
        raw_rows = [
            json.loads(
                _resolve_materialized_artifact_path(
                    output_root=aggregated_root.parent,
                    artifact_path=Path(path),
                ).read_text(encoding="utf-8")
            )
            for path in group_frame["raw_row_path"].tolist()
        ]
        raw_frame = pd.DataFrame.from_records(raw_rows)
        raw_frame = add_decoding_factor_columns(raw_frame)
        raw_path = combo_root / "raw.csv"
        raw_frame.to_csv(raw_path, index=False)

        evaluated_path = combo_root / "evaluated.csv"
        factorial_path = combo_root / "factorial.csv"
        variability_path = combo_root / "output_variability.csv"
        stability_path = combo_root / "condition_stability.csv"
        strict_budget_path = combo_root / "replication_budget_strict.csv"
        relaxed_budget_path = combo_root / "replication_budget_relaxed.csv"

        if algorithm in {"algo1", "algo2"}:
            evaluate_connection_results_file(raw_path, evaluated_path)
            evaluated_frame = pd.read_csv(evaluated_path)
            evaluated_frame = add_decoding_factor_columns(evaluated_frame)
            evaluated_frame.to_csv(evaluated_path, index=False)
            _write_combined_factorial(
                algorithm=str(algorithm),
                evaluated_path=evaluated_path,
                output_path=factorial_path,
            )
            stability_group_by = [
                "pair_name",
                "Explanation",
                "Example",
                "Counterexample",
                "Array/List(1/-1)",
                "Tag/Adjacency(1/-1)",
                "Decoding Algorithm",
                "Beam Width Level",
                "Contrastive Penalty Level",
            ]
            if algorithm == "algo2":
                stability_group_by.insert(5, "Convergence")
            write_grouped_metric_stability(
                [evaluated_path],
                stability_path,
                group_by=stability_group_by,
                metrics=["accuracy", "recall", "precision"],
            )
            variability_group_by = [
                "pair_name",
                "Explanation",
                "Example",
                "Counterexample",
                "Array/List(1/-1)",
                "Tag/Adjacency(1/-1)",
                "Decoding Algorithm",
                "Beam Width Level",
                "Contrastive Penalty Level",
            ]
            if algorithm == "algo2":
                variability_group_by.append("Convergence")
            write_output_variability_analysis(
                [raw_path],
                variability_path,
                group_by=variability_group_by,
                result_column="Result",
            )
        else:
            evaluate_algo3_results(raw_path, evaluated_path)
            evaluated_frame = pd.read_csv(evaluated_path)
            evaluated_frame = add_decoding_factor_columns(evaluated_frame)
            evaluated_frame.to_csv(evaluated_path, index=False)
            _write_combined_factorial(
                algorithm=str(algorithm),
                evaluated_path=evaluated_path,
                output_path=factorial_path,
            )
            write_grouped_metric_stability(
                [evaluated_path],
                stability_path,
                group_by=[
                    "pair_name",
                    "Depth",
                    "Number of Words",
                    "Example",
                    "Counter-Example",
                    "Decoding Algorithm",
                    "Beam Width Level",
                    "Contrastive Penalty Level",
                ],
                metrics=["Recall"],
            )
            write_output_variability_analysis(
                [raw_path],
                variability_path,
                group_by=[
                    "pair_name",
                    "Depth",
                    "Number of Words",
                    "Example",
                    "Counter-Example",
                    "Decoding Algorithm",
                    "Beam Width Level",
                    "Contrastive Penalty Level",
                ],
                result_column="Results",
            )

        write_replication_budget_analysis(
            [stability_path],
            strict_budget_path,
            relative_half_width_target=0.05,
            z_score=1.96,
        )
        write_replication_budget_analysis(
            [stability_path],
            relaxed_budget_path,
            relative_half_width_target=0.10,
            z_score=1.645,
        )


def _write_combined_factorial(
    *,
    algorithm: str,
    evaluated_path: Path,
    output_path: Path,
) -> None:
    if algorithm == "algo1":
        factor_columns = [
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List(1/-1)",
            "Tag/Adjacency(1/-1)",
            "Decoding Algorithm",
            "Beam Width Level",
            "Contrastive Penalty Level",
        ]
        metric_columns = ["accuracy", "recall", "precision"]
        output_columns = ["accuracy", "recall", "precision", "Feature"]
    elif algorithm == "algo2":
        factor_columns = [
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List(1/-1)",
            "Tag/Adjacency(1/-1)",
            "Convergence",
            "Decoding Algorithm",
            "Beam Width Level",
            "Contrastive Penalty Level",
        ]
        metric_columns = ["accuracy", "recall", "precision"]
        output_columns = ["accuracy", "recall", "precision", "Feature"]
    else:
        factor_columns = [
            "Example",
            "Counter-Example",
            "Number of Words",
            "Depth",
            "Decoding Algorithm",
            "Beam Width Level",
            "Contrastive Penalty Level",
        ]
        metric_columns = ["Recall"]
        output_columns = ["Recall", "Feature"]
    run_generalized_factorial_analysis(
        [evaluated_path],
        output_path,
        GeneralizedFactorialSpec(
            factor_columns=factor_columns,
            metric_columns=metric_columns,
            output_columns=output_columns,
            replication_column="Repetition",
        ),
    )
