import json
from pathlib import Path

from llm_conceptual_modeling.post_revision_debug.replay_postprocess import (
    collect_replay_summaries,
)


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_collect_replay_summaries_extracts_model_and_counts(tmp_path: Path) -> None:
    root = tmp_path / "paper_replay_2models" / "mistral-small-2603"
    _write_summary(
        root / "algo1" / "sg1_sg2" / "rep0_cond00000" / "summary.json",
        {
            "run_name": "algo1_sg1_sg2_rep0_cond00000",
            "model": "mistral-small-2603",
            "candidate_edges": [["a", "b"], ["c", "d"]],
            "verified_edges": [["a", "b"]],
        },
    )
    _write_summary(
        root / "algo2" / "sg2_sg3" / "rep1_cond00001" / "summary.json",
        {
            "run_name": "algo2_sg2_sg3_rep1_cond00001",
            "model": "mistral-small-2603",
            "expanded_labels": ["x", "y"],
            "raw_edges": [["a", "b"]],
            "normalized_edges": [["a", "b"]],
            "final_similarity": 0.8,
            "iteration_count": 3,
        },
    )
    _write_summary(
        root / "algo3" / "subgraph_1_to_subgraph_3" / "rep4_cond1111" / "summary.json",
        {
            "run_name": "algo3_subgraph_1_to_subgraph_3_rep4_cond1111",
            "model": "mistral-small-2603",
            "expanded_nodes": [
                {
                    "root_label": "r",
                    "parent_label": None,
                    "label": "child",
                    "depth": 1,
                    "matched_target": True,
                }
            ],
            "matched_labels": ["child"],
        },
    )

    frame = collect_replay_summaries(root)

    assert list(frame["algorithm"]) == ["algo1", "algo2", "algo3"]
    assert list(frame["model"]) == ["mistral-small-2603"] * 3
    assert list(frame["pair"]) == [
        "sg1_sg2",
        "sg2_sg3",
        "subgraph_1_to_subgraph_3",
    ]
    assert list(frame["run_name"]) == [
        "algo1_sg1_sg2_rep0_cond00000",
        "algo2_sg2_sg3_rep1_cond00001",
        "algo3_subgraph_1_to_subgraph_3_rep4_cond1111",
    ]
    assert list(frame["candidate_edge_count"]) == [2, 0, 0]
    assert list(frame["verified_edge_count"]) == [1, 0, 0]
    assert list(frame["expanded_label_count"]) == [0, 2, 1]
    assert list(frame["matched_label_count"]) == [0, 0, 1]
