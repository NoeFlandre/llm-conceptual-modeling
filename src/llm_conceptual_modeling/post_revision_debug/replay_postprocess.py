import json
from pathlib import Path

import pandas as pd


def collect_replay_summaries(model_root: str | Path) -> pd.DataFrame:
    root = Path(model_root)
    summary_paths = sorted(root.rglob("summary.json"))
    records: list[dict[str, object]] = []

    for summary_path in summary_paths:
        record = _build_summary_record(root=root, summary_path=summary_path)
        if record is None:
            continue
        records.append(record)

    return pd.DataFrame(records)


def write_replay_postprocess_outputs(model_root: str | Path) -> dict[str, Path]:
    root = Path(model_root)
    frame = collect_replay_summaries(root)

    output_dir = root / "postprocess"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "replay_summary.csv"
    findings_path = output_dir / "findings.md"
    frame.to_csv(summary_csv_path, index=False)
    findings_path.write_text(_build_findings(frame))
    return {
        "summary_csv": summary_csv_path,
        "findings_md": findings_path,
    }


def _build_summary_record(*, root: Path, summary_path: Path) -> dict[str, object] | None:
    relative_path = summary_path.relative_to(root)
    parts = relative_path.parts
    if len(parts) < 4:
        return None

    algorithm = parts[0]
    pair = parts[1]
    run_leaf = parts[2]
    payload = json.loads(summary_path.read_text())

    record: dict[str, object] = {
        "algorithm": algorithm,
        "pair": pair,
        "run_leaf": run_leaf,
        "run_name": payload.get("run_name", ""),
        "model": payload.get("model", ""),
        "summary_path": str(summary_path),
    }

    if algorithm == "algo1":
        candidate_edges = _as_edge_list(payload.get("candidate_edges", []))
        verified_edges = _as_edge_list(payload.get("verified_edges", []))
        record["candidate_edge_count"] = len(candidate_edges)
        record["verified_edge_count"] = len(verified_edges)
        record["expanded_label_count"] = 0
        record["matched_label_count"] = 0
        record["final_similarity"] = None
        record["iteration_count"] = None
    elif algorithm == "algo2":
        expanded_labels = list(payload.get("expanded_labels", []))
        raw_edges = _as_edge_list(payload.get("raw_edges", []))
        normalized_edges = _as_edge_list(payload.get("normalized_edges", []))
        record["candidate_edge_count"] = 0
        record["verified_edge_count"] = 0
        record["expanded_label_count"] = len(expanded_labels)
        record["matched_label_count"] = 0
        record["raw_edge_count"] = len(raw_edges)
        record["normalized_edge_count"] = len(normalized_edges)
        record["final_similarity"] = float(payload.get("final_similarity", 0.0))
        record["iteration_count"] = int(payload.get("iteration_count", 0))
    elif algorithm == "algo3":
        expanded_nodes = list(payload.get("expanded_nodes", []))
        matched_labels = list(payload.get("matched_labels", []))
        record["candidate_edge_count"] = 0
        record["verified_edge_count"] = 0
        record["expanded_label_count"] = len(expanded_nodes)
        record["matched_label_count"] = len(matched_labels)
        record["final_similarity"] = None
        record["iteration_count"] = None
    else:
        return None

    return record


def _build_findings(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Replay Findings\n\nNo summaries found.\n"

    lines = ["# Replay Findings", ""]
    grouped = frame.groupby(["algorithm", "model"], dropna=False)
    for (algorithm, model), group in grouped:
        lines.append(
            f"- {algorithm} / {model}: {len(group)} summaries, "
            f"mean candidate edges={group['candidate_edge_count'].mean():.2f}, "
            f"mean verified edges={group['verified_edge_count'].mean():.2f}, "
            f"mean matched labels={group['matched_label_count'].mean():.2f}"
        )
    lines.append("")
    return "\n".join(lines)


def _as_edge_list(value: object) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    if not isinstance(value, list):
        return edges

    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            edges.append((str(item[0]), str(item[1])))
    return edges
