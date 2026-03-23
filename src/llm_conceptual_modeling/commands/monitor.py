from __future__ import annotations

import json
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ConditionSnapshot:
    path: Path
    status: str
    updated_at: float
    latest_file: str | None
    stage: str | None


@dataclass(frozen=True)
class PairSnapshot:
    pair_name: str
    total: int
    completed: int
    failed: int
    active: int
    pending: int
    latest_condition: ConditionSnapshot | None
    recent_conditions: list[ConditionSnapshot]


@dataclass(frozen=True)
class MonitorSnapshot:
    root: Path
    algorithm: str
    pair_snapshots: list[PairSnapshot]


@dataclass(frozen=True)
class AlgorithmLayout:
    pair_names: tuple[str, ...]
    replications: int
    condition_bits: int


_LAYOUTS: dict[str, AlgorithmLayout] = {
    "algo1": AlgorithmLayout(
        pair_names=("sg1_sg2", "sg2_sg3", "sg3_sg1"),
        replications=5,
        condition_bits=5,
    ),
    "algo2": AlgorithmLayout(
        pair_names=("sg1_sg2", "sg2_sg3", "sg3_sg1"),
        replications=5,
        condition_bits=6,
    ),
    "algo3": AlgorithmLayout(
        pair_names=(
            "subgraph_1_to_subgraph_3",
            "subgraph_2_to_subgraph_1",
            "subgraph_2_to_subgraph_3",
        ),
        replications=5,
        condition_bits=4,
    ),
}


def handle_monitor(args: Namespace) -> int:
    root = Path(args.root)
    interval = float(args.interval)
    watch = bool(args.watch)

    try:
        while True:
            snapshot = _scan_monitor_snapshot(root=root, algorithm=args.algorithm)
            _render_monitor_snapshot(snapshot, stream=sys.stdout)
            if not watch:
                return 0
            time.sleep(interval)
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
    except KeyboardInterrupt:
        return 130


def _scan_monitor_snapshot(*, root: Path, algorithm: str) -> MonitorSnapshot:
    algorithm_root = _resolve_algorithm_root(root=root, algorithm=algorithm)
    if not algorithm_root.exists():
        raise FileNotFoundError(f"Monitor root does not exist: {algorithm_root}")

    layout = _LAYOUTS.get(algorithm)
    if layout is None:
        pair_dirs = [p for p in sorted(algorithm_root.iterdir()) if p.is_dir()]
    else:
        pair_dirs = _expected_pair_dirs(algorithm_root=algorithm_root, layout=layout)
    pair_snapshots = [_scan_pair_snapshot(pair_dir, layout=layout) for pair_dir in pair_dirs]
    return MonitorSnapshot(root=algorithm_root, algorithm=algorithm, pair_snapshots=pair_snapshots)


def _resolve_algorithm_root(*, root: Path, algorithm: str) -> Path:
    candidate = root / algorithm
    return candidate if candidate.exists() else root


def _expected_pair_dirs(*, algorithm_root: Path, layout: AlgorithmLayout) -> list[Path]:
    return [algorithm_root / pair_name for pair_name in layout.pair_names]


def _scan_pair_snapshot(pair_dir: Path, *, layout: AlgorithmLayout | None) -> PairSnapshot:
    if layout is None:
        condition_dirs = (
            [p for p in sorted(pair_dir.iterdir()) if p.is_dir()] if pair_dir.exists() else []
        )
        condition_snapshots = [_scan_condition_snapshot(cond_dir) for cond_dir in condition_dirs]
    else:
        condition_snapshots = [
            _scan_condition_snapshot(cond_dir)
            for cond_dir in _expected_condition_dirs(pair_dir=pair_dir, layout=layout)
        ]
    completed = sum(1 for snapshot in condition_snapshots if snapshot.status == "completed")
    failed = sum(1 for snapshot in condition_snapshots if snapshot.status == "failed")
    active = sum(1 for snapshot in condition_snapshots if snapshot.status == "active")
    pending = sum(1 for snapshot in condition_snapshots if snapshot.status == "pending")
    latest_condition = max(condition_snapshots, key=lambda item: item.updated_at, default=None)
    recent_conditions = sorted(condition_snapshots, key=lambda item: item.updated_at, reverse=True)[
        :5
    ]
    return PairSnapshot(
        pair_name=pair_dir.name,
        total=len(condition_snapshots),
        completed=completed,
        failed=failed,
        active=active,
        pending=pending,
        latest_condition=latest_condition,
        recent_conditions=recent_conditions,
    )


def _expected_condition_dirs(*, pair_dir: Path, layout: AlgorithmLayout) -> list[Path]:
    return [
        pair_dir / f"rep{rep_index}_cond{''.join(str(bit) for bit in condition_bits)}"
        for rep_index in range(layout.replications)
        for condition_bits in product([0, 1], repeat=layout.condition_bits)
    ]


def _scan_condition_snapshot(condition_dir: Path) -> ConditionSnapshot:
    summary_path = condition_dir / "summary.json"
    error_path = condition_dir / "error.json"
    state_path = condition_dir / "state.json"
    files = (
        [path for path in condition_dir.iterdir() if path.is_file()]
        if condition_dir.exists()
        else []
    )
    latest_file = max(files, key=lambda path: path.stat().st_mtime, default=None)
    latest_file_name = latest_file.name if latest_file is not None else None
    updated_at = latest_file.stat().st_mtime if latest_file is not None else 0.0

    if summary_path.exists():
        status = "completed"
    elif error_path.exists():
        status = "failed"
    elif state_path.exists() or condition_dir.exists():
        status = "active"
    else:
        status = "pending"

    stage = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
        completed_stages = state.get("completed_stages", [])
        if isinstance(completed_stages, list) and completed_stages:
            stage = str(completed_stages[-1])

    return ConditionSnapshot(
        path=condition_dir,
        status=status,
        updated_at=updated_at,
        latest_file=latest_file_name,
        stage=stage,
    )


def _render_monitor_snapshot(snapshot: MonitorSnapshot, *, stream) -> None:
    total_completed = sum(pair.completed for pair in snapshot.pair_snapshots)
    total_failed = sum(pair.failed for pair in snapshot.pair_snapshots)
    total_active = sum(pair.active for pair in snapshot.pair_snapshots)
    total_pending = sum(pair.pending for pair in snapshot.pair_snapshots)
    total_conditions = sum(pair.total for pair in snapshot.pair_snapshots)

    lines = [
        f"Method 1 monitor: {snapshot.root}",
        "",
        "Pair        Done   Fail  Active Pending Total  Latest",
        "----------  -----  ----  ------ ------- -----  -------------------------------",
    ]
    for pair in snapshot.pair_snapshots:
        latest = _format_latest_condition(pair.latest_condition)
        lines.append(
            f"{pair.pair_name:<10}  {pair.completed:>5}  {pair.failed:>4}  "
            f"{pair.active:>6}  {pair.pending:>7}  {pair.total:>5}  {latest}"
        )

    lines.extend(
        [
            "",
            f"Overall     {total_completed:>5}  {total_failed:>4}  {total_active:>6}  "
            f"{total_pending:>7}  {total_conditions:>5}",
            "",
            "Recent activity:",
        ]
    )

    recent_conditions = _collect_recent_conditions(snapshot.pair_snapshots)
    if recent_conditions:
        for condition in recent_conditions:
            lines.append(f"- {condition}")
    else:
        lines.append("- no condition directories found")

    stream.write("\n".join(lines))
    stream.write("\n")
    stream.flush()


def _collect_recent_conditions(pair_snapshots: Iterable[PairSnapshot]) -> list[str]:
    recent_entries: list[tuple[float, str]] = []
    for pair in pair_snapshots:
        for condition in pair.recent_conditions:
            label = condition.path.relative_to(condition.path.parents[1]).as_posix()
            stage_text = f", stage={condition.stage}" if condition.stage else ""
            file_text = f", file={condition.latest_file}" if condition.latest_file else ""
            recent_entries.append(
                (
                    condition.updated_at,
                    f"{label} [{condition.status}{stage_text}{file_text}]",
                )
            )

    recent_entries.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in recent_entries[:10]]


def _format_latest_condition(condition: ConditionSnapshot | None) -> str:
    if condition is None:
        return "-"
    stage_text = f" stage={condition.stage}" if condition.stage else ""
    file_text = f" file={condition.latest_file}" if condition.latest_file else ""
    return f"{condition.path.name}{stage_text}{file_text}"
