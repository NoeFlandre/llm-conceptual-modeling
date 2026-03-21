import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_conceptual_modeling.post_revision_debug.artifacts import append_jsonl_event


@dataclass(frozen=True)
class ReplaySpec:
    model: str
    algorithm: str
    pair: str
    output_root: Path
    command: list[str]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mistral-small-2603", "mistral-medium-2508"],
    )
    parser.add_argument(
        "--embedding-model",
        default="mistral-embed-2312",
    )
    parser.add_argument(
        "--output-root",
        default="data/analysis_artifacts/post_revision_debug/mistral/2026-03-21/paper_replay",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("Missing required environment variable: MISTRAL_API_KEY")

    output_root = Path(args.output_root)
    run_name = output_root.name
    run_dir = output_root
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": run_name,
        "models": args.models,
        "embedding_model": args.embedding_model,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "replay_specs": [
            asdict(spec)
            | {"output_root": str(spec.output_root), "command": spec.command}
            for spec in _build_replay_plan(
                models=args.models,
                embedding_model=args.embedding_model,
                output_root=output_root,
                resume=args.resume,
            )
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    append_jsonl_event(
        run_dir / "events.jsonl",
        {"event": "run_started", "run_name": run_name, "models": args.models},
    )

    if args.dry_run:
        (run_dir / "dry_run.json").write_text(json.dumps(manifest, indent=2))
        return 0

    state_path = run_dir / "state.json"
    state = _load_state(state_path) if args.resume else _new_state(run_name)
    state["run_name"] = run_name
    state["models"] = args.models
    state["embedding_model"] = args.embedding_model
    state["resume"] = args.resume
    state["updated_at"] = datetime.now(UTC).isoformat()
    _write_state(state_path, state)

    failures: list[dict[str, object]] = []
    for spec in _build_replay_plan(
        models=args.models,
        embedding_model=args.embedding_model,
        output_root=output_root,
        resume=args.resume,
    ):
        job_key = _build_job_key(spec)
        if args.resume and job_key in state["completed_jobs"]:
            append_jsonl_event(
                run_dir / "events.jsonl",
                {
                    "event": "command_skipped",
                    "model": spec.model,
                    "algorithm": spec.algorithm,
                    "pair": spec.pair,
                    "reason": "already_completed",
                },
            )
            continue

        spec.output_root.mkdir(parents=True, exist_ok=True)
        log_path = spec.output_root / f"{spec.algorithm}_{spec.pair}.log"
        append_jsonl_event(
            run_dir / "events.jsonl",
            {
                "event": "command_started",
                "model": spec.model,
                "algorithm": spec.algorithm,
                "pair": spec.pair,
                "command": spec.command,
            },
        )
        with log_path.open("a", encoding="utf-8") as log_file:
            result = subprocess.run(
                spec.command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        append_jsonl_event(
            run_dir / "events.jsonl",
            {
                "event": "command_finished",
                "model": spec.model,
                "algorithm": spec.algorithm,
                "pair": spec.pair,
                "returncode": result.returncode,
            },
        )
        if result.returncode != 0:
            failures.append(
                {
                    "model": spec.model,
                    "algorithm": spec.algorithm,
                    "pair": spec.pair,
                    "returncode": result.returncode,
                }
            )
            state["failed_jobs"].append(
                {
                    "model": spec.model,
                    "algorithm": spec.algorithm,
                    "pair": spec.pair,
                    "returncode": result.returncode,
                }
            )
        else:
            state["completed_jobs"].append(job_key)
        state["updated_at"] = datetime.now(UTC).isoformat()
        _write_state(state_path, state)

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "completed_jobs": state["completed_jobs"],
                "completed_at": datetime.now(UTC).isoformat(),
                "failure_count": len(failures),
                "failures": failures,
            },
            indent=2,
        )
    )
    append_jsonl_event(
        run_dir / "events.jsonl",
        {"event": "run_finished", "run_name": run_name, "failure_count": len(failures)},
    )
    return 0 if not failures else 1


def _build_replay_plan(
    *,
    models: list[str],
    embedding_model: str,
    output_root: Path,
    resume: bool,
) -> list[ReplaySpec]:
    replay_specs: list[ReplaySpec] = []
    algo_pair_map = {
        "algo1": ["sg1_sg2", "sg2_sg3", "sg3_sg1"],
        "algo2": ["sg1_sg2", "sg2_sg3", "sg3_sg1"],
        "algo3": [
            "subgraph_1_to_subgraph_3",
            "subgraph_2_to_subgraph_1",
            "subgraph_2_to_subgraph_3",
        ],
    }

    for model in models:
        model_output_root = output_root / model
        for algorithm, pairs in algo_pair_map.items():
            for pair in pairs:
                command = [
                    "uv",
                    "run",
                    "lcm",
                    "generate",
                    algorithm,
                    "--model",
                    model,
                    "--pair",
                    pair,
                    "--output-root",
                    str(model_output_root),
                ]
                if algorithm == "algo2":
                    command.extend(["--embedding-model", embedding_model])
                if resume:
                    command.append("--resume")
                replay_specs.append(
                    ReplaySpec(
                        model=model,
                        algorithm=algorithm,
                        pair=pair,
                        output_root=model_output_root,
                        command=command,
                    )
                )

    return replay_specs


def _build_job_key(spec: ReplaySpec) -> dict[str, str]:
    return {
        "model": spec.model,
        "algorithm": spec.algorithm,
        "pair": spec.pair,
    }


def _new_state(run_name: str) -> dict[str, object]:
    return {
        "run_name": run_name,
        "completed_jobs": [],
        "failed_jobs": [],
    }


def _load_state(state_path: Path) -> dict[str, object]:
    if not state_path.exists():
        return _new_state(state_path.stem)
    return json.loads(state_path.read_text())


def _write_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
