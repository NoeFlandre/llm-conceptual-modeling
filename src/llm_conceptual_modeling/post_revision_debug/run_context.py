import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llm_conceptual_modeling.post_revision_debug.artifacts import append_jsonl_event


@dataclass
class ProbeRunContext:
    output_dir: Path
    run_name: str
    algorithm: str
    resume: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def state_path(self) -> Path:
        return self.output_dir / "state.json"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "run.log"

    @property
    def events_path(self) -> Path:
        return self.output_dir / "events.jsonl"

    def log(
        self,
        message: str,
        *,
        level: str = "INFO",
        stage: str | None = None,
    ) -> None:
        timestamp = datetime.now(UTC).isoformat()
        stage_text = f" stage={stage}" if stage else ""
        line = (
            f"{timestamp} level={level} run={self.run_name} "
            f"algorithm={self.algorithm}{stage_text} message={message}"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")

    def append_event(self, event: dict[str, object]) -> None:
        append_jsonl_event(self.events_path, event)

    def write_json(self, filename: str, payload: dict[str, object]) -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def write_text(self, filename: str, content: str) -> Path:
        path = self.output_dir / filename
        path.write_text(content)
        return path

    def load_json(self, filename: str) -> dict[str, object] | None:
        path = self.output_dir / filename
        if not path.exists():
            return None

        content = path.read_text()
        parsed_content = json.loads(content)
        if not isinstance(parsed_content, dict):
            msg = f"Expected JSON object in {path}"
            raise ValueError(msg)

        return parsed_content

    def load_state(self) -> dict[str, object]:
        state = self.load_json("state.json")
        if state is not None:
            return state

        return {
            "algorithm": self.algorithm,
            "run_name": self.run_name,
            "completed_stages": [],
        }

    def mark_stage_complete(
        self,
        stage_name: str,
        *,
        details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        state = self.load_state()
        completed_stages = list(state.get("completed_stages", []))  # type: ignore[arg-type]
        if stage_name not in completed_stages:
            completed_stages.append(stage_name)
        state["completed_stages"] = completed_stages
        state["algorithm"] = self.algorithm
        state["run_name"] = self.run_name
        if details is not None:
            state[stage_name] = details
        self.write_json("state.json", state)
        return state

    def is_stage_complete(self, stage_name: str) -> bool:
        state = self.load_state()
        completed_stages = state.get("completed_stages", [])
        if not isinstance(completed_stages, list):
            return False

        return stage_name in completed_stages

    def record_manifest(self, manifest: dict[str, object]) -> None:
        self.write_json("manifest.json", manifest)
        self.mark_stage_complete("manifest_written", details=manifest)
        self.log("manifest written", stage="manifest")

    def record_prompt(self, filename: str, prompt: str, *, stage: str) -> None:
        self.write_text(filename, prompt)
        self.mark_stage_complete(stage, details={"filename": filename})
        self.log(f"prompt written to {filename}", stage=stage)

    def record_checkpoint(self, filename: str, payload: dict[str, object], *, stage: str) -> None:
        self.write_json(filename, payload)
        self.mark_stage_complete(stage, details={"filename": filename})
        self.log(f"checkpoint written to {filename}", stage=stage)

    def record_failure(self, *, error: Exception) -> None:
        error_record: dict[str, object] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        self.write_json("error.json", error_record)
        self.mark_stage_complete("probe_failed", details=error_record)
        self.log(str(error), level="ERROR", stage="probe_failed")
