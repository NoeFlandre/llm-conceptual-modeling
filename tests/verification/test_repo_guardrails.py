import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_repo_guardrails.py"


def _load_guardrails_module():
    spec = importlib.util.spec_from_file_location("check_repo_guardrails", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_file_guard_allows_known_artifact_roots(tmp_path: Path) -> None:
    guardrails = _load_guardrails_module()
    tracked_paths = [
        "README.md",
        "data/results/open_weights/hf-paper-batch-canonical/ledger.json",
    ]
    size_by_path = {
        "README.md": 128,
        "data/results/open_weights/hf-paper-batch-canonical/ledger.json": 60_000_000,
    }

    violations = guardrails.find_large_file_violations(
        tracked_paths,
        size_by_path=size_by_path,
    )

    assert violations == []


def test_large_file_guard_blocks_unexpected_large_files() -> None:
    guardrails = _load_guardrails_module()
    tracked_paths = ["docs/huge-local-export.csv"]
    size_by_path = {"docs/huge-local-export.csv": 2_000_000}

    violations = guardrails.find_large_file_violations(
        tracked_paths,
        size_by_path=size_by_path,
    )

    assert violations == ["docs/huge-local-export.csv (2000000 bytes > 1000000 bytes)"]


def test_private_filename_guard_blocks_credential_shaped_paths() -> None:
    guardrails = _load_guardrails_module()
    tracked_paths = [
        "README.md",
        ".env",
        "secrets/model.key",
        "data/inputs/raw_private_dump.csv",
    ]

    violations = guardrails.find_private_or_raw_path_violations(tracked_paths)

    assert violations == [
        ".env",
        "secrets/model.key",
        "data/inputs/raw_private_dump.csv",
    ]


def test_secret_content_guard_blocks_high_confidence_tokens(tmp_path: Path) -> None:
    guardrails = _load_guardrails_module()
    path = tmp_path / "notes.txt"
    path.write_text("OPENAI_API_KEY=sk-" + ("a" * 30), encoding="utf-8")

    violations = guardrails.find_secret_content_violations(tmp_path, ["notes.txt"])

    assert violations == ["notes.txt"]
