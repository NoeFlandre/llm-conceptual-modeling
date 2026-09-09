from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _target_body(makefile_text: str, target: str) -> str:
    marker = f"{target}:"
    start = makefile_text.index(marker)
    next_target = makefile_text.find("\n\n", start)
    if next_target == -1:
        return makefile_text[start:]
    return makefile_text[start:next_target]


def test_github_actions_runs_the_local_ci_gate() -> None:
    workflow_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "uv sync --locked --dev" in workflow_text
    assert "make ci" in workflow_text


def test_make_ci_collects_fast_quality_gates() -> None:
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    ci_target = _target_body(makefile_text, "ci")

    expected_targets = [
        "lock-check",
        "lint",
        "guardrails",
        "cli-smoke",
        "test",
        "verify",
    ]

    assert "format-check:" in makefile_text
    assert "typecheck:" in makefile_text
    for target in expected_targets:
        assert f"{target}:" in makefile_text
        assert f"$(MAKE) {target}" in ci_target
