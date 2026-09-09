#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

GENERAL_FILE_SIZE_LIMIT_BYTES = 1_000_000
ARTIFACT_FILE_SIZE_LIMIT_BYTES = 75_000_000

LARGE_ARTIFACT_PREFIXES = (
    "data/results/",
    "data/analysis_artifacts/",
    "tests/reference_fixtures/",
)

ALLOWED_INPUT_PREFIXES = ("data/inputs/open_weight_map_extension/",)

PRIVATE_PATH_PATTERNS = [
    re.compile(r"(^|/)\.env($|\.)"),
    re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)$"),
    re.compile(r"(^|/)(credentials?|secrets?|tokens?)(\.|/|$)", re.IGNORECASE),
    re.compile(r"\.(pem|p12|pfx|key)$", re.IGNORECASE),
]

SECRET_CONTENT_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{30,}\b"),
]

CONTENT_SCAN_SKIP_PREFIXES = LARGE_ARTIFACT_PREFIXES


def tracked_files(repo_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def file_size_limit(path: str) -> int:
    if path.startswith(LARGE_ARTIFACT_PREFIXES):
        return ARTIFACT_FILE_SIZE_LIMIT_BYTES
    return GENERAL_FILE_SIZE_LIMIT_BYTES


def find_large_file_violations(
    paths: list[str],
    *,
    size_by_path: dict[str, int],
) -> list[str]:
    violations: list[str] = []
    for path in paths:
        size = size_by_path[path]
        limit = file_size_limit(path)
        if size > limit:
            violations.append(f"{path} ({size} bytes > {limit} bytes)")
    return violations


def find_private_or_raw_path_violations(paths: list[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        if any(pattern.search(path) for pattern in PRIVATE_PATH_PATTERNS):
            violations.append(path)
            continue
        if path.startswith("data/inputs/") and not path.startswith(ALLOWED_INPUT_PREFIXES):
            violations.append(path)
            continue
        if path.startswith(("results/", "runs/", "temporary/")):
            violations.append(path)
    return violations


def find_secret_content_violations(repo_root: Path, paths: list[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        if path.startswith(CONTENT_SCAN_SKIP_PREFIXES):
            continue
        absolute_path = repo_root / path
        if not absolute_path.is_file():
            continue
        if absolute_path.stat().st_size > GENERAL_FILE_SIZE_LIMIT_BYTES:
            continue
        try:
            content = absolute_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in SECRET_CONTENT_PATTERNS):
            violations.append(path)
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    paths = tracked_files(repo_root)
    size_by_path = {path: (repo_root / path).stat().st_size for path in paths}

    large_file_violations = find_large_file_violations(paths, size_by_path=size_by_path)
    private_path_violations = find_private_or_raw_path_violations(paths)
    secret_content_violations = find_secret_content_violations(repo_root, paths)

    if not large_file_violations and not private_path_violations and not secret_content_violations:
        print("Repository guardrails passed.")
        return 0

    if large_file_violations:
        print("Unexpected large tracked files:", file=sys.stderr)
        for violation in large_file_violations:
            print(f"- {violation}", file=sys.stderr)
    if private_path_violations:
        print("Private-looking or raw input paths are tracked:", file=sys.stderr)
        for violation in private_path_violations:
            print(f"- {violation}", file=sys.stderr)
    if secret_content_violations:
        print("High-confidence secret patterns were found:", file=sys.stderr)
        for violation in secret_content_violations:
            print(f"- {violation}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
