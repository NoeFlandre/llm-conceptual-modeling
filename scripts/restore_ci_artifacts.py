#!/usr/bin/env python3
"""Restore the small, hash-pinned artifact subset used by integration tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / ".github" / "ci-artifact-pins.json"


class RestoreError(RuntimeError):
    """A pinned artifact could not be listed, downloaded, or verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_output(output: str) -> Any:
    """Parse JSON when the CLI prints its upgrade notice before the payload."""
    starts = [position for position in (output.find("{"), output.find("[")) if position >= 0]
    if not starts:
        raise RestoreError("The hf CLI returned no JSON payload.")
    try:
        value, _ = json.JSONDecoder().raw_decode(output[min(starts) :])
    except json.JSONDecodeError as exc:
        raise RestoreError("The hf CLI returned malformed JSON.") from exc
    return value


def _run_hf(repo_root: Path, args: list[str], hf_home: str) -> str:
    env = os.environ.copy()
    for name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN_PATH",
    ):
        env.pop(name, None)
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["HF_HOME"] = hf_home
    try:
        result = subprocess.run(
            ["uv", "run", "--locked", "--no-sync", "--no-env-file", "hf", *args],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise RestoreError(
            "Could not start uv; install the locked project environment first."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RestoreError(f"hf buckets {' '.join(args)} timed out after 120 seconds.") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "no command output").strip()
        raise RestoreError(
            f"hf buckets {' '.join(args)} failed with exit code {exc.returncode}: {details}"
        ) from exc
    return result.stdout


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RestoreError(f"Could not read artifact pin manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RestoreError("Artifact pin manifest must use schema_version 1.")
    bucket_id = manifest.get("bucket")
    if not isinstance(bucket_id, str) or not bucket_id:
        raise RestoreError("Artifact pin manifest must name a bucket.")
    if not isinstance(manifest.get("expected_hf_cli_version"), str):
        raise RestoreError("Artifact pin manifest must pin the hf CLI version.")
    if manifest.get("global_revision") is not None:
        raise RestoreError("Expected per-file pins because this bucket has no immutable revision.")

    assets = manifest.get("assets")
    expected_count = manifest.get("expected_asset_count")
    expected_bytes = manifest.get("expected_total_bytes")
    if not isinstance(assets, list) or len(assets) != expected_count:
        raise RestoreError(f"Expected exactly {expected_count} pinned integration-test files.")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        raise RestoreError("Artifact pin manifest must include expected_total_bytes.")

    seen_paths: set[str] = set()
    total_bytes = 0
    for asset in assets:
        if not isinstance(asset, dict):
            raise RestoreError("Each pinned asset must be an object.")
        bucket_path = asset.get("bucket_path")
        if not isinstance(bucket_path, str) or not bucket_path:
            raise RestoreError("Each pinned asset must include a bucket_path.")
        relative = PurePosixPath(bucket_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in bucket_path:
            raise RestoreError(f"Unsafe bucket path in artifact pin manifest: {bucket_path}")
        if bucket_path in seen_paths:
            raise RestoreError(f"Duplicate bucket path in artifact pin manifest: {bucket_path}")
        seen_paths.add(bucket_path)

        size_bytes = asset.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise RestoreError(f"Invalid size pin for {bucket_path}.")
        for field in ("xet_hash", "sha256"):
            value = asset.get(field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise RestoreError(f"Invalid {field} pin for {bucket_path}.")
        total_bytes += size_bytes

    if total_bytes != expected_bytes:
        raise RestoreError(
            f"Artifact pin manifest totals {total_bytes} bytes, expected {expected_bytes}."
        )
    return manifest


def _validate_bucket_listing(
    bucket_id: str,
    assets: list[dict[str, Any]],
    listed_by_prefix: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for asset in assets:
        bucket_path = asset["bucket_path"]
        prefix = PurePosixPath(bucket_path).parent.as_posix()
        row = listed_by_prefix[prefix].get(bucket_path)
        if row is None:
            raise RestoreError(f"Pinned artifact is missing from bucket {bucket_id}: {bucket_path}")
        if row.get("size") != asset["size_bytes"] or row.get("xet_hash") != asset["xet_hash"]:
            raise RestoreError(
                f"Bucket pin drift for {bucket_path}: expected size={asset['size_bytes']} "
                f"xet={asset['xet_hash']}; found size={row.get('size')} "
                f"xet={row.get('xet_hash')}."
            )


def _existing_matches(path: Path, asset: dict[str, Any]) -> bool:
    if path.is_symlink():
        raise RestoreError(f"Refusing to follow a symlink at pinned artifact destination: {path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise RestoreError(f"Pinned artifact destination is not a regular file: {path}")
    if path.stat().st_size != asset["size_bytes"] or _sha256(path) != asset["sha256"]:
        raise RestoreError(f"Refusing to replace mismatched local artifact: {path}")
    return True


def restore_artifacts(
    *,
    repo_root: Path,
    destination_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> None:
    manifest = _load_manifest(manifest_path)
    bucket_id = manifest["bucket"]
    assets: list[dict[str, Any]] = manifest["assets"]
    total_bytes = manifest["expected_total_bytes"]
    destination_root.mkdir(parents=True, exist_ok=True)

    runner_temp = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
    runner_temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lcm-hf-anon-", dir=runner_temp) as hf_home:
        version_output = _run_hf(repo_root, ["version"], hf_home)
        versions = re.findall(r"(?m)^\s*(\d+\.\d+\.\d+)\s*$", version_output)
        if not versions or versions[-1] != manifest["expected_hf_cli_version"]:
            actual = versions[-1] if versions else "unrecognized"
            raise RestoreError(
                f"Expected hf CLI {manifest['expected_hf_cli_version']}, found {actual}."
            )
        info = _parse_json_output(_run_hf(repo_root, ["buckets", "info", bucket_id], hf_home))
        if info.get("id") != bucket_id or info.get("private") is not False:
            raise RestoreError(
                f"Expected public bucket {bucket_id}; received incompatible metadata."
            )

        listed_by_prefix: dict[str, dict[str, dict[str, Any]]] = {}
        for asset in assets:
            prefix = PurePosixPath(asset["bucket_path"]).parent.as_posix()
            if prefix in listed_by_prefix:
                continue
            rows = _parse_json_output(
                _run_hf(
                    repo_root,
                    ["buckets", "list", f"{bucket_id}/{prefix}", "--format", "json"],
                    hf_home,
                )
            )
            if not isinstance(rows, list):
                raise RestoreError(f"Bucket listing for {prefix} was not a JSON array.")
            listed_by_prefix[prefix] = {
                row["path"]: row
                for row in rows
                if isinstance(row, dict) and row.get("type") == "file" and "path" in row
            }

        _validate_bucket_listing(bucket_id, assets, listed_by_prefix)

        destinations = {
            asset["bucket_path"]: destination_root / PurePosixPath(asset["bucket_path"])
            for asset in assets
        }
        existing_paths = {
            asset["bucket_path"]
            for asset in assets
            if _existing_matches(destinations[asset["bucket_path"]], asset)
        }
        missing_assets = [asset for asset in assets if asset["bucket_path"] not in existing_paths]
        downloaded_bytes = 0

        with tempfile.TemporaryDirectory(
            prefix=".lcm-ci-artifacts-", dir=destination_root
        ) as stage:
            stage_root = Path(stage)
            staged_paths: dict[str, Path] = {}
            for asset in missing_assets:
                bucket_path = asset["bucket_path"]
                staged_path = stage_root / PurePosixPath(bucket_path)
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                _run_hf(
                    repo_root,
                    [
                        "buckets",
                        "cp",
                        f"hf://buckets/{bucket_id}/{bucket_path}",
                        str(staged_path),
                    ],
                    hf_home,
                )
                if (
                    not staged_path.is_file()
                    or staged_path.stat().st_size != asset["size_bytes"]
                    or _sha256(staged_path) != asset["sha256"]
                ):
                    raise RestoreError(
                        f"Downloaded bytes failed size or SHA-256 verification: {bucket_path}"
                    )
                staged_paths[bucket_path] = staged_path

            for asset in missing_assets:
                bucket_path = asset["bucket_path"]
                destination = destinations[bucket_path]
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_paths[bucket_path], destination)
                downloaded_bytes += asset["size_bytes"]
                print(f"Downloaded and verified {bucket_path}")

    print(
        f"Verified {len(assets)} pinned integration-test artifacts ({total_bytes} bytes total); "
        f"downloaded {downloaded_bytes} bytes. This is not a full research-data restore."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    destination_root = (args.destination_root or repo_root / "data").resolve()
    try:
        restore_artifacts(
            repo_root=repo_root,
            destination_root=destination_root,
            manifest_path=args.manifest.resolve(),
        )
    except (OSError, RestoreError) as exc:
        print(f"CI artifact restore failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
