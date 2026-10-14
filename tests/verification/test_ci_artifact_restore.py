import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESTORE_SCRIPT = REPO_ROOT / "scripts" / "restore_ci_artifacts.py"
PIN_MANIFEST = REPO_ROOT / ".github" / "ci-artifact-pins.json"
EXPECTED_BUCKET_PATHS = {
    "inputs/Giabbanelli & Macewan (categories).csv",
    "inputs/Giabbanelli & Macewan (edges).csv",
    "inputs/algo2_thesaurus.json",
    "inputs/wordnet_label_lexicon.json",
    "results/open_weights/hf-paper-batch-canonical/runs/algo1/Qwen__Qwen3.5-9B/beam_num_beams_2/sg1_sg2/00000/rep_00/state.json",
}

FAKE_UV = """#!/usr/bin/env python3
import base64
import json
import os
import sys
from pathlib import Path

state = json.loads(Path(os.environ["FAKE_HF_STATE"]).read_text())
if any(
    name in os.environ
    for name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN_PATH",
    )
):
    print("credential env leaked to hf command", file=sys.stderr)
    raise SystemExit(3)
if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1" or not os.environ.get("HF_HOME"):
    print("anonymous hf environment was not configured", file=sys.stderr)
    raise SystemExit(4)

args = sys.argv[1:]
if args[:5] != ["run", "--locked", "--no-sync", "--no-env-file", "hf"]:
    print("unexpected uv invocation", file=sys.stderr)
    raise SystemExit(5)
command = args[5:]
bucket = state["bucket"]
assets = state["assets"]

if command == ["version"]:
    print(state.get("hf_cli_version", "1.8.0"))
    raise SystemExit(0)

if command[:2] == ["buckets", "info"]:
    payload = {"id": bucket, "private": False}
elif command[:2] == ["buckets", "list"]:
    prefix = command[2].removeprefix(bucket + "/")
    payload = []
    for asset in assets:
        path = asset["bucket_path"]
        if Path(path).parent.as_posix() == prefix and path != state.get("missing_path"):
            row = {
                "type": "file",
                "path": path,
                "size": asset["size_bytes"],
                "xet_hash": asset["xet_hash"],
            }
            if path == state.get("xet_drift_path"):
                row["xet_hash"] = state.get("xet_override", "f" * 64)
            if path == state.get("size_drift_path"):
                row["size"] = asset["size_bytes"] + 1
            payload.append(row)
else:
    payload = None

if payload is not None:
    print("A new version of huggingface_hub is available")
    print(json.dumps(payload))
    raise SystemExit(0)

if command[:2] == ["buckets", "cp"]:
    source = command[2].removeprefix("hf://buckets/" + bucket + "/")
    asset = next(item for item in assets if item["bucket_path"] == source)
    log = Path(state["copy_log"])
    with log.open("a", encoding="utf-8") as stream:
        stream.write(source + "\\n")
    data = base64.b64decode(asset["content_b64"])
    if source == state.get("corrupt_download_path"):
        data = b"corrupt bytes"
    destination = Path(command[3])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    raise SystemExit(0)

print("unexpected hf command", file=sys.stderr)
raise SystemExit(6)
"""


def test_committed_manifest_is_the_five_file_integration_subset() -> None:
    manifest = json.loads(PIN_MANIFEST.read_text(encoding="utf-8"))
    assets = manifest["assets"]

    assert manifest["bucket"] == "NoeFlandre/llm-variability-conceptual-modeling"
    assert manifest["global_revision"] is None
    assert manifest["expected_hf_cli_version"] == "1.8.0"
    assert "not a complete research corpus" in manifest["purpose"]
    assert {asset["bucket_path"] for asset in assets} == EXPECTED_BUCKET_PATHS
    assert manifest["expected_asset_count"] == len(assets) == 5
    assert (
        manifest["expected_total_bytes"] == sum(asset["size_bytes"] for asset in assets) == 42_943
    )
    assert sum(asset["bucket_path"].startswith("inputs/") for asset in assets) == 4
    assert sum("/runs/" in asset["bucket_path"] for asset in assets) == 1


def _case(
    tmp_path: Path,
    *,
    missing_path: str | None = None,
    xet_drift_path: str | None = None,
    size_drift_path: str | None = None,
    corrupt_download_path: str | None = None,
    hf_cli_version: str = "1.8.0",
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    destination_root = repo_root / "data"
    asset_bytes = b"synthetic fixture bytes"
    bucket_path = "inputs/example.json"
    asset = {
        "bucket_path": bucket_path,
        "size_bytes": len(asset_bytes),
        "xet_hash": "a" * 64,
        "sha256": hashlib.sha256(asset_bytes).hexdigest(),
    }
    manifest = {
        "schema_version": 1,
        "bucket": "example/public-bucket",
        "global_revision": None,
        "expected_hf_cli_version": "1.8.0",
        "expected_asset_count": 1,
        "expected_total_bytes": len(asset_bytes),
        "assets": [asset],
    }
    manifest_path = tmp_path / "pins.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    copy_log = tmp_path / "copy.log"
    fake_state = {
        "bucket": manifest["bucket"],
        "assets": [{**asset, "content_b64": base64.b64encode(asset_bytes).decode("ascii")}],
        "copy_log": str(copy_log),
        "missing_path": missing_path,
        "xet_drift_path": xet_drift_path,
        "size_drift_path": size_drift_path,
        "corrupt_download_path": corrupt_download_path,
        "hf_cli_version": hf_cli_version,
    }
    state_path = tmp_path / "fake-hf.json"
    state_path.write_text(json.dumps(fake_state), encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_path = fake_bin / "uv"
    uv_path.write_text(FAKE_UV, encoding="utf-8")
    uv_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_HF_STATE"] = str(state_path)
    env["HF_TOKEN"] = "must-not-reach-hf"
    env["HUGGINGFACE_HUB_TOKEN"] = "must-not-reach-hf"
    env["HUGGING_FACE_HUB_TOKEN"] = "must-not-reach-hf"
    env["HF_TOKEN_PATH"] = str(tmp_path / "token")
    env["RUNNER_TEMP"] = str(tmp_path / "runner-temp")
    result = _invoke(repo_root, manifest_path, destination_root, env)
    return result, destination_root / bucket_path, copy_log


def _invoke(
    repo_root: Path,
    manifest_path: Path,
    destination_root: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RESTORE_SCRIPT),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest_path),
            "--destination-root",
            str(destination_root),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("drift", ["xet", "size"])
def test_restore_fails_before_copy_when_bucket_pin_drifts(tmp_path: Path, drift: str) -> None:
    bucket_path = "inputs/example.json"
    result, destination, copy_log = _case(
        tmp_path,
        xet_drift_path=bucket_path if drift == "xet" else None,
        size_drift_path=bucket_path if drift == "size" else None,
    )

    assert result.returncode == 1
    assert f"Bucket pin drift for {bucket_path}" in result.stderr
    assert not destination.exists()
    assert not copy_log.exists()


def test_restore_fails_if_a_pinned_artifact_is_missing_from_listing(tmp_path: Path) -> None:
    result, destination, copy_log = _case(tmp_path, missing_path="inputs/example.json")

    assert result.returncode == 1
    assert "Pinned artifact is missing from bucket" in result.stderr
    assert not destination.exists()
    assert not copy_log.exists()


def test_restore_fails_if_locked_hf_cli_version_changes(tmp_path: Path) -> None:
    result, destination, copy_log = _case(tmp_path, hf_cli_version="1.9.0")

    assert result.returncode == 1
    assert "Expected hf CLI 1.8.0, found 1.9.0" in result.stderr
    assert not destination.exists()
    assert not copy_log.exists()


def test_restore_removes_download_when_sha256_does_not_match(tmp_path: Path) -> None:
    result, destination, copy_log = _case(tmp_path, corrupt_download_path="inputs/example.json")

    assert result.returncode == 1
    assert "failed size or SHA-256 verification" in result.stderr
    assert not destination.exists()
    assert copy_log.read_text(encoding="utf-8").strip() == "inputs/example.json"
    assert not list(destination.parents[1].glob(".lcm-ci-artifacts-*"))


def test_restore_is_idempotent_and_does_not_redownload_matching_file(tmp_path: Path) -> None:
    first, destination, copy_log = _case(tmp_path)
    repo_root = tmp_path / "repo"
    manifest_path = tmp_path / "pins.json"
    destination_root = repo_root / "data"
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_HF_STATE"] = str(tmp_path / "fake-hf.json")
    env["HF_TOKEN"] = "must-not-reach-hf"
    env["HUGGINGFACE_HUB_TOKEN"] = "must-not-reach-hf"
    env["HUGGING_FACE_HUB_TOKEN"] = "must-not-reach-hf"
    env["HF_TOKEN_PATH"] = str(tmp_path / "token")
    env["RUNNER_TEMP"] = str(tmp_path / "runner-temp")
    second = _invoke(repo_root, manifest_path, destination_root, env)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert destination.read_bytes() == b"synthetic fixture bytes"
    assert copy_log.read_text(encoding="utf-8").splitlines() == ["inputs/example.json"]
    assert "downloaded 0 bytes" in second.stdout


def test_restore_refuses_to_overwrite_mismatched_existing_file(tmp_path: Path) -> None:
    first, destination, copy_log = _case(tmp_path)
    assert first.returncode == 0, first.stderr
    destination.write_bytes(b"changed local bytes")

    repo_root = tmp_path / "repo"
    manifest_path = tmp_path / "pins.json"
    destination_root = repo_root / "data"
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_HF_STATE"] = str(tmp_path / "fake-hf.json")
    env["HF_TOKEN"] = "must-not-reach-hf"
    env["HUGGINGFACE_HUB_TOKEN"] = "must-not-reach-hf"
    env["HUGGING_FACE_HUB_TOKEN"] = "must-not-reach-hf"
    env["HF_TOKEN_PATH"] = str(tmp_path / "token")
    env["RUNNER_TEMP"] = str(tmp_path / "runner-temp")
    retry = _invoke(repo_root, manifest_path, destination_root, env)

    assert retry.returncode == 1
    assert "Refusing to replace mismatched local artifact" in retry.stderr
    assert destination.read_bytes() == b"changed local bytes"
    assert copy_log.read_text(encoding="utf-8").splitlines() == ["inputs/example.json"]
