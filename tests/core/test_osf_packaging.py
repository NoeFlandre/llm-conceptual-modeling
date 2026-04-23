from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from llm_conceptual_modeling.osf_packaging import (
    EXPECTED_ZIP_NAMES,
    build_osf_manifest,
    write_osf_package,
)


def test_osf_manifest_uses_exact_reviewer_zip_names(tmp_path: Path) -> None:
    data_root = _write_fixture_data_tree(tmp_path)

    manifest = build_osf_manifest(data_root)

    assert [archive.name for archive in manifest] == list(EXPECTED_ZIP_NAMES)


def test_osf_manifest_excludes_archives_garbage_internal_names_and_olmo(tmp_path: Path) -> None:
    data_root = _write_fixture_data_tree(tmp_path)

    manifest = build_osf_manifest(data_root)
    archive_paths = [str(entry.archive_path) for archive in manifest for entry in archive.entries]
    joined_paths = "\n".join(archive_paths)

    assert "README.md" in {path for path in archive_paths if path == "README.md"}
    assert "archives" not in joined_paths
    assert "stale-shards" not in joined_paths
    assert "worker-queues" not in joined_paths
    assert "preview" not in joined_paths
    assert ".DS_Store" not in joined_paths
    assert ".log" not in joined_paths
    assert ".pid" not in joined_paths
    assert "results-sync" not in joined_paths
    assert "Olmo" not in joined_paths
    assert "olmo" not in joined_paths
    assert "hf-paper-batch-canonical" not in joined_paths
    assert "hf-map-extension-canonical" not in joined_paths


def test_osf_manifest_groups_open_weight_archives_by_single_llm(tmp_path: Path) -> None:
    data_root = _write_fixture_data_tree(tmp_path)

    archives = {archive.name: archive for archive in build_osf_manifest(data_root)}

    qwen_paths = "\n".join(
        str(entry.archive_path) for entry in archives["results_open_weight_sweep_qwen.zip"].entries
    )
    mistral_paths = "\n".join(
        str(entry.archive_path)
        for entry in archives["results_open_weight_sweep_mistral.zip"].entries
    )

    assert "results/open_weight_sweep/qwen/algo1/" in qwen_paths
    assert "results/open_weight_sweep/mistral/algo1/" not in qwen_paths
    assert "results/open_weight_sweep/mistral/algo1/" in mistral_paths
    assert "results/open_weight_sweep/qwen/algo1/" not in mistral_paths


def test_write_osf_package_creates_readmes_checksums_and_zip_contents(tmp_path: Path) -> None:
    data_root = _write_fixture_data_tree(tmp_path)
    output_dir = tmp_path / "package"

    written = write_osf_package(data_root=data_root, output_dir=output_dir)

    assert [path.name for path in written.zip_paths] == list(EXPECTED_ZIP_NAMES)
    assert (output_dir / "README.md").exists()
    assert (output_dir / "checksums.txt").exists()
    for zip_path in written.zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
        assert "README.md" in names
        assert len(names) == len(set(names))
        assert all("hf-paper-batch-canonical" not in name for name in names)
        assert all("hf-map-extension-canonical" not in name for name in names)


def test_write_osf_package_dry_run_reports_paths_without_writing(tmp_path: Path) -> None:
    data_root = _write_fixture_data_tree(tmp_path)
    output_dir = tmp_path / "package"

    written = write_osf_package(data_root=data_root, output_dir=output_dir, dry_run=True)

    assert [path.name for path in written.zip_paths] == list(EXPECTED_ZIP_NAMES)
    assert written.readme_path == output_dir / "README.md"
    assert written.checksum_path == output_dir / "checksums.txt"
    assert not output_dir.exists()


def test_write_osf_package_rejects_duplicate_archive_paths(tmp_path: Path) -> None:
    data_root = _write_fixture_data_tree(tmp_path)
    manifest = build_osf_manifest(data_root)
    first_archive = manifest[0]
    duplicate_entry = first_archive.entries[0]
    first_archive.entries.append(duplicate_entry)

    with pytest.raises(ValueError, match="Duplicate archive path"):
        write_osf_package(
            data_root=data_root,
            output_dir=tmp_path / "package",
            manifest=manifest,
        )


def _write_fixture_data_tree(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    _write(data_root / "inputs" / "Giabbanelli & Macewan (edges).csv", "a,b\n")
    _write(data_root / "inputs" / ".DS_Store", "garbage\n")
    _write(
        data_root / "analysis_artifacts" / "revision_tracker" / "summary.csv",
        "metric,value\nrecall,0.1\n",
    )
    _write(data_root / "results" / "archives" / "stale-shards" / "old.csv", "garbage\n")

    _write(
        data_root / "results" / "frontier" / "algo1" / "gpt-5" / "raw" / "run.csv",
        "frontier\n",
    )
    _write(
        data_root
        / "results"
        / "frontier"
        / "algo1"
        / "openai-gpt-4o"
        / "evaluated"
        / "metrics.csv",
        "frontier\n",
    )
    _write(
        data_root / "results" / "frontier" / "algo1" / "gpt-5" / "run.log",
        "garbage\n",
    )

    sweep_root = data_root / "results" / "open_weights" / "hf-paper-batch-canonical"
    _write(sweep_root / "worker-queues" / "queue.json", "garbage\n")
    _write(sweep_root / "results-sync-status.json", "{}\n")
    _write(sweep_root / "run.log", "garbage\n")
    _write(sweep_root / "batch_summary.csv", "model,status\nQwen/Qwen3.5-9B,finished\n")
    _write(
        sweep_root / "ledger.json",
        '{"records":[{"identity":{"model":"Qwen/Qwen3.5-9B"},"status":"finished"}]}\n',
    )
    _write(
        sweep_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
        / "summary.json",
        "{}\n",
    )
    _write(
        sweep_root
        / "runs"
        / "algo1"
        / "mistralai__Ministral-3-8B-Instruct-2512"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
        / "summary.json",
        "{}\n",
    )
    _write(
        sweep_root / "runs" / "algo1" / "allenai__Olmo-3-7B-Instruct" / "greedy" / "x.json",
        "garbage\n",
    )

    map_root = data_root / "results" / "open_weights" / "hf-map-extension-canonical"
    _write(map_root / "batch_summary.csv", "model,status\nQwen/Qwen3.5-9B,finished\n")
    _write(
        map_root
        / "runs"
        / "algo3"
        / "Qwen__Qwen3.5-9B"
        / "beam_num_beams_6"
        / "babs_johnson"
        / "subgraph_1_to_subgraph_2"
        / "000"
        / "rep_00"
        / "summary.json",
        "{}\n",
    )
    return data_root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
