from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from llm_conceptual_modeling import osf_packaging
from llm_conceptual_modeling.osf_packaging import (
    EXPECTED_ZIP_NAMES,
    PackageWriteResult,
    _filter_csv,
    _filter_json_payload,
    _is_excluded_path,
    _record_model,
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


def test_package_readme_links_source_of_truth_repositories(tmp_path: Path) -> None:
    data_root = _write_fixture_data_tree(tmp_path)
    output_dir = tmp_path / "package"

    write_osf_package(data_root=data_root, output_dir=output_dir)

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/NoeFlandre/llm-conceptual-modeling" in readme
    assert "https://huggingface.co/NoeFlandre/llm-variability-conceptual-modeling" in readme
    assert "source of truth" in readme


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


@pytest.mark.parametrize(
    ("text", "full_model_name", "model_label", "model_column_prefix", "expected"),
    [
        ("", "Qwen/Qwen3.5-9B", "Qwen", "qwen", ""),
        (
            "model,status\nQwen/Qwen3.5-9B,finished\nMistral,failed\n",
            "Qwen/Qwen3.5-9B",
            "Qwen",
            "qwen",
            "model,status\r\nQwen/Qwen3.5-9B,finished\r\n",
        ),
        (
            "Model,status\nQwen,finished\nMistral,failed\n",
            "Qwen/Qwen3.5-9B",
            "Qwen",
            "qwen",
            "Model,status\r\nQwen,finished\r\n",
        ),
        (
            "qwen_mean,mistral_mean,common\n1,2,3\n",
            "Qwen/Qwen3.5-9B",
            "Qwen",
            "qwen",
            "qwen_mean,common\r\n1,3\r\n",
        ),
    ],
)
def test_filter_csv_keeps_selected_model_rows_and_columns(
    text: str,
    full_model_name: str,
    model_label: str,
    model_column_prefix: str,
    expected: str,
) -> None:
    assert _filter_csv(
        text,
        full_model_name=full_model_name,
        model_label=model_label,
        model_column_prefix=model_column_prefix,
    ) == expected


def test_filter_csv_returns_none_when_model_filter_removes_all_rows() -> None:
    assert (
        _filter_csv(
            "model,status\nMistral,failed\n",
            full_model_name="Qwen/Qwen3.5-9B",
            model_label="Qwen",
            model_column_prefix="qwen",
        )
        is None
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "ok"}, {"status": "ok"}),
        (
            {
                "records": [
                    {"identity": {"model": "Qwen/Qwen3.5-9B"}},
                    {"identity": {"model": "Mistral"}},
                ],
                "status": "ok",
            },
            {
                "records": [{"identity": {"model": "Qwen/Qwen3.5-9B"}}],
                "status": "ok",
            },
        ),
    ],
)
def test_filter_json_payload_keeps_only_selected_model_records(
    payload: object,
    expected: object,
) -> None:
    assert _filter_json_payload(payload, full_model_name="Qwen/Qwen3.5-9B") == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (Path(".DS_Store"), True),
        (Path("run.log"), True),
        (Path("results-sync-status.json"), True),
        (Path("archives") / "old.csv", True),
        (Path("results") / "kept.csv", False),
    ],
)
def test_is_excluded_path_applies_packaging_exclusions(path: Path, expected: bool) -> None:
    assert _is_excluded_path(path) is expected


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ("not a mapping", None),
        ({"identity": {"model": "Qwen/Qwen3.5-9B"}}, "Qwen/Qwen3.5-9B"),
        ({"model": "Qwen/Qwen3.5-9B"}, "Qwen/Qwen3.5-9B"),
        ({"status": "finished"}, None),
    ],
)
def test_record_model_reads_identity_then_top_level_model(
    record: object,
    expected: str | None,
) -> None:
    assert _record_model(record) == expected


def test_osf_packaging_main_prints_write_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    result = PackageWriteResult(
        output_dir=tmp_path / "package",
        readme_path=tmp_path / "package" / "README.md",
        checksum_path=tmp_path / "package" / "checksums.txt",
        zip_paths=[tmp_path / "package" / "inputs.zip"],
    )

    monkeypatch.setattr(osf_packaging, "write_osf_package", lambda **_: result)

    assert osf_packaging.main(["--data-root", str(tmp_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Output directory:" in output
    assert "inputs.zip" in output


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
