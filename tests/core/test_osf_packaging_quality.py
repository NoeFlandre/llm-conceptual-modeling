from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast

import pytest

from llm_conceptual_modeling import osf_packaging
from llm_conceptual_modeling.osf_packaging import (
    EXPECTED_ZIP_NAMES,
    PackageArchive,
    PackageEntry,
    PackageWriteResult,
    _add_filtered_csv,
    _add_filtered_json,
    _add_filtered_report_entries,
    _add_open_weight_aggregated_entries,
    _add_open_weight_run_entries,
    _analysis_archive,
    _checksums,
    _drop_other_model_columns,
    _filter_csv,
    _frontier_archive,
    _open_weight_archive,
    _package_readme,
    _package_write_result,
    _prepend_archive_readme,
    _read_source_entry,
    _render_csv,
    _shared_inputs_archive,
    _tree_entries,
    _write_package_artifacts,
    _write_zip,
    build_osf_manifest,
    write_osf_package,
)


def test_build_osf_manifest_forwards_all_archive_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_calls: list[Path] = []
    analysis_calls: list[Path] = []
    frontier_calls: list[tuple[Path, str, str, tuple[str, ...]]] = []
    open_weight_calls: list[dict[str, object]] = []
    readme_calls: list[PackageArchive] = []
    unique_calls: list[PackageArchive] = []

    monkeypatch.setattr(
        osf_packaging,
        "_shared_inputs_archive",
        lambda data_root: shared_calls.append(data_root) or PackageArchive("inputs.zip", "inputs"),
    )
    monkeypatch.setattr(
        osf_packaging,
        "_analysis_archive",
        lambda data_root: (
            analysis_calls.append(data_root)
            or PackageArchive("analysis_revision_tracker.zip", "analysis")
        ),
    )

    def frontier(
        data_root: Path,
        zip_name: str,
        model_name: str,
        aliases: tuple[str, ...],
    ) -> PackageArchive:
        frontier_calls.append((data_root, zip_name, model_name, aliases))
        return PackageArchive(zip_name, model_name)

    def open_weight(**kwargs: object) -> PackageArchive:
        open_weight_calls.append(kwargs)
        return PackageArchive(str(kwargs["zip_name"]), str(kwargs["study_name"]))

    monkeypatch.setattr(osf_packaging, "_frontier_archive", frontier)
    monkeypatch.setattr(osf_packaging, "_open_weight_archive", open_weight)
    monkeypatch.setattr(
        osf_packaging,
        "_prepend_archive_readme",
        lambda archive: readme_calls.append(archive),
    )
    monkeypatch.setattr(
        osf_packaging,
        "_assert_unique_archive_paths",
        lambda archive: unique_calls.append(archive),
    )

    archives = build_osf_manifest(tmp_path)

    assert shared_calls == [tmp_path]
    assert analysis_calls == [tmp_path]
    assert frontier_calls == [
        (tmp_path, zip_name, model_name, aliases)
        for zip_name, model_name, aliases in osf_packaging._FRONTIER_MODELS
    ]
    expected_open_weight_calls = [
        {
            "data_root": tmp_path,
            "zip_name": f"results_open_weight_sweep_{llm_slug}.zip",
            "study_name": "open_weight_sweep",
            "source_root": tmp_path / "results" / "open_weights" / "hf-paper-batch-canonical",
            "llm_slug": llm_slug,
            "source_model_dir": source_model_dir,
            "full_model_name": full_model_name,
            "model_label": model_label,
            "model_column_prefix": model_column_prefix,
        }
        for (
            llm_slug,
            source_model_dir,
            full_model_name,
            model_label,
            model_column_prefix,
        ) in osf_packaging._OPEN_WEIGHT_MODELS
    ] + [
        {
            "data_root": tmp_path,
            "zip_name": f"results_open_weight_map_extension_{llm_slug}.zip",
            "study_name": "open_weight_map_extension",
            "source_root": tmp_path / "results" / "open_weights" / "hf-map-extension-canonical",
            "llm_slug": llm_slug,
            "source_model_dir": source_model_dir,
            "full_model_name": full_model_name,
            "model_label": model_label,
            "model_column_prefix": model_column_prefix,
        }
        for (
            llm_slug,
            source_model_dir,
            full_model_name,
            model_label,
            model_column_prefix,
        ) in osf_packaging._OPEN_WEIGHT_MODELS
    ]
    assert open_weight_calls == expected_open_weight_calls
    assert [archive.name for archive in archives] == list(EXPECTED_ZIP_NAMES)
    assert readme_calls == archives
    assert unique_calls == archives


def test_write_osf_package_returns_exact_paths_for_dry_and_real_writes(
    tmp_path: Path,
) -> None:
    manifest = [
        PackageArchive(
            "one.zip",
            "one",
            [PackageEntry(PurePosixPath("one.txt"), data=b"one")],
        )
    ]
    output_dir = tmp_path / "package"

    dry_result = write_osf_package(
        data_root=tmp_path,
        output_dir=output_dir,
        manifest=manifest,
        dry_run=True,
    )
    expected = PackageWriteResult(
        output_dir=output_dir,
        readme_path=output_dir / "README.md",
        checksum_path=output_dir / "checksums.txt",
        zip_paths=[output_dir / "one.zip"],
    )
    assert dry_result == expected
    assert not output_dir.exists()

    written_result = write_osf_package(
        data_root=tmp_path,
        output_dir=output_dir,
        manifest=manifest,
    )
    assert written_result == expected


def test_write_package_artifacts_creates_nested_utf8_artifacts_and_checksums(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "nested" / "package"
    readme_path = output_dir / "README.md"
    checksum_path = output_dir / "checksums.txt"
    zip_path = output_dir / "one.zip"
    archive = PackageArchive(
        "one.zip",
        "one",
        [PackageEntry(PurePosixPath("payload.txt"), data="café".encode())],
    )

    _write_package_artifacts(
        output_dir=output_dir,
        readme_path=readme_path,
        checksum_path=checksum_path,
        readme_text="# Café\n",
        archives=[archive],
        zip_paths=[zip_path],
    )

    assert readme_path.read_bytes() == "# Café\n".encode()
    assert checksum_path.read_text() == (
        f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  one.zip\n"
    )


def test_write_package_artifacts_honors_existing_directory_and_strict_zip_lengths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    rmtree_calls: list[object] = []
    monkeypatch.setattr(
        osf_packaging.shutil,
        "rmtree",
        lambda path: rmtree_calls.append(path),
    )
    archive = PackageArchive(
        "one.zip",
        "one",
        [PackageEntry(PurePosixPath("one.txt"), data=b"one")],
    )

    _write_package_artifacts(
        output_dir=existing,
        readme_path=existing / "README.md",
        checksum_path=existing / "checksums.txt",
        readme_text="readme\n",
        archives=[archive],
        zip_paths=[existing / "one.zip"],
    )
    assert rmtree_calls == [existing]

    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        _write_package_artifacts(
            output_dir=tmp_path / "strict",
            readme_path=tmp_path / "strict" / "README.md",
            checksum_path=tmp_path / "strict" / "checksums.txt",
            readme_text="readme\n",
            archives=[archive],
            zip_paths=[tmp_path / "strict" / "one.zip", tmp_path / "strict" / "two.zip"],
        )


def test_main_converts_cli_arguments_for_write_and_prints_all_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = PackageWriteResult(
        output_dir=tmp_path / "out",
        readme_path=tmp_path / "out" / "README.md",
        checksum_path=tmp_path / "out" / "checksums.txt",
        zip_paths=[tmp_path / "out" / "one.zip"],
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        osf_packaging,
        "write_osf_package",
        lambda **kwargs: calls.append(kwargs) or result,
    )

    assert osf_packaging.main(["--data-root", "input", "--output-dir", "output", "--dry-run"]) == 0

    assert calls == [
        {
            "data_root": Path("input"),
            "output_dir": Path("output"),
            "dry_run": True,
        }
    ]
    output = capsys.readouterr().out
    assert "Output directory: " in output
    assert "Package README: " in output
    assert "Checksums: " in output
    assert "one.zip" in output


def test_main_defaults_are_path_objects_and_help_describes_the_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        osf_packaging,
        "write_osf_package",
        lambda **kwargs: (
            calls.append(kwargs)
            or PackageWriteResult(Path("dist"), Path("README.md"), Path("checksums.txt"), [])
        ),
    )

    assert osf_packaging.main([]) == 0
    assert calls == [
        {
            "data_root": Path("data"),
            "output_dir": Path("dist/osf_plos_package"),
            "dry_run": False,
        }
    ]
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        osf_packaging.main(["--help"])
    assert error.value.code == 0
    help_output = capsys.readouterr().out
    assert "Build OSF reviewer ZIP artifacts." in help_output
    assert "XX" not in help_output


def test_shared_and_analysis_archives_use_expected_roots_and_descriptions(
    tmp_path: Path,
) -> None:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "input.txt").write_text("input")
    tracker = tmp_path / "analysis_artifacts" / "revision_tracker"
    tracker.mkdir(parents=True)
    (tracker / "summary.csv").write_text("summary")

    inputs = _shared_inputs_archive(tmp_path)
    analysis = _analysis_archive(tmp_path)

    assert inputs.description == "Shared input causal maps, thesaurus, and lexicon files."
    assert inputs.entries[0].archive_path == PurePosixPath("inputs/input.txt")
    assert analysis.description == "Revision-tracker analysis artifacts used for paper reporting."
    assert analysis.entries[0].archive_path == PurePosixPath(
        "analysis/revision_tracker/summary.csv"
    )


def test_shared_and_analysis_archive_path_components_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingPath:
        def __init__(self, calls: list[str]) -> None:
            self.calls = calls

        def __truediv__(self, part: str) -> RecordingPath:
            self.calls.append(part)
            return self

    monkeypatch.setattr(osf_packaging, "_add_tree_entries", lambda *_args, **_kwargs: None)
    shared_calls: list[str] = []
    analysis_calls: list[str] = []

    _shared_inputs_archive(RecordingPath(shared_calls))  # type: ignore[arg-type]
    _analysis_archive(RecordingPath(analysis_calls))  # type: ignore[arg-type]

    assert shared_calls == ["inputs"]
    assert analysis_calls == ["analysis_artifacts", "revision_tracker"]


def test_frontier_archive_collects_all_aliases_and_skips_non_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontier = tmp_path / "results" / "frontier" / "algo1"
    for alias in ("first", "second"):
        path = frontier / alias / f"{alias}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(alias)
    (tmp_path / "results" / "frontier" / "algo-file").write_text("not a directory")

    archive = _frontier_archive(
        tmp_path,
        "frontier.zip",
        "model",
        ("missing", "first", "second"),
    )

    assert archive.description == "Frontier-model results for model."
    assert [str(entry.archive_path) for entry in archive.entries] == [
        "results/frontier/model/algo1/first.csv",
        "results/frontier/model/algo1/second.csv",
    ]

    class FakeSource:
        def exists(self) -> bool:
            return True

    class FakeAlgorithm:
        name = "algo-file"
        parent = SimpleNamespace(name="frontier")

        def is_dir(self) -> bool:
            return False

        def __truediv__(self, _part: str) -> FakeSource:
            return FakeSource()

    added: list[dict[str, object]] = []
    monkeypatch.setattr(Path, "glob", lambda _path, _pattern: [FakeAlgorithm()])
    monkeypatch.setattr(
        osf_packaging,
        "_add_tree_entries",
        lambda **kwargs: added.append(kwargs),
    )
    _frontier_archive(tmp_path, "frontier.zip", "model", ("first",))
    assert added == []


def test_frontier_archive_uses_exact_root_path_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingPath:
        def __init__(self, calls: list[str]) -> None:
            self.calls = calls

        def __truediv__(self, part: str) -> RecordingPath:
            self.calls.append(part)
            return self

        def glob(self, pattern: str) -> list[object]:
            assert pattern == "algo*"
            return []

    calls: list[str] = []
    _frontier_archive(
        cast(Path, RecordingPath(calls)),
        "frontier.zip",
        "model",
        (),
    )

    assert calls == ["results", "frontier"]


def test_open_weight_archive_forwards_run_aggregation_and_report_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls: list[dict[str, object]] = []
    aggregated_calls: list[dict[str, object]] = []
    report_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        osf_packaging,
        "_add_open_weight_run_entries",
        lambda **kwargs: run_calls.append(kwargs),
    )
    monkeypatch.setattr(
        osf_packaging,
        "_add_open_weight_aggregated_entries",
        lambda **kwargs: aggregated_calls.append(kwargs),
    )
    monkeypatch.setattr(
        osf_packaging,
        "_add_filtered_report_entries",
        lambda **kwargs: report_calls.append(kwargs),
    )
    archive = _open_weight_archive(
        data_root=tmp_path,
        zip_name="sweep.zip",
        study_name="open_weight_sweep",
        source_root=tmp_path / "source",
        llm_slug="qwen",
        source_model_dir="Qwen__Qwen3.5-9B",
        full_model_name="Qwen/Qwen3.5-9B",
        model_label="Qwen",
        model_column_prefix="qwen",
    )
    archive_root = PurePosixPath("results/open_weight_sweep/qwen")

    assert archive.description == "open weight sweep results for Qwen."
    assert run_calls == [
        {
            "archive": archive,
            "source_root": tmp_path / "source",
            "source_model_dir": "Qwen__Qwen3.5-9B",
            "archive_study_root": archive_root,
        }
    ]
    assert aggregated_calls == run_calls
    assert report_calls == [
        {
            "archive": archive,
            "source_root": tmp_path / "source",
            "archive_study_root": archive_root,
            "full_model_name": "Qwen/Qwen3.5-9B",
            "model_label": "Qwen",
            "model_column_prefix": "qwen",
        }
    ]


def test_open_weight_run_and_aggregated_entries_use_canonical_layout(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    run_file = source_root / "runs" / "algo1" / "model-dir" / "run.json"
    aggregated_file = source_root / "aggregated" / "algo1" / "model-dir" / "summary.csv"
    run_file.parent.mkdir(parents=True)
    aggregated_file.parent.mkdir(parents=True)
    run_file.write_text("run")
    aggregated_file.write_text("summary")
    archive = PackageArchive("results.zip", "results")
    archive_root = PurePosixPath("results/study/model")

    _add_open_weight_run_entries(
        archive=archive,
        source_root=source_root,
        source_model_dir="model-dir",
        archive_study_root=archive_root,
    )
    _add_open_weight_aggregated_entries(
        archive=archive,
        source_root=source_root,
        source_model_dir="model-dir",
        archive_study_root=archive_root,
    )

    assert [str(entry.archive_path) for entry in archive.entries] == [
        "results/study/model/algo1/runs/run.json",
        "results/study/model/algo1/aggregated/summary.csv",
    ]


def test_open_weight_entry_roots_use_exact_directory_components(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    run_file = source_root / "runs" / "algo1" / "model-dir" / "run.json"
    aggregated_file = source_root / "aggregated" / "algo1" / "model-dir" / "summary.csv"
    run_file.parent.mkdir(parents=True)
    aggregated_file.parent.mkdir(parents=True)
    run_file.write_text("run")
    aggregated_file.write_text("summary")
    run_components: list[str] = []
    aggregated_components: list[str] = []

    class RecordingSource:
        def __init__(self, calls: list[str]) -> None:
            self.calls = calls

        def __truediv__(self, part: str) -> Path:
            self.calls.append(part)
            return source_root / part

    run_archive = PackageArchive("run.zip", "run")
    aggregated_archive = PackageArchive("aggregated.zip", "aggregated")
    _add_open_weight_run_entries(
        archive=run_archive,
        source_root=RecordingSource(run_components),  # type: ignore[arg-type]
        source_model_dir="model-dir",
        archive_study_root=PurePosixPath("results/study/model"),
    )
    _add_open_weight_aggregated_entries(
        archive=aggregated_archive,
        source_root=RecordingSource(aggregated_components),  # type: ignore[arg-type]
        source_model_dir="model-dir",
        archive_study_root=PurePosixPath("results/study/model"),
    )

    assert run_components == ["runs"]
    assert aggregated_components == ["aggregated"]


def test_add_filtered_report_entries_forwards_all_summary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    (source_root / "variance_decomposition").mkdir(parents=True)
    (source_root / "variance_decomposition" / "algo1.csv").write_text("data")
    archive = PackageArchive("results.zip", "results")
    archive_root = PurePosixPath("results/study/qwen")
    json_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    csv_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        osf_packaging,
        "_add_filtered_json",
        lambda *args, **kwargs: json_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        osf_packaging,
        "_add_filtered_csv",
        lambda *args, **kwargs: csv_calls.append((args, kwargs)),
    )

    _add_filtered_report_entries(
        archive=archive,
        source_root=source_root,
        archive_study_root=archive_root,
        full_model_name="Qwen/Qwen3.5-9B",
        model_label="Qwen",
        model_column_prefix="qwen",
    )

    summaries = archive_root / "summaries"
    assert json_calls == [
        (
            (archive, source_root / "ledger.json", summaries / "ledger.json"),
            {"full_model_name": "Qwen/Qwen3.5-9B"},
        )
    ]
    expected_csv_paths = [
        ("batch_summary.csv", "batch_summary.csv"),
        ("aggregated_qwen_mistral.csv", "open_weight_ablation_summary.csv"),
        (
            "replication_budget_sufficiency_compact.csv",
            "replication_sufficiency_compact.csv",
        ),
        (
            "replication_budget_sufficiency_summary.csv",
            "replication_sufficiency_summary.csv",
        ),
    ]
    expected_csv_calls = [
        (
            (archive, source_root / source_name, summaries / target_name),
            {
                "full_model_name": "Qwen/Qwen3.5-9B",
                "model_label": "Qwen",
                "model_column_prefix": "qwen",
            },
        )
        for source_name, target_name in expected_csv_paths
    ] + [
        (
            (
                archive,
                source_root / "variance_decomposition" / "algo1.csv",
                summaries / "variance_decomposition" / "algo1.csv",
            ),
            {
                "full_model_name": "Qwen/Qwen3.5-9B",
                "model_label": "Qwen",
                "model_column_prefix": "qwen",
            },
        )
    ]
    assert csv_calls == expected_csv_calls


def test_tree_entries_continues_after_excluded_file(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.log").write_text("excluded")
    (source_root / "nested" / "kept.txt").parent.mkdir(parents=True)
    (source_root / "nested" / "kept.txt").write_text("kept")

    entries = _tree_entries(source_root, PurePosixPath("archive"))

    assert [(entry.archive_path, entry.source_path) for entry in entries] == [
        (PurePosixPath("archive/nested/kept.txt"), source_root / "nested" / "kept.txt")
    ]


def test_add_filtered_json_writes_sorted_indented_utf8_payload(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "ledger.json"
    payload = {
        "z": "café",
        "records": [
            {"identity": {"model": "Qwen/Qwen3.5-9B"}, "value": 1},
            {"identity": {"model": "Mistral"}, "value": 2},
        ],
    }
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    archive = PackageArchive("results.zip", "results")
    archive_path = PurePosixPath("summaries/ledger.json")

    _add_filtered_json(
        archive,
        source_path,
        archive_path,
        full_model_name="Qwen/Qwen3.5-9B",
    )

    expected_payload = {
        "z": "café",
        "records": [{"identity": {"model": "Qwen/Qwen3.5-9B"}, "value": 1}],
    }
    assert archive.entries == [
        PackageEntry(
            archive_path,
            data=(json.dumps(expected_payload, indent=2, sort_keys=True) + "\n").encode(),
        )
    ]


def test_add_filtered_csv_writes_filtered_content_and_archive_path(tmp_path: Path) -> None:
    source_path = tmp_path / "summary.csv"
    source_path.write_text(
        "model,qwen_mean,mistral_mean,shared\n"
        "Qwen/Qwen3.5-9B,1,2,café\n"
        "Qwen,9,8,label\n"
        "Mistral,3,4,no\n",
        encoding="utf-8",
    )
    archive = PackageArchive("results.zip", "results")

    _add_filtered_csv(
        archive,
        source_path,
        PurePosixPath("summaries/summary.csv"),
        full_model_name="Qwen/Qwen3.5-9B",
        model_label="Qwen",
        model_column_prefix="qwen",
    )

    assert archive.entries == [
        PackageEntry(
            PurePosixPath("summaries/summary.csv"),
            data=("model,qwen_mean,shared\r\nQwen/Qwen3.5-9B,1,café\r\nQwen,9,label\r\n").encode(),
        )
    ]


def test_filter_csv_handles_uppercase_model_header_and_empty_selection() -> None:
    assert (
        _filter_csv(
            "Model,status\nMistral,failed\n",
            full_model_name="Qwen/Qwen3.5-9B",
            model_label="Qwen",
            model_column_prefix="qwen",
        )
        is None
    )


def test_drop_other_model_columns_and_render_csv_preserve_selected_data() -> None:
    assert _drop_other_model_columns(
        ["qwen_mean", "mistral_mean", "qwen_status", "common"],
        model_column_prefix="qwen",
    ) == ["qwen_mean", "qwen_status", "common"]
    assert _drop_other_model_columns(
        ["qwen_mean", "mistral_mean", "mistral_status", "common"],
        model_column_prefix="mistral",
    ) == ["mistral_mean", "mistral_status", "common"]
    assert (
        _render_csv(
            ["name", "status"],
            [
                {"name": "one", "status": "ok", "ignored": "extra"},
                {"name": "two"},
            ],
        )
        == "name,status\r\none,ok\r\ntwo,\r\n"
    )


def test_package_write_result_and_archive_readme_are_exact() -> None:
    output_dir = Path("package")
    readme_path = output_dir / "README.md"
    checksum_path = output_dir / "checksums.txt"
    zip_paths = [output_dir / "one.zip"]
    assert _package_write_result(output_dir, readme_path, checksum_path, zip_paths) == (
        PackageWriteResult(output_dir, readme_path, checksum_path, zip_paths)
    )

    archive = PackageArchive("one.zip", "Description")
    first = PackageEntry(PurePosixPath("first.txt"), data=b"first")
    archive.entries.append(first)
    _prepend_archive_readme(archive)

    assert archive.entries[0].archive_path == PurePosixPath("README.md")
    assert archive.entries[0].data == osf_packaging._archive_readme(archive).encode()
    assert archive.entries[1] is first


def test_package_readme_lists_archives_and_upload_checklist() -> None:
    readme = _package_readme(
        [PackageArchive("one.zip", "First"), PackageArchive("two.zip", "Second")]
    )

    assert readme.startswith("# OSF PLOS Reviewer Package\n\n")
    assert "- `one.zip` — First\n" in readme
    assert "- `two.zip` — Second\n" in readme
    assert "1. Upload all ZIP files to the OSF results folder.\n" in readme
    assert "3. Verify uploaded file hashes against `checksums.txt`.\n" in readme
    assert "XX" not in readme


def test_write_zip_is_sorted_deterministic_and_reads_source_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(b"source")
    archive = PackageArchive(
        "one.zip",
        "one",
        [
            PackageEntry(PurePosixPath("z.txt"), source_path=source_path),
            PackageEntry(PurePosixPath("a.txt"), data=b"data"),
        ],
    )
    zip_path = tmp_path / "one.zip"

    original_zip_info = zipfile.ZipInfo
    zip_info_calls: list[tuple[str, tuple[int, int, int, int, int, int] | None]] = []

    class RecordingZipInfo(original_zip_info):
        def __init__(
            self,
            filename: str,
            date_time: tuple[int, int, int, int, int, int] | None = None,
        ) -> None:
            zip_info_calls.append((filename, date_time))
            if date_time is None:
                super().__init__(filename)
            else:
                super().__init__(filename, date_time=date_time)

    monkeypatch.setattr(osf_packaging.zipfile, "ZipInfo", RecordingZipInfo)
    _write_zip(zip_path, archive)

    assert zip_info_calls == [
        ("a.txt", (1980, 1, 1, 0, 0, 1)),
        ("z.txt", (1980, 1, 1, 0, 0, 1)),
    ]

    with zipfile.ZipFile(zip_path) as zip_file:
        infos = zip_file.infolist()
        assert [info.filename for info in infos] == ["a.txt", "z.txt"]
        assert [zip_file.read(info) for info in infos] == [b"data", b"source"]
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in infos)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)

    with pytest.raises(ValueError, match="neither data nor source path"):
        _read_source_entry(PackageEntry(PurePosixPath("missing.txt")))


def test_checksums_are_sha256_lines_with_trailing_newline(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert _checksums([first, second]) == (
        f"{hashlib.sha256(b'first').hexdigest()}  first.zip\n"
        f"{hashlib.sha256(b'second').hexdigest()}  second.zip\n"
    )
