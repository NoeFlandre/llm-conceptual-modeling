from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path, PurePosixPath

EXPECTED_ZIP_NAMES = (
    "inputs.zip",
    "analysis_revision_tracker.zip",
    "results_frontier_gpt-5.zip",
    "results_frontier_gpt-4o.zip",
    "results_frontier_gemini-2.5-pro.zip",
    "results_frontier_gemini-2.0-flash.zip",
    "results_frontier_deepseek-v3.1.zip",
    "results_frontier_deepseek-v3-0324.zip",
    "results_open_weight_sweep_qwen.zip",
    "results_open_weight_sweep_mistral.zip",
    "results_open_weight_map_extension_qwen.zip",
    "results_open_weight_map_extension_mistral.zip",
)

_FRONTIER_MODELS = (
    ("results_frontier_gpt-5.zip", "gpt-5", ("gpt-5",)),
    ("results_frontier_gpt-4o.zip", "gpt-4o", ("openai-gpt-4o",)),
    (
        "results_frontier_gemini-2.5-pro.zip",
        "gemini-2.5-pro",
        ("gemini-2.5-pro", "google-gemini-2.5-pro"),
    ),
    ("results_frontier_gemini-2.0-flash.zip", "gemini-2.0-flash", ("gemini-2.0-flash-exp",)),
    ("results_frontier_deepseek-v3.1.zip", "deepseek-v3.1", ("deepseek-chat-v3.1",)),
    (
        "results_frontier_deepseek-v3-0324.zip",
        "deepseek-v3-0324",
        ("deepseek-v3-chat-0324", "deepseek-chat-v3-0324"),
    ),
)

_OPEN_WEIGHT_MODELS = (
    ("qwen", "Qwen__Qwen3.5-9B", "Qwen/Qwen3.5-9B", "Qwen", "qwen"),
    (
        "mistral",
        "mistralai__Ministral-3-8B-Instruct-2512",
        "mistralai/Ministral-3-8B-Instruct-2512",
        "Mistral",
        "mistral",
    ),
)

_EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "archives",
    "stale-shards",
    "worker-queues",
    "preview",
    "preview_resume",
}

_EXCLUDED_NAMES = {".DS_Store"}
_EXCLUDED_SUFFIXES = {".log", ".pid", ".pyc"}


@dataclass(frozen=True)
class PackageEntry:
    archive_path: PurePosixPath
    source_path: Path | None = None
    data: bytes | None = None


@dataclass
class PackageArchive:
    name: str
    description: str
    entries: list[PackageEntry] = field(default_factory=list)


@dataclass(frozen=True)
class PackageWriteResult:
    output_dir: Path
    readme_path: Path
    checksum_path: Path
    zip_paths: list[Path]


def build_osf_manifest(data_root: Path) -> list[PackageArchive]:
    data_root = Path(data_root)
    archives = [
        _shared_inputs_archive(data_root),
        _analysis_archive(data_root),
        *[
            _frontier_archive(data_root, zip_name, model_name, aliases)
            for zip_name, model_name, aliases in _FRONTIER_MODELS
        ],
        *[
            _open_weight_archive(
                data_root=data_root,
                zip_name=f"results_open_weight_sweep_{llm_slug}.zip",
                study_name="open_weight_sweep",
                source_root=data_root / "results" / "open_weights" / "hf-paper-batch-canonical",
                llm_slug=llm_slug,
                source_model_dir=source_model_dir,
                full_model_name=full_model_name,
                model_label=model_label,
                model_column_prefix=model_column_prefix,
            )
            for (
                llm_slug,
                source_model_dir,
                full_model_name,
                model_label,
                model_column_prefix,
            ) in _OPEN_WEIGHT_MODELS
        ],
        *[
            _open_weight_archive(
                data_root=data_root,
                zip_name=f"results_open_weight_map_extension_{llm_slug}.zip",
                study_name="open_weight_map_extension",
                source_root=data_root / "results" / "open_weights" / "hf-map-extension-canonical",
                llm_slug=llm_slug,
                source_model_dir=source_model_dir,
                full_model_name=full_model_name,
                model_label=model_label,
                model_column_prefix=model_column_prefix,
            )
            for (
                llm_slug,
                source_model_dir,
                full_model_name,
                model_label,
                model_column_prefix,
            ) in _OPEN_WEIGHT_MODELS
        ],
    ]
    _assert_expected_archive_names(archives)
    for archive in archives:
        _prepend_archive_readme(archive)
        _assert_unique_archive_paths(archive)
    return archives


def write_osf_package(
    *,
    data_root: Path,
    output_dir: Path,
    manifest: list[PackageArchive] | None = None,
    dry_run: bool = False,
) -> PackageWriteResult:
    output_dir = Path(output_dir)
    archives = manifest or build_osf_manifest(Path(data_root))
    for archive in archives:
        _assert_unique_archive_paths(archive)

    readme_text = _package_readme(archives)
    zip_paths = [output_dir / archive.name for archive in archives]
    readme_path = output_dir / "README.md"
    checksum_path = output_dir / "checksums.txt"
    if dry_run:
        return PackageWriteResult(
            output_dir=output_dir,
            readme_path=readme_path,
            checksum_path=checksum_path,
            zip_paths=zip_paths,
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_text, encoding="utf-8")
    for archive, zip_path in zip(archives, zip_paths, strict=True):
        _write_zip(zip_path, archive)
    checksum_path.write_text(_checksums(zip_paths), encoding="utf-8")
    return PackageWriteResult(
        output_dir=output_dir,
        readme_path=readme_path,
        checksum_path=checksum_path,
        zip_paths=zip_paths,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OSF reviewer ZIP artifacts.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dist/osf_plos_package"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = write_osf_package(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print(f"Output directory: {result.output_dir}")
    print(f"Package README: {result.readme_path}")
    print(f"Checksums: {result.checksum_path}")
    for zip_path in result.zip_paths:
        print(zip_path)
    return 0


def _shared_inputs_archive(data_root: Path) -> PackageArchive:
    archive = PackageArchive(
        name="inputs.zip",
        description="Shared input causal maps, thesaurus, and lexicon files.",
    )
    inputs_root = data_root / "inputs"
    _add_tree_entries(
        archive,
        source_root=inputs_root,
        archive_root=PurePosixPath("inputs"),
    )
    return archive


def _analysis_archive(data_root: Path) -> PackageArchive:
    archive = PackageArchive(
        name="analysis_revision_tracker.zip",
        description="Revision-tracker analysis artifacts used for paper reporting.",
    )
    tracker_root = data_root / "analysis_artifacts" / "revision_tracker"
    _add_tree_entries(
        archive,
        source_root=tracker_root,
        archive_root=PurePosixPath("analysis/revision_tracker"),
    )
    return archive


def _frontier_archive(
    data_root: Path,
    zip_name: str,
    model_name: str,
    source_aliases: tuple[str, ...],
) -> PackageArchive:
    archive = PackageArchive(
        name=zip_name,
        description=f"Frontier-model results for {model_name}.",
    )
    frontier_root = data_root / "results" / "frontier"
    for algorithm_root in sorted(frontier_root.glob("algo*")):
        if not algorithm_root.is_dir():
            continue
        for source_alias in source_aliases:
            source_root = algorithm_root / source_alias
            if not source_root.exists():
                continue
            _add_tree_entries(
                archive,
                source_root=source_root,
                archive_root=PurePosixPath("results/frontier") / model_name / algorithm_root.name,
            )
    return archive


def _open_weight_archive(
    *,
    data_root: Path,
    zip_name: str,
    study_name: str,
    source_root: Path,
    llm_slug: str,
    source_model_dir: str,
    full_model_name: str,
    model_label: str,
    model_column_prefix: str,
) -> PackageArchive:
    archive = PackageArchive(
        name=zip_name,
        description=f"{study_name.replace('_', ' ')} results for {model_label}.",
    )
    archive_study_root = PurePosixPath("results") / study_name / llm_slug
    _add_open_weight_run_entries(
        archive=archive,
        source_root=source_root,
        source_model_dir=source_model_dir,
        archive_study_root=archive_study_root,
    )
    _add_open_weight_aggregated_entries(
        archive=archive,
        source_root=source_root,
        source_model_dir=source_model_dir,
        archive_study_root=archive_study_root,
    )
    _add_filtered_report_entries(
        archive=archive,
        source_root=source_root,
        archive_study_root=archive_study_root,
        full_model_name=full_model_name,
        model_label=model_label,
        model_column_prefix=model_column_prefix,
    )
    return archive


def _add_open_weight_run_entries(
    *,
    archive: PackageArchive,
    source_root: Path,
    source_model_dir: str,
    archive_study_root: PurePosixPath,
) -> None:
    runs_root = source_root / "runs"
    for model_root in sorted(runs_root.glob(f"algo*/{source_model_dir}")):
        algorithm = model_root.parent.name
        _add_tree_entries(
            archive,
            source_root=model_root,
            archive_root=archive_study_root / algorithm / "runs",
        )


def _add_open_weight_aggregated_entries(
    *,
    archive: PackageArchive,
    source_root: Path,
    source_model_dir: str,
    archive_study_root: PurePosixPath,
) -> None:
    aggregated_root = source_root / "aggregated"
    for model_root in sorted(aggregated_root.glob(f"algo*/{source_model_dir}")):
        algorithm = model_root.parent.name
        _add_tree_entries(
            archive,
            source_root=model_root,
            archive_root=archive_study_root / algorithm / "aggregated",
        )


def _add_filtered_report_entries(
    *,
    archive: PackageArchive,
    source_root: Path,
    archive_study_root: PurePosixPath,
    full_model_name: str,
    model_label: str,
    model_column_prefix: str,
) -> None:
    summaries_root = archive_study_root / "summaries"
    _add_filtered_json(
        archive,
        source_root / "ledger.json",
        summaries_root / "ledger.json",
        full_model_name=full_model_name,
    )
    for source_name, target_name in (
        ("batch_summary.csv", "batch_summary.csv"),
        ("aggregated_qwen_mistral.csv", "open_weight_ablation_summary.csv"),
        ("replication_budget_sufficiency_compact.csv", "replication_sufficiency_compact.csv"),
        ("replication_budget_sufficiency_summary.csv", "replication_sufficiency_summary.csv"),
    ):
        _add_filtered_csv(
            archive,
            source_root / source_name,
            summaries_root / target_name,
            full_model_name=full_model_name,
            model_label=model_label,
            model_column_prefix=model_column_prefix,
        )
    for csv_path in sorted((source_root / "variance_decomposition").glob("*.csv")):
        _add_filtered_csv(
            archive,
            csv_path,
            summaries_root / "variance_decomposition" / csv_path.name,
            full_model_name=full_model_name,
            model_label=model_label,
            model_column_prefix=model_column_prefix,
        )


def _add_tree_entries(
    archive: PackageArchive,
    *,
    source_root: Path,
    archive_root: PurePosixPath,
) -> None:
    if not source_root.exists():
        return
    for source_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        if _is_excluded_path(source_path):
            continue
        relative_path = source_path.relative_to(source_root)
        archive.entries.append(
            PackageEntry(
                archive_path=archive_root / _to_posix_path(relative_path),
                source_path=source_path,
            )
        )


def _add_filtered_json(
    archive: PackageArchive,
    source_path: Path,
    archive_path: PurePosixPath,
    *,
    full_model_name: str,
) -> None:
    if not source_path.exists() or _is_excluded_path(source_path):
        return
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = {
            **payload,
            "records": [
                record for record in payload["records"] if _record_model(record) == full_model_name
            ],
        }
    archive.entries.append(
        PackageEntry(
            archive_path=archive_path,
            data=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    )


def _add_filtered_csv(
    archive: PackageArchive,
    source_path: Path,
    archive_path: PurePosixPath,
    *,
    full_model_name: str,
    model_label: str,
    model_column_prefix: str,
) -> None:
    if not source_path.exists() or _is_excluded_path(source_path):
        return
    content = _filter_csv(
        source_path.read_text(encoding="utf-8"),
        full_model_name=full_model_name,
        model_label=model_label,
        model_column_prefix=model_column_prefix,
    )
    if content is None:
        return
    archive.entries.append(
        PackageEntry(
            archive_path=archive_path,
            data=content.encode("utf-8"),
        )
    )


def _filter_csv(
    text: str,
    *,
    full_model_name: str,
    model_label: str,
    model_column_prefix: str,
) -> str | None:
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        return text
    rows = list(reader)
    fieldnames = list(reader.fieldnames)
    if "model" in fieldnames:
        rows = [row for row in rows if row.get("model") in {full_model_name, model_label}]
    elif "Model" in fieldnames:
        rows = [row for row in rows if row.get("Model") == model_label]
    prefixed_columns = [name for name in fieldnames if name.startswith(f"{model_column_prefix}_")]
    other_model_prefixes = {
        prefix for prefix in ("qwen", "mistral") if prefix != model_column_prefix
    }
    if prefixed_columns:
        fieldnames = [
            name
            for name in fieldnames
            if not any(name.startswith(f"{prefix}_") for prefix in other_model_prefixes)
        ]
    if not rows and ("model" in fieldnames or "Model" in fieldnames):
        return None
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _prepend_archive_readme(archive: PackageArchive) -> None:
    archive.entries.insert(
        0,
        PackageEntry(
            archive_path=PurePosixPath("README.md"),
            data=_archive_readme(archive).encode("utf-8"),
        ),
    )


def _archive_readme(archive: PackageArchive) -> str:
    return (
        f"# {archive.name}\n\n"
        f"{archive.description}\n\n"
        "This ZIP is part of the OSF/PLOS reviewer package. It contains only "
        "paper-facing inputs, analysis artifacts, or canonical result files. "
        "Operational archives, stale shards, worker queues, preview folders, logs, "
        "sync files, caches, and duplicate runtime layouts are intentionally excluded.\n"
    )


def _package_readme(archives: list[PackageArchive]) -> str:
    lines = [
        "# OSF PLOS Reviewer Package",
        "",
        "This directory contains reviewer-facing ZIP files prepared from the local data tree.",
        "GitHub remains the source-code home, and Hugging Face remains the modular data bucket.",
        "OSF is the PLOS-compatible archival package.",
        "",
        "## ZIP files",
        "",
    ]
    for archive in archives:
        lines.append(f"- `{archive.name}` — {archive.description}")
    lines.extend(
        [
            "",
            "## Upload checklist",
            "",
            "1. Upload all ZIP files to the OSF results folder.",
            "2. Upload `README.md` and `checksums.txt` next to the ZIP files.",
            "3. Verify uploaded file hashes against `checksums.txt`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_zip(zip_path: Path, archive: PackageArchive) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for entry in sorted(archive.entries, key=lambda item: str(item.archive_path)):
            data = entry.data if entry.data is not None else _read_source_entry(entry)
            zip_info = zipfile.ZipInfo(str(entry.archive_path), date_time=(1980, 1, 1, 0, 0, 0))
            zip_info.compress_type = zipfile.ZIP_DEFLATED
            zip_file.writestr(zip_info, data)


def _read_source_entry(entry: PackageEntry) -> bytes:
    if entry.source_path is None:
        raise ValueError(f"Package entry has neither data nor source path: {entry.archive_path}")
    return entry.source_path.read_bytes()


def _checksums(zip_paths: list[Path]) -> str:
    lines: list[str] = []
    for zip_path in zip_paths:
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {zip_path.name}")
    return "\n".join(lines) + "\n"


def _is_excluded_path(path: Path) -> bool:
    if path.name in _EXCLUDED_NAMES:
        return True
    if path.suffix in _EXCLUDED_SUFFIXES:
        return True
    if "results-sync" in path.name:
        return True
    return any(part in _EXCLUDED_PARTS for part in path.parts)


def _record_model(record: object) -> str | None:
    normalized = _string_key_mapping(record)
    if normalized is None:
        return None
    identity = _string_key_mapping(normalized.get("identity"))
    identity_model = identity.get("model") if identity is not None else None
    if isinstance(identity_model, str):
        return identity_model
    model = normalized.get("model")
    return model if isinstance(model, str) else None


def _string_key_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _to_posix_path(path: Path) -> PurePosixPath:
    return PurePosixPath(*path.parts)


def _assert_expected_archive_names(archives: list[PackageArchive]) -> None:
    names = [archive.name for archive in archives]
    if names != list(EXPECTED_ZIP_NAMES):
        raise ValueError(f"Unexpected OSF ZIP names: {names!r}")


def _assert_unique_archive_paths(archive: PackageArchive) -> None:
    seen: set[PurePosixPath] = set()
    for entry in archive.entries:
        if entry.archive_path in seen:
            raise ValueError(f"Duplicate archive path in {archive.name}: {entry.archive_path}")
        seen.add(entry.archive_path)


if __name__ == "__main__":
    raise SystemExit(main())
