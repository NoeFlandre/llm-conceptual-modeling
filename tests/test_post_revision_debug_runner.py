from importlib import util
from pathlib import Path


def _load_runner_module():
    script_path = Path("scripts/post_revision_debug/run_mistral_probe_matrix.py")
    spec = util.spec_from_file_location("run_mistral_probe_matrix", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_mistral_probe_matrix.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_probe_specs_cover_the_expected_representative_rows() -> None:
    module = _load_runner_module()

    probe_specs = module._default_probe_specs(  # type: ignore[attr-defined]
        algo1_rows=[],
        algo2_rows=[],
        algo3_rows=[],
    )

    assert [(spec.algorithm, spec.row_index) for spec in probe_specs] == [
        ("algo1", 0),
        ("algo1", 80),
        ("algo2", 0),
        ("algo2", 160),
        ("algo3", 0),
        ("algo3", 1),
    ]


def test_custom_probe_rows_override_the_defaults() -> None:
    module = _load_runner_module()

    probe_specs = module._default_probe_specs(  # type: ignore[attr-defined]
        algo1_rows=[3],
        algo2_rows=[7, 11],
        algo3_rows=[13],
    )

    assert [(spec.algorithm, spec.row_index) for spec in probe_specs] == [
        ("algo1", 3),
        ("algo2", 7),
        ("algo2", 11),
        ("algo3", 13),
    ]
