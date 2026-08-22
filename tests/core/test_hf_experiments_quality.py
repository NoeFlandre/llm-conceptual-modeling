from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_conceptual_modeling import hf_experiments as experiments


def _spec(
    *,
    model: str = "model-a",
    algorithm: str = "algo1",
    graph_source: str = "default",
    worker_process_mode: str = "subprocess",
    decoding_algorithm: str = "greedy",
    replication: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        algorithm=algorithm,
        graph_source=graph_source,
        condition_label="condition",
        embedding_model="embed",
        pair_name="sg1_sg2",
        condition_bits="001",
        replication=replication,
        decoding=SimpleNamespace(algorithm=decoding_algorithm),
        context_policy={
            "worker_process_mode": worker_process_mode,
            "max_requests_per_worker_process": 7,
        },
        raw_context={"model": model, "algorithm": algorithm},
    )


def _setup(
    tmp_path: Path,
    *,
    specs: list[object] | None = None,
    runtime_factory: object | None = "runtime",
    monitored: bool = False,
    dry_run: bool = False,
    resume: bool = False,
) -> object:
    return experiments._BatchSetup(
        output_root=tmp_path / "output",
        planned_specs=specs or [_spec()],
        runtime_factory=runtime_factory,
        use_monitored_hf_subprocess=monitored,
        dry_run=dry_run,
        resume=resume,
    )


def _state(*, total_runs: int = 1, specs: list[object] | None = None) -> object:
    return experiments._BatchState(
        total_runs=total_runs,
        planned_specs=specs or [_spec()],
        status_snapshot={
            "total_runs": total_runs,
            "finished_count": 0,
            "failed_count": 0,
            "running_count": 0,
            "pending_count": total_runs,
            "failure_count": 0,
            "failures": [],
            "percent_complete": 0.0,
            "current_run": None,
            "last_completed_run": None,
            "started_at": "start",
            "updated_at": "start",
        },
        summary_rows=[],
        persistent_sessions={},
        seeded_finished_run_dirs=set(),
        seeded_failed_run_dirs=set(),
    )


def test_run_local_hf_spec_subprocess_forwards_all_runtime_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    run_dir = tmp_path / "run"
    observed: dict[str, object] = {}

    def implementation(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"raw_row": {"ok": True}}

    monkeypatch.setattr(experiments, "_run_local_hf_spec_subprocess_impl", implementation)

    result = experiments._run_local_hf_spec_subprocess(spec=spec, run_dir=run_dir)

    assert result == {"raw_row": {"ok": True}}
    assert observed["spec"] is spec
    assert observed["run_dir"] == run_dir
    assert observed["run_monitored_command_fn"] is experiments.run_monitored_command
    assert observed["build_worker_command_fn"] is experiments._build_worker_command
    assert observed["is_retryable_worker_error_fn"] is experiments._is_retryable_worker_error
    assert observed["validate_runtime_result_fn"] is experiments._validate_structural_runtime_result


def test_close_incompatible_persistent_sessions_closes_only_other_models() -> None:
    class Session:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    selected = Session()
    other = Session()
    sessions = {"model-a": selected, "model-b": other}

    experiments._close_incompatible_persistent_sessions(
        model="model-a",
        persistent_sessions=sessions,
    )

    assert sessions == {"model-a": selected}
    assert not selected.closed
    assert other.closed


def test_get_persistent_session_creates_once_with_resolved_worker_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class Session:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(experiments, "PersistentHFWorkerSession", Session)
    spec = _spec(model="org/model")
    sessions: dict[str, object] = {}

    first = experiments._get_persistent_session(
        spec=spec,
        output_root=tmp_path,
        persistent_sessions=sessions,
    )
    second = experiments._get_persistent_session(
        spec=spec,
        output_root=tmp_path,
        persistent_sessions=sessions,
    )

    assert first is second
    assert len(created) == 1
    assert created[0]["queue_dir"] == tmp_path / "worker-queues" / "org__model"
    assert created[0]["worker_python"] == experiments.sys.executable
    assert created[0]["max_requests_per_process"] == 7


def test_run_local_hf_spec_uses_subprocess_for_nonpersistent_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(worker_process_mode="subprocess")
    expected = {"raw_row": {"path": "subprocess"}}
    observed: list[tuple[object, Path]] = []
    close_calls: list[dict[str, object]] = []
    mode_calls: list[object] = []

    monkeypatch.setattr(
        experiments,
        "_resolve_worker_process_mode",
        lambda policy: mode_calls.append(policy) or "subprocess",
    )
    monkeypatch.setattr(
        experiments,
        "_close_incompatible_persistent_sessions",
        lambda **kwargs: close_calls.append(kwargs),
    )
    monkeypatch.setattr(
        experiments,
        "_run_local_hf_spec_subprocess",
        lambda *, spec, run_dir: observed.append((spec, run_dir)) or expected,
    )

    result = experiments._run_local_hf_spec(
        spec=spec,
        run_dir=tmp_path / "run",
        output_root=tmp_path,
        persistent_sessions={},
    )

    assert result is expected
    assert observed == [(spec, tmp_path / "run")]
    assert close_calls == [{"model": spec.model, "persistent_sessions": {}}]
    assert mode_calls == [spec.context_policy]


def test_run_local_hf_spec_uses_cached_persistent_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(worker_process_mode="persistent")
    observed: list[tuple[object, Path]] = []
    close_calls: list[dict[str, object]] = []
    session_calls: list[dict[str, object]] = []

    class Session:
        def run_spec(self, *, spec: object, run_dir: Path) -> dict[str, object]:
            observed.append((spec, run_dir))
            return {"raw_row": {"path": "persistent"}}

    session = Session()
    mode_calls: list[object] = []
    monkeypatch.setattr(
        experiments,
        "_resolve_worker_process_mode",
        lambda policy: mode_calls.append(policy) or "persistent",
    )
    monkeypatch.setattr(
        experiments,
        "_close_incompatible_persistent_sessions",
        lambda **kwargs: close_calls.append(kwargs),
    )
    monkeypatch.setattr(
        experiments,
        "_get_persistent_session",
        lambda **kwargs: session_calls.append(kwargs) or session,
    )
    persistent_sessions: dict[str, object] = {}

    result = experiments._run_local_hf_spec(
        spec=spec,
        run_dir=tmp_path / "run",
        output_root=tmp_path,
        persistent_sessions=persistent_sessions,
    )

    assert result == {"raw_row": {"path": "persistent"}}
    assert observed == [(spec, tmp_path / "run")]
    assert close_calls == [{"model": spec.model, "persistent_sessions": persistent_sessions}]
    assert session_calls == [
        {
            "spec": spec,
            "output_root": tmp_path,
            "persistent_sessions": persistent_sessions,
        }
    ]
    assert mode_calls == [spec.context_policy]


def test_resolve_batch_arguments_uses_explicit_arguments_without_config(tmp_path: Path) -> None:
    result = experiments._resolve_batch_arguments(
        output_root=tmp_path,
        models=["model"],
        embedding_model="embed",
        replications=3,
        config=None,
    )

    assert result == experiments._BatchArguments(tmp_path, ["model"], "embed", 3)


def test_resolve_batch_arguments_uses_config_as_source_of_truth(tmp_path: Path) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(output_root=tmp_path / "configured", replications=9),
        models=SimpleNamespace(
            chat_models=["configured-model"], embedding_model="configured-embed"
        ),
    )

    result = experiments._resolve_batch_arguments(
        output_root=tmp_path / "ignored",
        models=["ignored"],
        embedding_model="ignored",
        replications=1,
        config=config,
    )

    assert result == experiments._BatchArguments(
        tmp_path / "configured",
        ["configured-model"],
        "configured-embed",
        9,
    )


def test_build_batch_hf_runtime_only_builds_for_local_non_monitored_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str | None] = []
    monkeypatch.setattr(experiments, "_resolve_hf_token", lambda: "token")
    monkeypatch.setattr(
        experiments,
        "build_runtime_factory",
        lambda *, hf_token: observed.append(hf_token) or "built",
    )

    assert (
        experiments._build_batch_hf_runtime(
            runtime_factory=None,
            use_monitored_hf_subprocess=False,
        )
        == "built"
    )
    assert (
        experiments._build_batch_hf_runtime(
            runtime_factory="given",
            use_monitored_hf_subprocess=False,
        )
        is None
    )
    assert (
        experiments._build_batch_hf_runtime(
            runtime_factory=None,
            use_monitored_hf_subprocess=True,
        )
        is None
    )
    assert observed == ["token"]


def test_resolve_batch_profile_provider_selects_default_or_runtime_provider() -> None:
    runtime = SimpleNamespace(profile_for_chat_model="profile-provider")

    assert (
        experiments._resolve_batch_profile_provider(
            dry_run=True,
            use_monitored_hf_subprocess=False,
            hf_runtime=runtime,
        )
        is experiments.default_runtime_profile_provider
    )
    assert (
        experiments._resolve_batch_profile_provider(
            dry_run=False,
            use_monitored_hf_subprocess=True,
            hf_runtime=runtime,
        )
        is experiments.default_runtime_profile_provider
    )
    assert (
        experiments._resolve_batch_profile_provider(
            dry_run=False,
            use_monitored_hf_subprocess=False,
            hf_runtime=runtime,
        )
        == "profile-provider"
    )
    assert (
        experiments._resolve_batch_profile_provider(
            dry_run=False,
            use_monitored_hf_subprocess=False,
            hf_runtime=None,
        )
        is None
    )


def test_resolve_batch_runtime_factory_covers_monitored_local_and_explicit_paths() -> None:
    def runtime_factory(_spec: object) -> dict[str, object]:
        return {"raw_row": {}}

    runtime = SimpleNamespace()

    assert (
        experiments._resolve_batch_runtime_factory(
            runtime_factory=runtime_factory,
            use_monitored_hf_subprocess=False,
            hf_runtime=runtime,
        )
        is runtime_factory
    )
    assert (
        experiments._resolve_batch_runtime_factory(
            runtime_factory=None,
            use_monitored_hf_subprocess=True,
            hf_runtime=None,
        )
        is None
    )

    converted = object()
    converted_calls: list[object] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            experiments,
            "_runtime_factory_from_hf_runtime",
            lambda value: converted_calls.append(value) or converted,
        )
        assert (
            experiments._resolve_batch_runtime_factory(
                runtime_factory=None,
                use_monitored_hf_subprocess=False,
                hf_runtime=runtime,
            )
            is converted
        )
        assert converted_calls == [runtime]
    finally:
        monkeypatch.undo()

    with pytest.raises(ValueError, match="Missing HF runtime"):
        experiments._resolve_batch_runtime_factory(
            runtime_factory=None,
            use_monitored_hf_subprocess=False,
            hf_runtime=None,
        )

    with pytest.raises(
        ValueError,
        match=r"^Missing HF runtime for non-dry local execution\.$",
    ):
        experiments._resolve_batch_runtime_factory(
            runtime_factory=None,
            use_monitored_hf_subprocess=False,
            hf_runtime=None,
        )


def test_resolve_batch_runtime_composes_runtime_resolution_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    hf_runtime = object()
    resolved_factory = object()

    def build(**kwargs: object) -> object:
        calls.append(("build", kwargs))
        return hf_runtime

    def profile(**kwargs: object) -> object:
        calls.append(("profile", kwargs))
        return "profile"

    def factory(**kwargs: object) -> object:
        calls.append(("factory", kwargs))
        return resolved_factory

    monkeypatch.setattr(experiments, "_build_batch_hf_runtime", build)
    monkeypatch.setattr(experiments, "_resolve_batch_profile_provider", profile)
    monkeypatch.setattr(experiments, "_resolve_batch_runtime_factory", factory)

    result = experiments._resolve_batch_runtime(runtime_factory=None, dry_run=False)

    assert result == experiments._BatchRuntime(resolved_factory, "profile", True)
    assert calls == [
        (
            "build",
            {"runtime_factory": None, "use_monitored_hf_subprocess": True},
        ),
        (
            "profile",
            {
                "dry_run": False,
                "use_monitored_hf_subprocess": True,
                "hf_runtime": hf_runtime,
            },
        ),
        (
            "factory",
            {
                "runtime_factory": None,
                "use_monitored_hf_subprocess": True,
                "hf_runtime": hf_runtime,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("runtime_factory", "dry_run", "expected_monitored"),
    [(object(), False, False), (None, True, False)],
)
def test_resolve_batch_runtime_preserves_mode_inputs(
    monkeypatch: pytest.MonkeyPatch,
    runtime_factory: object | None,
    dry_run: bool,
    expected_monitored: bool,
) -> None:
    observed: list[tuple[str, dict[str, object]]] = []
    hf_runtime = object()
    profile = object()
    resolved_factory = object()

    monkeypatch.setattr(
        experiments,
        "_build_batch_hf_runtime",
        lambda **kwargs: observed.append(("build", kwargs)) or hf_runtime,
    )
    monkeypatch.setattr(
        experiments,
        "_resolve_batch_profile_provider",
        lambda **kwargs: observed.append(("profile", kwargs)) or profile,
    )
    monkeypatch.setattr(
        experiments,
        "_resolve_batch_runtime_factory",
        lambda **kwargs: observed.append(("factory", kwargs)) or resolved_factory,
    )

    result = experiments._resolve_batch_runtime(
        runtime_factory=runtime_factory,
        dry_run=dry_run,
    )

    assert result == experiments._BatchRuntime(
        resolved_factory,
        profile,
        expected_monitored,
    )
    assert observed == [
        (
            "build",
            {
                "runtime_factory": runtime_factory,
                "use_monitored_hf_subprocess": expected_monitored,
            },
        ),
        (
            "profile",
            {
                "dry_run": dry_run,
                "use_monitored_hf_subprocess": expected_monitored,
                "hf_runtime": hf_runtime,
            },
        ),
        (
            "factory",
            {
                "runtime_factory": runtime_factory,
                "use_monitored_hf_subprocess": expected_monitored,
                "hf_runtime": hf_runtime,
            },
        ),
    ]


def test_prepare_batch_setup_resolves_arguments_runtime_and_planned_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    output_root = tmp_path / "missing" / "output"
    arguments = experiments._BatchArguments(output_root, ["m"], "e", 2)
    provided_runtime_factory = object()
    planned = ["planned"]
    calls: dict[str, object] = {}

    argument_calls: list[dict[str, object]] = []
    runtime_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_resolve_batch_arguments",
        lambda **kwargs: argument_calls.append(kwargs) or arguments,
    )
    monkeypatch.setattr(
        experiments,
        "_resolve_batch_runtime",
        lambda **kwargs: (
            runtime_calls.append(kwargs)
            or experiments._BatchRuntime(kwargs["runtime_factory"], "profile", True)
        ),
    )

    def plan(**kwargs: object) -> list[object]:
        calls["plan"] = kwargs
        return planned

    def filter_specs(**kwargs: object) -> list[object]:
        calls["filter"] = kwargs
        return ["filtered"]

    monkeypatch.setattr(experiments, "plan_paper_batch_specs", plan)
    monkeypatch.setattr(experiments, "_filter_planned_specs_for_output_root", filter_specs)

    result = experiments._prepare_batch_setup(
        output_root=tmp_path / "ignored",
        models=["ignored"],
        embedding_model="ignored",
        replications=1,
        algorithms=("algo1",),
        config=config,
        runtime_factory=provided_runtime_factory,
        resume=True,
        dry_run=False,
    )

    assert result == experiments._BatchSetup(
        output_root=output_root,
        planned_specs=["filtered"],
        runtime_factory=provided_runtime_factory,
        use_monitored_hf_subprocess=True,
        dry_run=False,
        resume=True,
    )
    assert calls["plan"] == {
        "models": ["m"],
        "embedding_model": "e",
        "replications": 2,
        "algorithms": ("algo1",),
        "config": config,
        "runtime_profile_provider": "profile",
    }
    assert calls["filter"] == {
        "planned_specs": planned,
        "output_root": output_root,
    }
    assert argument_calls == [
        {
            "output_root": tmp_path / "ignored",
            "models": ["ignored"],
            "embedding_model": "ignored",
            "replications": 1,
            "config": config,
        }
    ]
    assert runtime_calls == [{"runtime_factory": provided_runtime_factory, "dry_run": False}]
    assert output_root.is_dir()

    result_again = experiments._prepare_batch_setup(
        output_root=tmp_path / "ignored-again",
        models=["ignored-again"],
        embedding_model="ignored-again",
        replications=2,
        algorithms=("algo2",),
        config=config,
        runtime_factory=None,
        resume=False,
        dry_run=True,
    )
    assert result_again.output_root == result.output_root
    assert result_again.planned_specs == result.planned_specs


def test_initialize_batch_state_builds_fresh_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    setup = _setup(tmp_path, specs=[spec], resume=False)
    observed: dict[str, object] = {}

    def order(**kwargs: object) -> list[object]:
        observed.update(kwargs)
        run_dir_for_spec = kwargs["run_dir_for_spec_fn"]
        assert callable(run_dir_for_spec)
        assert run_dir_for_spec(tmp_path, spec) == experiments._run_dir_for_spec(
            output_root=tmp_path,
            spec=spec,
        )
        return [spec]

    monkeypatch.setattr(experiments, "_resume_order_planned_specs_for_resume", order)
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "fixed-time")

    state = experiments._initialize_batch_state(setup)

    assert state.total_runs == 1
    assert state.planned_specs == [spec]
    assert state.summary_rows == []
    assert state.persistent_sessions == {}
    assert state.seeded_finished_run_dirs == set()
    assert state.seeded_failed_run_dirs == set()
    assert observed["planned_specs"] == [spec]
    assert observed["output_root"] == tmp_path / "output"
    assert observed["resume"] is False
    assert observed["read_artifact_json_fn"] is experiments._read_artifact_json
    assert state.status_snapshot == {
        "total_runs": 1,
        "finished_count": 0,
        "failed_count": 0,
        "running_count": 0,
        "pending_count": 1,
        "failure_count": 0,
        "failures": [],
        "percent_complete": 0.0,
        "current_run": None,
        "last_completed_run": None,
        "started_at": "fixed-time",
        "updated_at": "fixed-time",
    }


def test_initialize_batch_state_uses_seeded_resume_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    setup = _setup(tmp_path, specs=[spec], resume=True)
    seeded = (
        {"total_runs": 1},
        [{"summary": True}],
        {tmp_path / "finished"},
        {tmp_path / "failed"},
    )
    order_observed: dict[str, object] = {}

    def order(**kwargs: object) -> list[object]:
        order_observed.update(kwargs)
        run_dir_for_spec = kwargs["run_dir_for_spec_fn"]
        assert callable(run_dir_for_spec)
        assert run_dir_for_spec(tmp_path, spec) == experiments._run_dir_for_spec(
            output_root=tmp_path,
            spec=spec,
        )
        return [spec]

    monkeypatch.setattr(experiments, "_resume_order_planned_specs_for_resume", order)
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "start")
    observed: dict[str, object] = {}

    def build_snapshot(**kwargs: object) -> object:
        observed.update(kwargs)
        return seeded

    monkeypatch.setattr(experiments, "_build_seeded_resume_snapshot", build_snapshot)

    state = experiments._initialize_batch_state(setup)

    assert state.total_runs == 1
    assert state.status_snapshot == {"total_runs": 1}
    assert state.summary_rows == [{"summary": True}]
    assert state.seeded_finished_run_dirs == {tmp_path / "finished"}
    assert state.seeded_failed_run_dirs == {tmp_path / "failed"}
    assert observed["output_root"] == tmp_path / "output"
    assert observed["planned_specs"] == [spec]
    assert observed["started_at"] == "start"
    assert order_observed["planned_specs"] == [spec]
    assert order_observed["output_root"] == tmp_path / "output"
    assert order_observed["resume"] is True
    assert order_observed["read_artifact_json_fn"] is experiments._read_artifact_json


@pytest.mark.parametrize(
    ("finished", "failed", "expected"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_is_seeded_run_checks_finished_and_failed_sets(
    tmp_path: Path,
    finished: bool,
    failed: bool,
    expected: bool,
) -> None:
    state = _state()
    run_dir = tmp_path / "run"
    if finished:
        state.seeded_finished_run_dirs.add(run_dir)
    if failed:
        state.seeded_failed_run_dirs.add(run_dir)

    assert experiments._is_seeded_run(state, run_dir) is expected


def test_process_planned_spec_skips_seeded_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup(tmp_path)
    state = _state()
    spec = _spec()
    run_dir = tmp_path / "nested" / "run"
    run_dir_calls: list[dict[str, object]] = []
    seeded_calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        experiments,
        "_run_dir_for_spec",
        lambda **kwargs: run_dir_calls.append(kwargs) or run_dir,
    )
    monkeypatch.setattr(
        experiments,
        "_is_seeded_run",
        lambda current_state, current_run_dir: (
            seeded_calls.append((current_state, current_run_dir)) or True
        ),
    )
    monkeypatch.setattr(
        experiments,
        "_handle_resume_run",
        lambda **_: pytest.fail("seeded runs must not resume"),
    )
    monkeypatch.setattr(
        experiments,
        "_run_fresh_spec",
        lambda **_: pytest.fail("seeded runs must not execute fresh"),
    )

    experiments._process_planned_spec(setup=setup, state=state, spec=spec)

    assert run_dir.is_dir()
    assert run_dir_calls == [{"output_root": setup.output_root, "spec": spec}]
    assert seeded_calls == [(state, run_dir)]

    experiments._process_planned_spec(setup=setup, state=state, spec=spec)
    assert run_dir.is_dir()


def test_process_planned_spec_handles_resume_before_fresh_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path)
    state = _state()
    spec = _spec()
    run_dir = tmp_path / "run"
    run_dir_calls: list[dict[str, object]] = []
    seeded_calls: list[tuple[object, Path]] = []
    resume_calls: list[dict[str, object]] = []
    fresh_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_run_dir_for_spec",
        lambda **kwargs: run_dir_calls.append(kwargs) or run_dir,
    )
    monkeypatch.setattr(
        experiments,
        "_is_seeded_run",
        lambda current_state, current_run_dir: (
            seeded_calls.append((current_state, current_run_dir)) or False
        ),
    )
    monkeypatch.setattr(
        experiments,
        "_handle_resume_run",
        lambda **kwargs: resume_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(
        experiments,
        "_run_fresh_spec",
        lambda **kwargs: fresh_calls.append(kwargs),
    )

    experiments._process_planned_spec(setup=setup, state=state, spec=spec)

    assert run_dir_calls == [{"output_root": setup.output_root, "spec": spec}]
    assert seeded_calls == [(state, run_dir)]
    assert resume_calls == [{"setup": setup, "state": state, "spec": spec, "run_dir": run_dir}]
    assert fresh_calls == []


def test_process_planned_spec_runs_fresh_when_resume_does_not_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path)
    state = _state()
    spec = _spec()
    run_dir = tmp_path / "run"
    run_dir_calls: list[dict[str, object]] = []
    seeded_calls: list[tuple[object, Path]] = []
    resume_calls: list[dict[str, object]] = []
    fresh_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_run_dir_for_spec",
        lambda **kwargs: run_dir_calls.append(kwargs) or run_dir,
    )
    monkeypatch.setattr(
        experiments,
        "_is_seeded_run",
        lambda current_state, current_run_dir: (
            seeded_calls.append((current_state, current_run_dir)) or False
        ),
    )
    monkeypatch.setattr(
        experiments,
        "_handle_resume_run",
        lambda **kwargs: resume_calls.append(kwargs) or False,
    )
    monkeypatch.setattr(
        experiments,
        "_run_fresh_spec",
        lambda **kwargs: fresh_calls.append(kwargs),
    )

    experiments._process_planned_spec(setup=setup, state=state, spec=spec)

    assert run_dir_calls == [{"output_root": setup.output_root, "spec": spec}]
    assert seeded_calls == [(state, run_dir)]
    assert resume_calls == [{"setup": setup, "state": state, "spec": spec, "run_dir": run_dir}]
    assert fresh_calls == [{"setup": setup, "state": state, "spec": spec, "run_dir": run_dir}]


def test_handle_resume_run_returns_false_without_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, resume=False)
    state = _state()
    monkeypatch.setattr(
        experiments,
        "_load_deferred_failed_summary",
        lambda **_: pytest.fail("deferred summary should not be loaded"),
    )

    assert (
        experiments._handle_resume_run(
            setup=setup,
            state=state,
            spec=_spec(),
            run_dir=tmp_path / "run",
        )
        is False
    )


def test_handle_resume_run_records_deferred_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, resume=True)
    state = _state()
    spec = _spec()
    failure = {"type": "TimeoutError", "message": "deferred"}
    run_dir = tmp_path / "run"
    normalize_calls: list[Path] = []
    deferred_load_calls: list[dict[str, object]] = []
    deferred_record_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_normalize_stale_running_run",
        lambda current_run_dir: normalize_calls.append(current_run_dir),
    )
    monkeypatch.setattr(
        experiments,
        "_load_deferred_failed_summary",
        lambda **kwargs: (
            deferred_load_calls.append(kwargs)
            or (failure if kwargs["run_dir"] == run_dir else None)
        ),
    )
    monkeypatch.setattr(
        experiments,
        "_record_deferred_failure",
        lambda **kwargs: deferred_record_calls.append(kwargs),
    )

    assert (
        experiments._handle_resume_run(
            setup=setup,
            state=state,
            spec=spec,
            run_dir=run_dir,
        )
        is True
    )
    assert normalize_calls == [run_dir]
    assert deferred_load_calls == [{"run_dir": run_dir, "context_policy": spec.context_policy}]
    assert deferred_record_calls == [{"setup": setup, "state": state, "failure": failure}]


def test_handle_resume_run_records_cached_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, resume=True)
    state = _state()
    spec = _spec()
    cached = {"status": "finished"}
    run_dir = tmp_path / "run"
    normalize_calls: list[Path] = []
    deferred_load_calls: list[dict[str, object]] = []
    cached_load_calls: list[dict[str, object]] = []
    cached_record_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_normalize_stale_running_run",
        lambda current_run_dir: normalize_calls.append(current_run_dir),
    )
    monkeypatch.setattr(
        experiments,
        "_load_deferred_failed_summary",
        lambda **kwargs: deferred_load_calls.append(kwargs) or None,
    )
    monkeypatch.setattr(
        experiments,
        "_load_valid_finished_summary",
        lambda **kwargs: cached_load_calls.append(kwargs) or cached,
    )
    monkeypatch.setattr(
        experiments,
        "_record_cached_success",
        lambda **kwargs: cached_record_calls.append(kwargs),
    )

    assert (
        experiments._handle_resume_run(
            setup=setup,
            state=state,
            spec=spec,
            run_dir=run_dir,
        )
        is True
    )
    assert normalize_calls == [run_dir]
    assert deferred_load_calls == [{"run_dir": run_dir, "context_policy": spec.context_policy}]
    assert cached_load_calls == [{"run_dir": run_dir, "algorithm": spec.algorithm}]
    assert cached_record_calls == [
        {"setup": setup, "state": state, "spec": spec, "summary": cached}
    ]


def test_handle_resume_run_returns_false_for_pending_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, resume=True)
    state = _state()
    spec = _spec()
    run_dir = tmp_path / "run"
    normalize_calls: list[Path] = []
    deferred_load_calls: list[dict[str, object]] = []
    cached_load_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_normalize_stale_running_run",
        lambda current_run_dir: normalize_calls.append(current_run_dir),
    )
    monkeypatch.setattr(
        experiments,
        "_load_deferred_failed_summary",
        lambda **kwargs: deferred_load_calls.append(kwargs) or None,
    )
    monkeypatch.setattr(
        experiments,
        "_load_valid_finished_summary",
        lambda **kwargs: cached_load_calls.append(kwargs) or None,
    )

    assert (
        experiments._handle_resume_run(
            setup=setup,
            state=state,
            spec=spec,
            run_dir=run_dir,
        )
        is False
    )
    assert normalize_calls == [run_dir]
    assert deferred_load_calls == [{"run_dir": run_dir, "context_policy": spec.context_policy}]
    assert cached_load_calls == [{"run_dir": run_dir, "algorithm": spec.algorithm}]


def test_record_deferred_failure_updates_counts_and_failure_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path)
    state = _state()
    state.status_snapshot["failed_count"] = 1
    state.status_snapshot["pending_count"] = 3
    failure = {"type": "TimeoutError"}
    written: list[dict[str, object]] = []
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "updated")
    monkeypatch.setattr(
        experiments,
        "_write_status_snapshot",
        lambda *, output_root, status: written.append(
            {"root": output_root, "status": status.copy()}
        ),
    )

    experiments._record_deferred_failure(setup=setup, state=state, failure=failure)

    assert state.status_snapshot["failed_count"] == 2
    assert state.status_snapshot["pending_count"] == 2
    assert state.status_snapshot["failures"] == [failure]
    assert state.status_snapshot["failure_count"] == 1
    assert state.status_snapshot["updated_at"] == "updated"
    assert written == [{"root": tmp_path / "output", "status": state.status_snapshot}]


def test_record_cached_success_appends_summary_and_completes_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    setup = _setup(tmp_path)
    state = _state(total_runs=2, specs=[spec])
    state.status_snapshot["pending_count"] = 2
    summary = {"summary": True}
    written: list[dict[str, object]] = []
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "updated")
    monkeypatch.setattr(
        experiments,
        "_write_status_snapshot",
        lambda *, output_root, status: written.append(
            {"output_root": output_root, "status": status.copy()}
        ),
    )

    experiments._record_cached_success(setup=setup, state=state, spec=spec, summary=summary)

    assert state.summary_rows == [summary]
    assert state.status_snapshot["finished_count"] == 1
    assert state.status_snapshot["pending_count"] == 1
    assert state.status_snapshot["last_completed_run"]["model"] == "model-a"
    assert state.status_snapshot["percent_complete"] == 50.0
    assert state.status_snapshot["updated_at"] == "updated"
    assert written == [{"output_root": setup.output_root, "status": state.status_snapshot}]


def test_current_run_payload_and_completion_percent_are_deterministic() -> None:
    spec = _spec(
        graph_source="alternate",
        decoding_algorithm="beam",
        replication=3,
    )
    payload = experiments._current_run_payload_for_spec(spec)

    assert payload == {
        "algorithm": "algo1",
        "model": "model-a",
        "decoding_algorithm": "beam",
        "graph_source": "alternate",
        "pair_name": "sg1_sg2",
        "condition_bits": "001",
        "replication": 3,
    }
    state = _state(total_runs=3)
    state.status_snapshot["finished_count"] = 2
    assert experiments._completion_percent(state) == pytest.approx(66.67)


def test_run_fresh_spec_writes_artifacts_and_records_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    setup = _setup(tmp_path, dry_run=False)
    state = _state()
    run_dir = tmp_path / "run"
    runtime_result = {"raw_row": {"Result": "[]"}}
    summary = {"summary": True}
    calls: list[tuple[str, object]] = []
    execute_calls: list[dict[str, object]] = []
    validate_calls: list[dict[str, object]] = []
    artifact_calls: list[dict[str, object]] = []
    summary_calls: list[dict[str, object]] = []
    completion_calls: list[dict[str, object]] = []
    status_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments, "_clear_retry_artifacts", lambda path: calls.append(("clear", path))
    )
    monkeypatch.setattr(
        experiments,
        "_manifest_for_spec",
        lambda value: calls.append(("manifest", value)) or {"manifest": True},
    )
    monkeypatch.setattr(
        experiments,
        "_write_json",
        lambda path, payload: calls.append((f"json:{path.name}", payload)),
    )
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "now")
    monkeypatch.setattr(
        experiments,
        "_write_status_snapshot",
        lambda **kwargs: status_calls.append(kwargs) or calls.append(("status", kwargs)),
    )
    monkeypatch.setattr(
        experiments,
        "_execute_batch_spec",
        lambda **kwargs: execute_calls.append(kwargs) or runtime_result,
    )
    monkeypatch.setattr(
        experiments,
        "_validate_structural_runtime_result",
        lambda **kwargs: validate_calls.append(kwargs) or calls.append(("validate", kwargs)),
    )
    monkeypatch.setattr(
        experiments,
        "_write_run_artifacts",
        lambda **kwargs: artifact_calls.append(kwargs) or calls.append(("artifacts", kwargs)),
    )
    monkeypatch.setattr(
        experiments,
        "_build_run_summary",
        lambda **kwargs: summary_calls.append(kwargs) or summary,
    )
    monkeypatch.setattr(
        experiments,
        "_record_completed_run",
        lambda **kwargs: completion_calls.append(kwargs) or calls.append(("complete", None)),
    )

    expected_status = dict(state.status_snapshot)
    expected_status["current_run"] = experiments._current_run_payload_for_spec(spec)
    expected_status["running_count"] = 1
    expected_status["updated_at"] = "now"

    experiments._run_fresh_spec(setup=setup, state=state, spec=spec, run_dir=run_dir)

    assert status_calls == [{"output_root": setup.output_root, "status": expected_status}]
    assert any(name == "clear" and value == run_dir for name, value in calls)
    assert any(name == "manifest" and value is spec for name, value in calls)
    assert any(
        name == "json:manifest.json" and value == {"manifest": True} for name, value in calls
    )
    assert any(
        name == "json:state.json" and value == {"status": "running"} for name, value in calls
    )
    assert any(name == "json:summary.json" and value == summary for name, value in calls)
    assert validate_calls == [{"algorithm": spec.algorithm, "raw_row": runtime_result["raw_row"]}]
    assert execute_calls == [{"setup": setup, "state": state, "spec": spec, "run_dir": run_dir}]
    assert artifact_calls == [
        {
            "run_dir": run_dir,
            "spec": spec,
            "runtime_result": runtime_result,
            "raw_row": runtime_result["raw_row"],
            "raw_row_path": run_dir / "raw_row.json",
            "manifest_for_spec_fn": experiments._manifest_for_spec,
        }
    ]
    assert summary_calls == [
        {
            "spec": spec,
            "raw_row": runtime_result["raw_row"],
            "runtime_result": runtime_result,
            "raw_row_path": run_dir / "raw_row.json",
        }
    ]
    assert completion_calls == [{"setup": setup, "state": state, "spec": spec}]
    assert ("complete", None) in calls
    assert state.status_snapshot["current_run"]["model"] == "model-a"
    assert state.status_snapshot["running_count"] == 1
    assert state.summary_rows == [summary]


def test_run_fresh_spec_records_error_without_writing_success_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path)
    state = _state()
    spec = _spec()
    run_dir = tmp_path / "run"
    error = RuntimeError("failed")
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(experiments, "_clear_retry_artifacts", lambda _path: None)
    monkeypatch.setattr(experiments, "_manifest_for_spec", lambda _spec: {})
    monkeypatch.setattr(experiments, "_write_json", lambda *_args: None)
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "now")
    monkeypatch.setattr(experiments, "_write_status_snapshot", lambda **_kwargs: None)
    monkeypatch.setattr(
        experiments, "_execute_batch_spec", lambda **_: (_ for _ in ()).throw(error)
    )
    monkeypatch.setattr(
        experiments,
        "_record_batch_failure",
        lambda **kwargs: observed.append(kwargs),
    )

    experiments._run_fresh_spec(
        setup=setup,
        state=state,
        spec=spec,
        run_dir=run_dir,
    )

    assert len(observed) == 1
    assert observed[0]["setup"] is setup
    assert observed[0]["state"] is state
    assert observed[0]["spec"] is spec
    assert observed[0]["run_dir"] == run_dir
    assert observed[0]["error"] is error


def test_execute_batch_spec_selects_monitored_or_in_process_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    state = _state()
    monitored_setup = _setup(tmp_path, monitored=True)
    monitored_result = {"raw_row": {"mode": "monitored"}}
    monitored_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_run_local_hf_spec",
        lambda **kwargs: monitored_calls.append(kwargs) or monitored_result,
    )
    monitored_run_dir = tmp_path / "monitored"
    assert (
        experiments._execute_batch_spec(
            setup=monitored_setup,
            state=state,
            spec=spec,
            run_dir=monitored_run_dir,
        )
        is monitored_result
    )
    assert monitored_calls == [
        {
            "spec": spec,
            "run_dir": monitored_run_dir,
            "output_root": monitored_setup.output_root,
            "persistent_sessions": state.persistent_sessions,
        }
    ]

    missing_setup = _setup(tmp_path, runtime_factory=None, monitored=False)
    with pytest.raises(
        ValueError,
        match=r"^Missing runtime_factory for in-process execution\.$",
    ):
        experiments._execute_batch_spec(
            setup=missing_setup,
            state=state,
            spec=spec,
            run_dir=tmp_path / "missing",
        )

    in_process_setup = _setup(tmp_path, runtime_factory="factory", monitored=False)
    expected = {"raw_row": {"mode": "in-process"}}
    observed: dict[str, object] = {}

    def execute(**kwargs: object) -> object:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(experiments, "_execute_run", execute)
    assert (
        experiments._execute_batch_spec(
            setup=in_process_setup,
            state=state,
            spec=spec,
            run_dir=tmp_path / "in-process",
        )
        is expected
    )
    assert observed == {
        "spec": spec,
        "runtime_factory": "factory",
        "dry_run": False,
        "run_dir": tmp_path / "in-process",
    }


@pytest.mark.parametrize("retry", [False, True])
def test_record_batch_failure_updates_artifacts_and_retry_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry: bool,
) -> None:
    setup = _setup(tmp_path, resume=True)
    state = _state()
    spec = _spec()
    error = RuntimeError("bad")
    writes: list[tuple[Path, object]] = []
    statuses: list[dict[str, object]] = []
    classify_calls: list[dict[str, object]] = []
    retry_calls: list[dict[str, object]] = []
    state.status_snapshot["running_count"] = 1
    state.status_snapshot["current_run"] = {"active": True}
    monkeypatch.setattr(
        experiments,
        "_classify_failure_payload",
        lambda payload: classify_calls.append(payload) or "worker",
    )
    monkeypatch.setattr(
        experiments,
        "_should_keep_failure_pending_on_resume",
        lambda **kwargs: retry_calls.append(kwargs) or retry,
    )
    monkeypatch.setattr(
        experiments, "_write_json", lambda path, payload: writes.append((path, payload))
    )
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "updated")
    monkeypatch.setattr(
        experiments,
        "_write_status_snapshot",
        lambda *, output_root, status: statuses.append(
            {"output_root": output_root, "status": status.copy()}
        ),
    )

    experiments._record_batch_failure(
        setup=setup,
        state=state,
        spec=spec,
        run_dir=tmp_path / "run",
        error=error,
    )

    assert writes == [
        (
            tmp_path / "run" / "error.json",
            {"type": "RuntimeError", "message": "bad", "status": "failed"},
        ),
        (tmp_path / "run" / "state.json", {"status": "failed"}),
    ]
    assert state.status_snapshot["running_count"] == 0
    assert state.status_snapshot["current_run"] is None
    if retry:
        assert state.status_snapshot["failed_count"] == 0
        assert state.status_snapshot["pending_count"] == 1
    else:
        assert state.status_snapshot["failed_count"] == 1
        assert state.status_snapshot["pending_count"] == 0
        assert state.status_snapshot["failure_count"] == 1
        assert state.status_snapshot["failures"] == [
            {
                "run_dir": str(tmp_path / "run"),
                "message": "bad",
                "type": "RuntimeError",
            }
        ]
    assert state.status_snapshot["updated_at"] == "updated"
    expected_failure_payload = {
        "type": "RuntimeError",
        "message": "bad",
        "status": "failed",
    }
    assert classify_calls == [expected_failure_payload]
    assert retry_calls == [
        {
            "resume": True,
            "failure_kind": "worker",
            "context_policy": spec.context_policy,
        }
    ]
    assert statuses == [{"output_root": setup.output_root, "status": state.status_snapshot}]


def test_record_batch_failure_raises_infrastructure_failure_after_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiments, "_classify_failure_payload", lambda _payload: "infrastructure")
    monkeypatch.setattr(experiments, "_should_keep_failure_pending_on_resume", lambda **_: False)
    monkeypatch.setattr(experiments, "_write_json", lambda *_args: None)
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "now")
    monkeypatch.setattr(experiments, "_write_status_snapshot", lambda **_: None)

    with pytest.raises(experiments.BatchInfrastructureFailure, match="Infrastructure failure"):
        experiments._record_batch_failure(
            setup=_setup(tmp_path),
            state=_state(),
            spec=_spec(),
            run_dir=tmp_path / "run",
            error=RuntimeError("infrastructure"),
        )


def test_record_completed_run_updates_all_progress_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    setup = _setup(tmp_path)
    state = _state(total_runs=2)
    state.status_snapshot["running_count"] = 1
    state.status_snapshot["current_run"] = {"active": True}
    monkeypatch.setattr(experiments, "_status_timestamp_now", lambda: "done")
    written: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_write_status_snapshot",
        lambda *, output_root, status: written.append(
            {"output_root": output_root, "status": status.copy()}
        ),
    )

    experiments._record_completed_run(setup=setup, state=state, spec=spec)

    assert state.status_snapshot["running_count"] == 0
    assert state.status_snapshot["current_run"] is None
    assert state.status_snapshot["finished_count"] == 1
    assert state.status_snapshot["pending_count"] == 1
    assert state.status_snapshot["percent_complete"] == 50.0
    assert state.status_snapshot["updated_at"] == "done"
    assert state.status_snapshot["last_completed_run"] == experiments._current_run_payload_for_spec(
        spec
    )
    assert written == [{"output_root": setup.output_root, "status": state.status_snapshot}]


def test_finalize_batch_outputs_writes_summary_and_skips_aggregation_for_dry_run(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, dry_run=True)
    setup.output_root.mkdir(parents=True)
    state = _state()
    state.summary_rows.append({"model": "model-a", "score": 1})

    experiments._finalize_batch_outputs(setup=setup, state=state)

    assert (setup.output_root / "batch_summary.csv").read_text(encoding="utf-8") == (
        "model,score\nmodel-a,1\n"
    )


def test_finalize_batch_outputs_aggregates_nonempty_non_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, dry_run=False)
    setup.output_root.mkdir(parents=True)
    state = _state()
    state.summary_rows.append({"model": "model-a"})
    observed: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        experiments,
        "write_aggregated_outputs",
        lambda output_root, frame: observed.append((output_root, frame)),
    )

    experiments._finalize_batch_outputs(setup=setup, state=state)

    assert len(observed) == 1
    assert observed[0][0] == setup.output_root
    assert observed[0][1].to_dict("records") == [{"model": "model-a"}]


def test_finalize_batch_outputs_does_not_aggregate_empty_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, dry_run=False)
    setup.output_root.mkdir(parents=True)
    monkeypatch.setattr(
        experiments,
        "write_aggregated_outputs",
        lambda *_args: pytest.fail("empty summaries must not be aggregated"),
    )

    experiments._finalize_batch_outputs(setup=setup, state=_state())

    assert (setup.output_root / "batch_summary.csv").exists()


def test_finalize_batch_outputs_writes_without_dataframe_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, dry_run=True)
    state = _state()
    frame_calls: list[dict[str, object]] = []

    class Frame:
        empty = False

        def to_csv(self, path: Path, *, index: bool) -> None:
            frame_calls.append({"path": path, "index": index})

    frame = Frame()
    monkeypatch.setattr(
        experiments.pd,
        "DataFrame",
        SimpleNamespace(from_records=lambda rows: frame),
    )

    experiments._finalize_batch_outputs(setup=setup, state=state)

    assert frame_calls == [{"path": setup.output_root / "batch_summary.csv", "index": False}]


def test_build_seeded_resume_snapshot_forwards_all_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ({"status": True}, [], set(), set())
    observed: dict[str, object] = {}

    def build(**kwargs: object) -> object:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(experiments, "_resume_build_seeded_resume_snapshot", build)
    specs = [_spec()]

    assert (
        experiments._build_seeded_resume_snapshot(
            output_root=tmp_path,
            planned_specs=specs,
            started_at="started",
        )
        == expected
    )
    assert observed["output_root"] == tmp_path
    assert observed["planned_specs"] == specs
    assert observed["started_at"] == "started"
    run_dir_for_spec = observed["run_dir_for_spec_fn"]
    assert callable(run_dir_for_spec)
    assert run_dir_for_spec(tmp_path, specs[0]) == experiments._run_dir_for_spec(
        output_root=tmp_path,
        spec=specs[0],
    )
    assert observed["current_run_payload_fn"] is experiments._current_run_payload
    assert observed["status_timestamp_now_fn"] is experiments._status_timestamp_now
    assert (
        observed["validate_structural_runtime_result_fn"]
        is experiments._validate_structural_runtime_result
    )
    assert observed["normalize_stale_running_run_fn"] is experiments._normalize_stale_running_run
    assert observed["read_artifact_json_fn"] is experiments._read_artifact_json
    assert observed["write_json_fn"] is experiments._write_json


def test_run_single_spec_uses_false_dry_run_default_and_returns_cached_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    setup = object()
    cached = {"cached": True}
    setup_calls: list[dict[str, object]] = []
    cache_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_prepare_single_spec_setup",
        lambda **kwargs: setup_calls.append(kwargs) or setup,
    )
    monkeypatch.setattr(
        experiments,
        "_load_cached_single_spec_summary",
        lambda **kwargs: cache_calls.append(kwargs) or cached,
    )
    monkeypatch.setattr(
        experiments,
        "_prepare_single_spec_run_directory",
        lambda *_args: pytest.fail("cached runs must not be prepared again"),
    )

    assert experiments.run_single_spec(spec=spec, output_root=tmp_path) == cached
    assert setup_calls == [
        {
            "spec": spec,
            "output_root": tmp_path,
            "runtime_factory": None,
            "dry_run": False,
        }
    ]
    assert cache_calls == [{"setup": setup, "resume": False}]


def test_run_single_spec_runs_fresh_path_and_closes_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    setup = SimpleNamespace(persistent_sessions=object())
    runtime_result = {"raw_row": {"ok": True}}
    summary = {"summary": True}
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_prepare_single_spec_setup",
        lambda **kwargs: events.append(("setup", kwargs)) or setup,
    )
    monkeypatch.setattr(
        experiments,
        "_load_cached_single_spec_summary",
        lambda **kwargs: events.append(("cache", kwargs)) or None,
    )
    monkeypatch.setattr(
        experiments,
        "_prepare_single_spec_run_directory",
        lambda value: events.append(("prepare", value)),
    )
    monkeypatch.setattr(
        experiments,
        "_execute_single_spec",
        lambda value: events.append(("execute", value)) or runtime_result,
    )
    monkeypatch.setattr(
        experiments,
        "_close_persistent_sessions",
        lambda value: events.append(("close", value)),
    )
    monkeypatch.setattr(
        experiments,
        "_complete_single_spec",
        lambda **kwargs: events.append(("complete", kwargs)) or summary,
    )

    assert (
        experiments.run_single_spec(
            spec=spec,
            output_root=tmp_path,
            runtime_factory="factory",
            dry_run=True,
            resume=True,
        )
        == summary
    )
    assert events == [
        (
            "setup",
            {
                "spec": spec,
                "output_root": tmp_path,
                "runtime_factory": "factory",
                "dry_run": True,
            },
        ),
        ("cache", {"setup": setup, "resume": True}),
        ("prepare", setup),
        ("execute", setup),
        ("close", setup.persistent_sessions),
        ("complete", {"setup": setup, "runtime_result": runtime_result}),
    ]


def test_prepare_single_spec_setup_creates_run_directory_and_mode_flags(
    tmp_path: Path,
) -> None:
    spec = _spec()
    output_root = tmp_path / "missing" / "output"
    setup = experiments._prepare_single_spec_setup(
        spec=spec,
        output_root=output_root,
        runtime_factory=None,
        dry_run=False,
    )

    assert setup.spec is spec
    assert setup.output_root == output_root
    assert setup.run_dir.is_dir()
    assert setup.summary_path == setup.run_dir / "summary.json"
    assert setup.raw_row_path == setup.run_dir / "raw_row.json"
    assert setup.runtime_factory is None
    assert setup.persistent_sessions == {}
    assert setup.use_monitored_hf_subprocess is True
    assert setup.dry_run is False

    setup_again = experiments._prepare_single_spec_setup(
        spec=spec,
        output_root=output_root,
        runtime_factory="factory",
        dry_run=False,
    )
    assert setup_again.run_dir == setup.run_dir
    assert setup_again.runtime_factory == "factory"
    assert setup_again.use_monitored_hf_subprocess is False


def test_prepare_single_spec_setup_disables_monitoring_for_factory_or_dry_run(
    tmp_path: Path,
) -> None:
    for runtime_factory, dry_run in (("factory", False), (None, True)):
        setup = experiments._prepare_single_spec_setup(
            spec=_spec(),
            output_root=tmp_path / str(dry_run),
            runtime_factory=runtime_factory,
            dry_run=dry_run,
        )
        assert setup.runtime_factory is runtime_factory
        assert setup.use_monitored_hf_subprocess is False


def test_load_cached_single_spec_summary_requires_resume_and_writes_success_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory="factory",
        persistent_sessions={},
        use_monitored_hf_subprocess=False,
        dry_run=False,
    )
    cached = {"cached": True}
    calls: list[dict[str, object]] = []
    load_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments, "_normalize_stale_running_run", lambda path: calls.append({"normalize": path})
    )
    monkeypatch.setattr(
        experiments,
        "_load_valid_finished_summary",
        lambda **kwargs: load_calls.append(kwargs) or cached,
    )
    monkeypatch.setattr(
        experiments,
        "_write_smoke_verdict",
        lambda **kwargs: calls.append(kwargs),
    )

    assert experiments._load_cached_single_spec_summary(setup=setup, resume=True) == cached
    assert calls[0] == {"normalize": setup.run_dir}
    assert load_calls == [{"run_dir": setup.run_dir, "algorithm": setup.spec.algorithm}]
    assert calls[1] == {
        "output_root": tmp_path,
        "run_dir": setup.run_dir,
        "spec_identity": experiments._smoke_spec_identity(setup.spec),
        "status": "success",
        "worker_loaded_model": True,
    }


def test_load_cached_single_spec_summary_returns_none_when_not_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory="factory",
        persistent_sessions={},
        use_monitored_hf_subprocess=False,
        dry_run=False,
    )
    load_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_load_valid_finished_summary",
        lambda **kwargs: load_calls.append(kwargs) or {"cached": True},
    )
    normalize_calls: list[Path] = []
    monkeypatch.setattr(
        experiments,
        "_normalize_stale_running_run",
        lambda run_dir: normalize_calls.append(run_dir),
    )
    assert experiments._load_cached_single_spec_summary(setup=setup, resume=False) is None
    assert experiments._load_cached_single_spec_summary(setup=setup, resume=True) == {
        "cached": True
    }
    assert load_calls == [
        {"run_dir": setup.run_dir, "algorithm": setup.spec.algorithm},
        {"run_dir": setup.run_dir, "algorithm": setup.spec.algorithm},
    ]
    assert normalize_calls == [setup.run_dir]


def test_prepare_single_spec_run_directory_clears_retry_artifacts_and_marks_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory="factory",
        persistent_sessions={},
        use_monitored_hf_subprocess=False,
        dry_run=False,
    )
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        experiments, "_clear_retry_artifacts", lambda path: calls.append(("clear", path))
    )
    monkeypatch.setattr(experiments, "_manifest_for_spec", lambda spec: {"spec": spec})
    monkeypatch.setattr(
        experiments, "_write_json", lambda path, payload: calls.append((path, payload))
    )

    experiments._prepare_single_spec_run_directory(setup)

    assert calls == [
        ("clear", setup.run_dir),
        (setup.run_dir / "manifest.json", {"spec": setup.spec}),
        (setup.run_dir / "state.json", {"status": "running"}),
    ]


def test_run_paper_batch_initializes_processes_closes_and_finalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, specs=["one", "two"])
    state = _state(total_runs=2, specs=["one", "two"])
    events: list[object] = []
    prepare_calls: list[dict[str, object]] = []
    initialize_calls: list[object] = []
    process_calls: list[dict[str, object]] = []
    finalize_calls: list[dict[str, object]] = []

    class Session:
        def close(self) -> None:
            events.append("close")

    state.persistent_sessions["model"] = Session()
    monkeypatch.setattr(
        experiments,
        "_prepare_batch_setup",
        lambda **kwargs: prepare_calls.append(kwargs) or setup,
    )
    monkeypatch.setattr(
        experiments,
        "_initialize_batch_state",
        lambda value: initialize_calls.append(value) or state,
    )
    status_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_write_status_snapshot",
        lambda **kwargs: (
            status_calls.append(kwargs) or events.append(("initial-status", kwargs["status"]))
        ),
    )
    monkeypatch.setattr(
        experiments,
        "_process_planned_spec",
        lambda **kwargs: process_calls.append(kwargs) or events.append(("process", kwargs["spec"])),
    )
    monkeypatch.setattr(
        experiments,
        "_finalize_batch_outputs",
        lambda **kwargs: finalize_calls.append(kwargs) or events.append("finalize"),
    )

    experiments.run_paper_batch(
        output_root=tmp_path,
        models=["model"],
        embedding_model="embed",
        replications=1,
    )

    assert events == [
        ("initial-status", state.status_snapshot),
        ("process", "one"),
        ("process", "two"),
        "close",
        "finalize",
    ]
    assert prepare_calls == [
        {
            "output_root": tmp_path,
            "models": ["model"],
            "embedding_model": "embed",
            "replications": 1,
            "algorithms": None,
            "config": None,
            "runtime_factory": None,
            "resume": False,
            "dry_run": False,
        }
    ]
    assert initialize_calls == [setup]
    assert status_calls == [{"output_root": setup.output_root, "status": state.status_snapshot}]
    assert process_calls == [
        {"setup": setup, "state": state, "spec": "one"},
        {"setup": setup, "state": state, "spec": "two"},
    ]
    assert finalize_calls == [{"setup": setup, "state": state}]


def test_run_paper_batch_forwards_non_default_optional_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path)
    state = _state()
    observed: list[dict[str, object]] = []
    config = object()
    monkeypatch.setattr(
        experiments,
        "_prepare_batch_setup",
        lambda **kwargs: observed.append(kwargs) or setup,
    )
    monkeypatch.setattr(experiments, "_initialize_batch_state", lambda _: state)
    monkeypatch.setattr(experiments, "_write_status_snapshot", lambda **_: None)
    monkeypatch.setattr(experiments, "_process_planned_spec", lambda **_: None)
    monkeypatch.setattr(experiments, "_finalize_batch_outputs", lambda **_: None)

    experiments.run_paper_batch(
        output_root=tmp_path,
        models=["model"],
        embedding_model="embed",
        replications=2,
        algorithms=("algo2",),
        config=config,
        runtime_factory="factory",
        resume=True,
        dry_run=True,
    )

    assert observed == [
        {
            "output_root": tmp_path,
            "models": ["model"],
            "embedding_model": "embed",
            "replications": 2,
            "algorithms": ("algo2",),
            "config": config,
            "runtime_factory": "factory",
            "resume": True,
            "dry_run": True,
        }
    ]


def test_run_paper_batch_closes_sessions_when_processing_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup(tmp_path, specs=["one"])
    state = _state()
    closed: list[bool] = []

    class Session:
        def close(self) -> None:
            closed.append(True)

    state.persistent_sessions["model"] = Session()
    monkeypatch.setattr(experiments, "_prepare_batch_setup", lambda **_: setup)
    monkeypatch.setattr(experiments, "_initialize_batch_state", lambda _setup: state)
    monkeypatch.setattr(experiments, "_write_status_snapshot", lambda **_: None)
    monkeypatch.setattr(
        experiments,
        "_process_planned_spec",
        lambda **_: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    monkeypatch.setattr(
        experiments,
        "_finalize_batch_outputs",
        lambda **_: pytest.fail("finalization must not run after processing error"),
    )

    with pytest.raises(RuntimeError, match="stop"):
        experiments.run_paper_batch(
            output_root=tmp_path,
            models=["model"],
            embedding_model="embed",
            replications=1,
        )
    assert closed == [True]


def test_execute_single_spec_uses_monitored_runtime_and_validates_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory=None,
        persistent_sessions={},
        use_monitored_hf_subprocess=True,
        dry_run=False,
    )
    result = {"raw_row": {"ok": True}}
    calls: list[object] = []
    monitored_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiments,
        "_run_local_hf_spec",
        lambda **kwargs: monitored_calls.append(kwargs) or result,
    )
    monkeypatch.setattr(
        experiments,
        "_validate_structural_runtime_result",
        lambda **kwargs: calls.append(kwargs),
    )

    assert experiments._execute_single_spec(setup) is result
    assert monitored_calls == [
        {
            "spec": setup.spec,
            "run_dir": setup.run_dir,
            "output_root": setup.output_root,
            "persistent_sessions": setup.persistent_sessions,
        }
    ]
    assert calls == [{"algorithm": "algo1", "raw_row": {"ok": True}}]


def test_execute_single_spec_builds_runtime_for_in_process_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory=None,
        persistent_sessions={},
        use_monitored_hf_subprocess=False,
        dry_run=True,
    )
    runtime = object()
    factory = object()
    result = {"raw_row": {"ok": True}}
    calls: dict[str, object] = {}

    def build_runtime_factory(*, hf_token: str) -> object:
        calls["token"] = hf_token
        return runtime

    def runtime_factory_from_hf_runtime(value: object) -> object:
        calls["runtime"] = value
        return factory

    def execute_run(**kwargs: object) -> object:
        calls["execute"] = kwargs
        return result

    monkeypatch.setattr(experiments, "build_runtime_factory", build_runtime_factory)
    monkeypatch.setattr(experiments, "_resolve_hf_token", lambda: "token")
    monkeypatch.setattr(
        experiments, "_runtime_factory_from_hf_runtime", runtime_factory_from_hf_runtime
    )
    monkeypatch.setattr(experiments, "_execute_run", execute_run)

    assert experiments._execute_single_spec(setup) is result
    assert calls["token"] == "token"
    assert calls["runtime"] is runtime
    assert calls["execute"] == {
        "spec": setup.spec,
        "runtime_factory": factory,
        "dry_run": True,
        "run_dir": setup.run_dir,
    }


def test_execute_single_spec_uses_supplied_factory_for_in_process_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = object()
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory=supplied,
        persistent_sessions={},
        use_monitored_hf_subprocess=False,
        dry_run=True,
    )
    observed: dict[str, object] = {}
    expected = {"raw_row": {}}
    monkeypatch.setattr(
        experiments,
        "_execute_run",
        lambda **kwargs: observed.update(kwargs) or expected,
    )

    assert experiments._execute_single_spec(setup) is expected
    assert observed["runtime_factory"] is supplied


def test_record_single_spec_failure_writes_error_and_failure_smoke_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory="factory",
        persistent_sessions={},
        use_monitored_hf_subprocess=False,
        dry_run=False,
    )
    writes: list[tuple[Path, object]] = []
    verdicts: list[dict[str, object]] = []
    identity_calls: list[object] = []
    worker_calls: list[Path] = []
    monkeypatch.setattr(
        experiments, "_write_json", lambda path, payload: writes.append((path, payload))
    )
    monkeypatch.setattr(
        experiments,
        "_smoke_spec_identity",
        lambda value: identity_calls.append(value) or "identity",
    )
    monkeypatch.setattr(
        experiments,
        "_worker_loaded_model",
        lambda run_dir: worker_calls.append(run_dir) or False,
    )
    monkeypatch.setattr(
        experiments, "_write_smoke_verdict", lambda **kwargs: verdicts.append(kwargs)
    )

    experiments._record_single_spec_failure(setup=setup, error=ValueError("bad"))

    assert writes == [
        (
            setup.run_dir / "error.json",
            {"type": "ValueError", "message": "bad", "status": "failed"},
        ),
        (setup.run_dir / "state.json", {"status": "failed"}),
    ]
    assert verdicts == [
        {
            "output_root": tmp_path,
            "run_dir": setup.run_dir,
            "spec_identity": "identity",
            "status": "failed",
            "failure_type": "ValueError",
            "failure_message": "bad",
            "worker_loaded_model": False,
        }
    ]
    assert identity_calls == [setup.spec]
    assert worker_calls == [setup.run_dir]


def test_close_persistent_sessions_closes_every_session() -> None:
    closed: list[str] = []

    class Session:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    experiments._close_persistent_sessions({"a": Session("a"), "b": Session("b")})

    assert closed == ["a", "b"]


def test_complete_single_spec_writes_summary_and_success_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = experiments._SingleSpecSetup(
        spec=_spec(),
        output_root=tmp_path,
        run_dir=tmp_path / "run",
        summary_path=tmp_path / "run" / "summary.json",
        raw_row_path=tmp_path / "run" / "raw_row.json",
        runtime_factory="factory",
        persistent_sessions={},
        use_monitored_hf_subprocess=False,
        dry_run=False,
    )
    runtime_result = {"raw_row": {"result": "ok"}}
    summary = {"summary": True}
    calls: list[tuple[str, object]] = []
    artifact_calls: list[dict[str, object]] = []
    summary_calls: list[dict[str, object]] = []
    identity_calls: list[object] = []
    worker_calls: list[Path] = []
    monkeypatch.setattr(
        experiments,
        "_write_run_artifacts",
        lambda **kwargs: artifact_calls.append(kwargs),
    )
    monkeypatch.setattr(
        experiments,
        "_build_run_summary",
        lambda **kwargs: summary_calls.append(kwargs) or summary,
    )
    monkeypatch.setattr(
        experiments, "_write_json", lambda path, payload: calls.append((path.name, payload))
    )
    monkeypatch.setattr(
        experiments,
        "_smoke_spec_identity",
        lambda value: identity_calls.append(value) or "identity",
    )
    monkeypatch.setattr(
        experiments,
        "_worker_loaded_model",
        lambda run_dir: worker_calls.append(run_dir) or False,
    )
    monkeypatch.setattr(
        experiments, "_write_smoke_verdict", lambda **kwargs: calls.append(("smoke", kwargs))
    )

    assert experiments._complete_single_spec(setup=setup, runtime_result=runtime_result) == summary
    assert ("summary.json", summary) in calls
    assert calls[-1] == (
        "smoke",
        {
            "output_root": tmp_path,
            "run_dir": setup.run_dir,
            "spec_identity": "identity",
            "status": "success",
            "worker_loaded_model": True,
        },
    )
    assert artifact_calls == [
        {
            "run_dir": setup.run_dir,
            "spec": setup.spec,
            "runtime_result": runtime_result,
            "raw_row": runtime_result["raw_row"],
            "raw_row_path": setup.raw_row_path,
            "manifest_for_spec_fn": experiments._manifest_for_spec,
        }
    ]
    assert summary_calls == [
        {
            "spec": setup.spec,
            "raw_row": runtime_result["raw_row"],
            "runtime_result": runtime_result,
            "raw_row_path": setup.raw_row_path,
        }
    ]
    assert identity_calls == [setup.spec]
    assert worker_calls == [setup.run_dir]
