from importlib import util
from pathlib import Path


def _load_runner_module():
    script_path = Path("scripts/post_revision_debug/run_paper_replay.py")
    spec = util.spec_from_file_location("run_paper_replay", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_paper_replay.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_replay_plan_separates_models_and_pairs() -> None:
    module = _load_runner_module()

    replay_plan = module._build_replay_plan(  # type: ignore[attr-defined]
        models=["mistral-small-2603", "mistral-medium-2508"],
        embedding_model="mistral-embed-2312",
        output_root=Path("/tmp/replay"),
        resume=True,
    )

    assert [spec.model for spec in replay_plan] == [
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-small-2603",
        "mistral-medium-2508",
        "mistral-medium-2508",
        "mistral-medium-2508",
        "mistral-medium-2508",
        "mistral-medium-2508",
        "mistral-medium-2508",
        "mistral-medium-2508",
        "mistral-medium-2508",
        "mistral-medium-2508",
    ]
    assert [spec.algorithm for spec in replay_plan] == [
        "algo1",
        "algo1",
        "algo1",
        "algo2",
        "algo2",
        "algo2",
        "algo3",
        "algo3",
        "algo3",
        "algo1",
        "algo1",
        "algo1",
        "algo2",
        "algo2",
        "algo2",
        "algo3",
        "algo3",
        "algo3",
    ]
    assert replay_plan[0].output_root == Path("/tmp/replay/mistral-small-2603")
    assert replay_plan[9].output_root == Path("/tmp/replay/mistral-medium-2508")
    assert replay_plan[3].command[-1] == "--resume"


def test_job_key_roundtrip_supports_resume_state() -> None:
    module = _load_runner_module()

    spec = module.ReplaySpec(  # type: ignore[attr-defined]
        model="mistral-small-2603",
        algorithm="algo2",
        pair="sg1_sg2",
        output_root=Path("/tmp/replay/mistral-small-2603"),
        command=["uv", "run", "lcm", "generate", "algo2"],
    )

    job_key = module._build_job_key(spec)  # type: ignore[attr-defined]
    state = module._new_state("paper_replay_2models")  # type: ignore[attr-defined]
    state["completed_jobs"].append(job_key)

    state_path = Path("/tmp/replay/state.json")
    module._write_state(state_path, state)  # type: ignore[attr-defined]
    loaded_state = module._load_state(state_path)  # type: ignore[attr-defined]

    assert job_key in loaded_state["completed_jobs"]
