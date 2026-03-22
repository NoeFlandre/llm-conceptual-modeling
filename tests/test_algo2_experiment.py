from pathlib import Path

from llm_conceptual_modeling.algo2.experiment import (
    build_algo2_experiment_specs,
    run_algo2_experiment,
)
from llm_conceptual_modeling.algo2.mistral import Method2PromptConfig


def test_build_algo2_experiment_specs_uses_fixed_threshold_and_full_prompt_grid(
    tmp_path,
) -> None:
    specs = build_algo2_experiment_specs(
        pair_name="sg1_sg2",
        output_root=tmp_path / "runs",
        replications=5,
    )

    assert len(specs) == 160
    first_spec = specs[0]
    assert first_spec.convergence_threshold == 0.01
    assert first_spec.prompt_config == Method2PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )
    assert first_spec.output_dir == Path(tmp_path / "runs" / "algo2" / "sg1_sg2" / "rep0_cond00000")


def test_run_algo2_experiment_delegates_to_probe(monkeypatch, tmp_path) -> None:
    captured_calls: list[object] = []

    def fake_run_probe(*, spec, chat_client, embedding_client):
        captured_calls.append(
            (
                spec.run_name,
                type(chat_client).__name__,
                type(embedding_client).__name__,
            )
        )
        return {"run_name": spec.run_name, "expanded_labels": ["alpha"]}

    monkeypatch.setattr(
        "llm_conceptual_modeling.algo2.experiment.run_algo2_probe",
        fake_run_probe,
    )

    specs = build_algo2_experiment_specs(
        pair_name="sg1_sg2",
        output_root=tmp_path / "runs",
        replications=1,
    )

    class FakeChatClient:
        pass

    class FakeEmbeddingClient:
        pass

    summary_records = run_algo2_experiment(
        specs=specs[:2],
        chat_client=FakeChatClient(),  # type: ignore[arg-type]
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
    )

    assert summary_records == [
        {"run_name": specs[0].run_name, "expanded_labels": ["alpha"]},
        {"run_name": specs[1].run_name, "expanded_labels": ["alpha"]},
    ]
    assert captured_calls == [
        (specs[0].run_name, "FakeChatClient", "FakeEmbeddingClient"),
        (specs[1].run_name, "FakeChatClient", "FakeEmbeddingClient"),
    ]


def test_run_algo2_experiment_skips_failed_specs_and_continues(monkeypatch, tmp_path) -> None:
    call_order: list[str] = []

    def fake_run_probe(*, spec, chat_client, embedding_client):
        call_order.append(spec.run_name)
        if spec.run_name == "algo2_sg1_sg2_rep0_cond00000":
            raise RuntimeError("transient failure")
        return {"run_name": spec.run_name, "expanded_labels": ["alpha"]}

    monkeypatch.setattr(
        "llm_conceptual_modeling.algo2.experiment.run_algo2_probe",
        fake_run_probe,
    )

    specs = build_algo2_experiment_specs(
        pair_name="sg1_sg2",
        output_root=tmp_path / "runs",
        replications=1,
    )

    class FakeChatClient:
        pass

    class FakeEmbeddingClient:
        pass

    summary_records = run_algo2_experiment(
        specs=specs[:2],
        chat_client=FakeChatClient(),  # type: ignore[arg-type]
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
    )

    assert summary_records == [
        {"run_name": specs[1].run_name, "expanded_labels": ["alpha"]},
    ]
    assert call_order == [specs[0].run_name, specs[1].run_name]
